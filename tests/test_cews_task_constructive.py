"""cews_task_constructive 的接口、确定性、并发隔离和结果协议回归测试。

多数测试使用最小桩环境，只验证本问题新加的边界，不重复执行昂贵的完整仿真；
完整 DAX/Host/VM/能耗链路由纯评价命令另行覆盖。
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT

for import_root in (str(PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from base.hrl_env import (
    HrlHeftEnv,
    NoFeasibleVMError,
    validate_task_priority_scores,
)
from common.resource_opt import create_cluster
from problems.cews_task_constructive.eval import (
    _add_objective_and_constraints,
    load_priority_function,
)
# from seevo import SeEvo, individual_comparison_key, parse_result_json
from seevo import (
    SeEvo,
    individual_comparison_key,
    individual_performance_summary,
    parse_result_json,
)
from utils.utils import block_until_running


FEATURE_NAMES = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)


def _priority_only_environment(feature_count: int):
    """构造只含任务选择所需接口的轻量环境，隔离 VM 与事件逻辑。"""
    # 绕过庞大的真实环境构造器，直接复用待测类方法；所有特征长度保持一致。
    environment = object.__new__(HrlHeftEnv)
    environment._task_id = lambda task: int(task)
    environment.build_task_features = lambda ready: {
        name: np.arange(feature_count, dtype=float) for name in FEATURE_NAMES
    }
    return environment


def _vm_policy_environment(deadline: float):
    """构造一个 ready task、两台 VM 的最小确定性 VM 选择环境。"""
    environment = object.__new__(HrlHeftEnv)
    environment.task_state = ["Ready"]
    environment.task_meta = [(0, 0)]
    environment.task_baseline_finish = [float(deadline)]
    environment.current_time = 0.0
    environment.vm_ids = [0, 1]
    environment.vms = {
        0: SimpleNamespace(vm_id=0),
        1: SimpleNamespace(vm_id=1),
    }
    environment.vm_available_at = np.array([0.0, 0.0], dtype=float)
    environment.get_feasible_vms = lambda task: [0, 1]
    return environment


class TaskPriorityTests(unittest.TestCase):
    """验证候选分数接口、较小分数语义和空 ready 集事件推进。"""

    def test_single_ready_task(self):
        """N=1 时仍返回唯一任务，不能因 squeeze/标量处理破坏形状。"""
        environment = _priority_only_environment(1)
        selected = environment.select_task_with_priority_rule(
            [17], lambda *features: np.array([3.0])
        )
        self.assertEqual(selected, 17)

    def test_multiple_ready_tasks(self):
        """多个 ready tasks 中应选择分数最小项对应的原任务 ID。"""
        environment = _priority_only_environment(3)
        selected = environment.select_task_with_priority_rule(
            [5, 6, 7], lambda *features: np.array([2.0, -1.0, 0.0])
        )
        self.assertEqual(selected, 6)

    def test_wrong_priority_length_is_rejected(self):
        """候选返回长度与 ready 数不一致时必须立即报错，不能静默截断。"""
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_task_priority_scores([1.0], 2)

    def test_nan_priority_is_rejected(self):
        """NaN 不能进入 np.argmin，否则任务选择语义会依赖 NumPy 异常行为。"""
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate_task_priority_scores([0.0, np.nan], 2)

    def test_no_ready_task_advances_event(self):
        """空 ready 集不调用 LLM，而应委托现有环境推进下一离散事件。"""
        environment = object.__new__(HrlHeftEnv)
        environment._advance_until_decision_energy_only = mock.Mock(return_value=-0.5)
        reward = environment.advance_to_next_event()
        self.assertEqual(reward, -0.5)
        environment._advance_until_decision_energy_only.assert_called_once_with()


class DeterministicVMPolicyTests(unittest.TestCase):
    """验证 LLM 选任务后固定 VM 策略的硬约束与排序优先级。"""

    def test_all_vms_infeasible(self):
        """没有任何可行 VM 时抛出专用异常，不能制造虚假 VM 或丢弃任务。"""
        environment = _vm_policy_environment(deadline=10.0)
        environment.get_feasible_vms = lambda task: []
        with self.assertRaises(NoFeasibleVMError):
            environment.select_vm_deterministic(0)

    def test_on_time_vm_uses_energy_then_finish(self):
        """存在按时 VM 时能耗优先，即使低能耗 VM 的执行时间略长。"""
        environment = _vm_policy_environment(deadline=10.0)
        exec_time = {0: 4.0, 1: 5.0}
        energy = {0: 10.0, 1: 2.0}
        environment.estimate_exec_time = lambda task, vm: exec_time[int(vm)]
        environment.estimate_comm_time = lambda task, vm: 0.0
        environment.estimate_incremental_energy = lambda task, vm: energy[int(vm)]
        vm_id, details = environment.select_vm_deterministic(0)
        self.assertEqual(vm_id, 1)
        self.assertEqual(details["deadline_violation"], 0.0)

    def test_all_late_vms_use_violation_before_energy(self):
        """全部延期时先最小化违反量，不能让极低能耗掩盖更严重延期。"""
        environment = _vm_policy_environment(deadline=3.0)
        exec_time = {0: 7.0, 1: 5.0}
        energy = {0: 1.0, 1: 100.0}
        environment.estimate_exec_time = lambda task, vm: exec_time[int(vm)]
        environment.estimate_comm_time = lambda task, vm: 0.0
        environment.estimate_incremental_energy = lambda task, vm: energy[int(vm)]
        vm_id, details = environment.select_vm_deterministic(0)
        self.assertEqual(vm_id, 1)
        self.assertEqual(details["deadline_violation"], 2.0)

    def test_cloud_and_edge_vms_exist(self):
        """真实资源构造器能同时提供 cloud 与 edge VM，覆盖两类分配路径。"""
        hosts, vms = create_cluster(
            num_cloud_hosts=1,
            num_edge_hosts=1,
            cloud_vms_per_host=(1,),
            edge_vms_per_host=(1,),
        )
        server_types = {hosts[vm.host_id].server_type for vm in vms.values()}
        self.assertEqual(server_types, {"cloud", "edge"})


class CandidateIsolationTests(unittest.TestCase):
    """验证并发候选模块不会共享文件内容或 Python 模块缓存。"""

    def test_two_candidates_load_concurrently_without_cross_read(self):
        """并发反复加载两个常量不同的规则，结果必须始终与各自文件对应。"""
        template = """import numpy as np
def get_task_priority_v2(a, b, c, d, e, f, g, h):
    return np.full(np.asarray(a).shape, {value}.0)
"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            first = Path(directory) / "candidate_iter1_ind0.py"
            second = Path(directory) / "candidate_iter1_ind1.py"
            first.write_text(template.format(value=11), encoding="utf-8")
            second.write_text(template.format(value=22), encoding="utf-8")

            def load_value(path):
                function = load_priority_function(path)
                inputs = [np.array([1.0, 2.0])] * 8
                return float(function(*inputs)[0])

            # 交错提交同一路径，增加模块缓存或共享文件竞态被触发的机会。
            paths = [first, second] * 10
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                values = list(executor.map(load_value, paths))
            self.assertEqual(values[0::2], [11.0] * 10)
            self.assertEqual(values[1::2], [22.0] * 10)

    def test_seevo_writes_distinct_candidate_modules(self):
        """SeEvo 应按迭代号/个体号写独立文件，并把对应路径传给 eval.py。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root = Path(directory)
            problem_dir = root / "problems" / "cews_task_constructive"
            generated_dir = problem_dir / "generated"
            generated_dir.mkdir(parents=True)
            (problem_dir / "eval.py").write_text("", encoding="utf-8")
            config_dir = root / "cfg" / "problem"
            config_dir.mkdir(parents=True)
            (config_dir / "cews_task_constructive.yaml").write_text("{}", encoding="utf-8")
            effective_config = root / "effective_cews_task_constructive_config.yaml"
            effective_config.write_text(
                "problem_size: 5\ndataset:\n  workflows_per_instance: 5\n",
                encoding="utf-8",
            )

            algorithm = object.__new__(SeEvo)
            algorithm.iteration = 4
            algorithm.generated_dir = str(generated_dir)
            algorithm.problem_dir = str(problem_dir)
            algorithm.root_dir = str(root)
            algorithm.problem = "cews_task_constructive"
            algorithm.mode = "train"
            algorithm.effective_problem_config_path = str(effective_config)
            individuals = [
                {
                    "code": "VALUE = 1",
                    "stdout_filepath": str(root / "stdout0.txt"),
                    "response_id": 0,
                },
                {
                    "code": "VALUE = 2",
                    "stdout_filepath": str(root / "stdout1.txt"),
                    "response_id": 1,
                },
            ]

            # 不真正启动评价子进程，只检查生成文件和 Popen 命令参数。
            with mock.patch("seevo.subprocess.Popen") as popen, mock.patch(
                "seevo.block_until_running"
            ):
                popen.return_value = SimpleNamespace()
                algorithm._run_code(individuals[0], 0, [0])
                algorithm._run_code(individuals[1], 1, [0])

            first_path = Path(individuals[0]["code_path"])
            second_path = Path(individuals[1]["code_path"])
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.read_text(encoding="utf-8").strip(), "VALUE = 1")
            self.assertEqual(second_path.read_text(encoding="utf-8").strip(), "VALUE = 2")
            commands = [call.args[0] for call in popen.call_args_list]
            self.assertIn(str(first_path), commands[0])
            self.assertIn(str(second_path), commands[1])
            self.assertIn(str(effective_config), commands[0])
            self.assertIn(str(effective_config), commands[1])
            # Windows 下必须显式强制评价子进程使用 UTF-8，否则中文绝对路径可能
            # 按 GBK 写入 stdout，导致父进程监视日志时解码失败。
            for call in popen.call_args_list:
                self.assertEqual(call.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
                self.assertEqual(call.kwargs["env"]["PYTHONUTF8"], "1")

    def test_startup_monitor_tolerates_legacy_gbk_log(self):
        """启动监视器应能读取旧 GBK 日志，不能把编码问题误判成 seed 无效。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            stdout_path = Path(directory) / "legacy_gbk_stdout.txt"
            stdout_path.write_bytes(
                "[cews-eval] 候选评价已经启动".encode("gbk")
            )
            block_until_running(str(stdout_path))


class ResultProtocolTests(unittest.TestCase):
    """验证 RESULT_JSON 定位、完整指标保存及无效候选惩罚。"""

    def test_result_json_is_parsed_from_last_matching_line(self):
        """stdout 含多条结果时必须选择最后一条，而不是依赖固定行号。"""
        payload = {"objective": 1.25, "energy": 100.0, "violation_rate": 0.1}
        stdout = "log line\nRESULT_JSON={\"objective\": 99}\nmore\nRESULT_JSON=" + json.dumps(payload)
        self.assertEqual(parse_result_json(stdout), payload)

    def test_evaluate_population_stores_metrics(self):
        """合法结果应同时写入用于选择的 obj 和用于分析的完整 metrics。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            stdout_path = Path(directory) / "valid.txt"
            stdout_path.write_text(
                'progress\nRESULT_JSON={"objective": 2.5, "energy": 7.0}\n',
                encoding="utf-8",
            )
            algorithm = object.__new__(SeEvo)
            algorithm.iteration = 0
            algorithm.cfg = SimpleNamespace(timeout=1)
            algorithm.obj_type = "min"
            # 用假进程复用真实 evaluate_population 解析逻辑，避免启动 Python 子进程。
            algorithm._run_code = lambda individual, response_id, cases: SimpleNamespace(
                communicate=lambda timeout: (None, None),
                kill=lambda: None,
            )
            population = [{
                "code": "def get_task_priority_v2(*args): return [0]",
                "stdout_filepath": str(stdout_path),
                "response_id": 0,
            }]
            result = algorithm.evaluate_population(population, [0])
            self.assertTrue(result[0]["exec_success"])
            self.assertEqual(result[0]["obj"], 2.5)
            self.assertEqual(result[0]["metrics"]["energy"], 7.0)

    def test_individual_performance_summary_contains_fuzzy_components(self):
        """反思摘要应包含模糊能耗分解、约束及跨 seed 诊断信息。"""
        individual = {
            "obj": 120.0,
            "metrics": {
                "constraint_feasible": True,
                "fuzzy_total_energy_score": 120.0,
                "fuzzy_total_energy_mean": 110.0,
                "fuzzy_total_energy_std": 10.0,
                "total_energy": 105.0,
                "deadline_violation_rate": 0.0,
                "total_lateness": 0.0,
                "objective_std_across_seeds": 3.0,
                "objective_max_across_seeds": 125.0,
                "feasible_seed_rate": 1.0,
            },
        }

        text = individual_performance_summary(individual)

        self.assertIn(
            "fuzzy_energy_score=120.0000",
            text,
        )
        self.assertIn(
            "fuzzy_energy_mean=110.0000",
            text,
        )
        self.assertIn(
            "fuzzy_energy_std=10.0000",
            text,
        )
        self.assertIn(
            "modal_energy=105.0000",
            text,
        )
        self.assertIn(
            "cross_seed_objective_std=3.0000",
            text,
        )
        self.assertIn(
            "worst_seed_objective=125.0000",
            text,
        )
        self.assertIn(
            "feasible_seed_rate=1.0000",
            text,
        )
        self.assertIn(
            "DDL_feasible=True",
            text,
        )

    def test_bad_candidate_output_is_marked_infinite(self):
        """缺少 RESULT_JSON 的候选必须 exec_success=False 且 obj=正无穷。"""
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            stdout_path = Path(directory) / "invalid.txt"
            stdout_path.write_text("candidate failed without result\n", encoding="utf-8")
            algorithm = object.__new__(SeEvo)
            algorithm.iteration = 0
            algorithm.cfg = SimpleNamespace(timeout=1)
            algorithm.obj_type = "min"
            algorithm._run_code = lambda individual, response_id, cases: SimpleNamespace(
                communicate=lambda timeout: (None, None),
                kill=lambda: None,
            )
            population = [{
                "code": "this is invalid candidate code",
                "stdout_filepath": str(stdout_path),
                "response_id": 0,
            }]
            result = algorithm.evaluate_population(population, [0])
            self.assertFalse(result[0]["exec_success"])
            self.assertEqual(result[0]["obj"], float("inf"))


class ConstrainedEnergyObjectiveTests(unittest.TestCase):
    """验证 objective 是纯能耗，并由独立比较键实施 DDL 约束。"""

    @staticmethod
    def _metrics(energy, violation_rate, total_lateness):
        return {
            "total_energy": float(energy),
            "deadline_violation_rate": float(violation_rate),
            "total_lateness": float(total_lateness),
        }

    def test_objective_equals_total_energy_without_deadline_penalty(self):
        """即使存在延期，objective 也必须严格等于焦耳能耗，不能加权 DDL。"""
        config = {
            "objective": {"metric": "total_energy"},
            "constraints": {"deadline_violation_rate_max": 0.0},
        }
        result = _add_objective_and_constraints(
            self._metrics(energy=123.5, violation_rate=0.4, total_lateness=80.0),
            config,
        )
        self.assertEqual(result["objective"], 123.5)
        self.assertFalse(result["constraint_feasible"])
        self.assertEqual(result["constraint_violation"], 0.4)

    def test_feasible_rule_beats_lower_energy_infeasible_rule(self):
        """可行性是硬约束：可行规则能耗较高时仍必须排在违约规则之前。"""
        feasible = {
            "obj": 200.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": True,
                "deadline_violation_rate": 0.0,
                "total_lateness": 0.0,
            },
        }
        infeasible = {
            "obj": 100.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": False,
                "constraint_violation": 0.1,
                "constraint_secondary_violation": 1.0,
                "deadline_violation_rate": 0.1,
                "total_lateness": 1.0,
            },
        }
        self.assertLess(
            individual_comparison_key(feasible),
            individual_comparison_key(infeasible),
        )

    def test_infeasible_rules_reduce_violation_before_energy(self):
        """均不可行时先降低违反率和总延期，最后才用纯能耗打破平局。"""
        fewer_violations = {
            "obj": 300.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": False,
                "constraint_violation": 0.1,
                "constraint_secondary_violation": 50.0,
            },
        }
        lower_energy_but_more_violations = {
            "obj": 10.0,
            "exec_success": True,
            "metrics": {
                "constraint_feasible": False,
                "constraint_violation": 0.2,
                "constraint_secondary_violation": 1.0,
            },
        }
        self.assertLess(
            individual_comparison_key(fewer_violations),
            individual_comparison_key(lower_energy_but_more_violations),
        )


if __name__ == "__main__":
    unittest.main()
