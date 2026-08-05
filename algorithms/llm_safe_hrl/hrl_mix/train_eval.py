# -*- coding: utf-8 -*-
"""
训练评估模块。

整体调用关系：
1. train.py 调用 train_runner.train()。
2. train_runner.py 在每个 episode 结束时调用 evaluate_hrl_three_layer_multi_seed()。
3. evaluate_hrl_three_layer_multi_seed() 使用当前训练好的 VM、Host、Manager agent，
   在指定 eval_seeds 上执行确定性评估。
4. 评估结果返回给 train_runner.py，由 train_runner.py 写入日志并判断是否保存 best checkpoint。

文件职责：
- 只负责评估，不负责训练参数解析、环境创建配置、模型保存或日志字段定义。
- 评估时使用 deterministic=True，尽量反映当前策略本身的效果，而不是探索噪声。
- 多 seed 结果取平均，返回 VM reward、Host reward、Manager reward 和总能耗。
"""
from __future__ import annotations

import time

import numpy as np

from hrl_mix.safe_metrics import (
    SafeMetricStore,
    aggregate_safe_metric_records,
    build_episode_metric_record,
)
from hrl_mix.train_utils import (
    manager_apply_action,
    select_layer_action,
    sync_env_scales,
)

# 训练 cost_budget 可以为稳定性临时设为非零，但评估合格标准不可继承该
# 容忍度。这里固定为零完成工作流 DDL 违反。
FINAL_EVALUATION_VIOLATION_BUDGET = 0.0


def _completed_fuzzy_lateness_summary(eval_env):
    """Reconstruct exact fuzzy lateness from the environment timelines.

    Production environments expose ``wf_finish_time``,
    ``_workflow_finish_tfn`` and ``fuzzy_deadline_measure``. The fallback is
    retained only for lightweight legacy test doubles that expose cumulative
    safety counters but not fuzzy timelines.
    """
    finish_ids = sorted(
        int(workflow_id)
        for workflow_id in getattr(
            eval_env,
            "wf_finish_time",
            {},
        )
    )
    finish_getter = getattr(
        eval_env,
        "_workflow_finish_tfn",
        None,
    )
    risk_measure = getattr(
        eval_env,
        "fuzzy_deadline_measure",
        None,
    )
    workflows = getattr(eval_env, "workflows", None)
    if (
        finish_ids
        and callable(finish_getter)
        and callable(risk_measure)
        and workflows is not None
    ):
        values = []
        for workflow_id in finish_ids:
            finish_tfn = finish_getter(workflow_id)
            risk_finish = float(risk_measure(finish_tfn))
            deadline = float(workflows[workflow_id].deadline)
            values.append(max(0.0, risk_finish - deadline))
        return {
            "fuzzy_lateness_sum": float(sum(values)),
            "max_fuzzy_lateness": float(
                max(values, default=0.0)
            ),
            "fuzzy_lateness_values": values,
            "exact_fuzzy_timeline_reconstruction": True,
        }

    cumulative = max(
        0.0,
        float(
            getattr(
                eval_env,
                "_safety_cumulative_fuzzy_lateness_cost",
                0.0,
            )
        ),
    )
    return {
        "fuzzy_lateness_sum": cumulative,
        # With no per-workflow timeline the cumulative value is the only
        # conservative upper bound available to a compatibility test double.
        "max_fuzzy_lateness": cumulative,
        "fuzzy_lateness_values": [],
        "exact_fuzzy_timeline_reconstruction": False,
    }


def evaluate_hrl_three_layer_multi_seed(
    env_cls,
    env_kwargs,
    vm_agent,
    host_agent,
    manager_agent,
    seeds,
    *,
    return_safety_metrics=False,
):
    """在多个随机种子上评估当前三层 HRL 策略。

    参数：
    - env_cls：环境类，一般为 CloudWorkflowEnv_VMAgents。
    - env_kwargs：创建环境所需参数，同时包含部分需要手动同步的尺度参数。
    - vm_agent、host_agent、manager_agent：当前训练中的三个智能体。
    - seeds：评估使用的随机种子列表。

    返回：
    - eval_vm：各评估 seed 上 VM 层平均 reward 的均值。
    - eval_host：各评估 seed 上 Host 层平均 reward 的均值。
    - eval_mgr：各评估 seed 上 Manager 层平均 reward 的均值。
    - eval_energy：各评估 seed 上总能耗的均值。
    - return_safety_metrics=True 时额外返回安全字典；其中合格线固定为
      deadline violation rate == 0，不继承训练 cost_budget。
    """
    # 这些尺度参数不是环境构造函数的入参，需要环境创建后手动同步。
    ctor_block = {
        "energy_reward_scale",
        "task_baseline_norm",
        "energy_norm_per_mi_ref",
        "alpha_delay_host",
        "alpha_delay_vm",
    }
    base_ctor_kwargs = {k: v for k, v in env_kwargs.items() if k not in ctor_block}

    vm_list, host_list, mgr_list, energy_list = [], [], [], []
    seed_metric_records = []

    seeds = tuple(seeds)
    if not seeds:
        raise ValueError(
            "multi-seed evaluation requires at least one seed"
        )

    for sd in seeds:
        seed_started_at = time.perf_counter()
        phase_metric_records = []
        # 每个 seed 创建一个独立评估环境，避免评估过程互相污染状态。
        ctor_kwargs = dict(base_ctor_kwargs)
        ctor_kwargs["random_seed"] = int(sd)

        eval_env = env_cls(**ctor_kwargs)
        sync_env_scales(eval_env, env_kwargs)

        eval_env.reset()

        # 评估时 manager 先选择一个阶段动作，后续进入任务分配循环。
        sH = eval_env.get_manager_state()
        m_mask = eval_env.get_manager_action_mask()
        m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
        manager_apply_action(eval_env, m_act)

        done = bool(getattr(eval_env, "done_flag", False))
        phases = 0
        ret_mgr = 0.0
        ret_vm_phase_mean = 0.0
        ret_host_phase_mean = 0.0

        while not done:
            vm_rewards = []
            host_rewards = []

            # 一个 phase 内不断让 HostAgent 选 host，再让 VMAgent 选 VM 槽位。
            while True:
                st_host, has_next = eval_env.get_host_state_for_next_assignment()
                if not has_next:
                    break

                a_host, _, _ = select_layer_action(
                    host_agent,
                    st_host,
                    safe_rl_enabled=bool(
                        getattr(
                            eval_env,
                            "safe_rl_enabled",
                            False,
                        )
                    ),
                    deterministic=True,
                    count_step=False,
                )
                eval_env.host_select(int(a_host))

                st_vm, ok_vm = eval_env.get_vm_state_for_current_task()
                if not ok_vm:
                    break

                a_vm, _, _ = select_layer_action(
                    vm_agent,
                    st_vm,
                    safe_rl_enabled=bool(
                        getattr(
                            eval_env,
                            "safe_rl_enabled",
                            False,
                        )
                    ),
                    deterministic=True,
                    count_step=False,
                )
                r_host, r_vm, info_task = eval_env.vm_assign(int(a_vm))

                if getattr(eval_env, "safe_rl_enabled", False):
                    host_rewards.append(
                        float(
                            info_task.get(
                                "total_performance_reward",
                                info_task.get(
                                    "performance_reward_host",
                                    r_host,
                                ),
                            )
                        )
                    )
                    vm_rewards.append(
                        float(
                            info_task.get(
                                "total_performance_reward",
                                info_task.get(
                                    "performance_reward_vm",
                                    r_vm,
                                ),
                            )
                        )
                    )
                else:
                    host_rewards.append(float(r_host))
                    vm_rewards.append(float(r_vm))

            # phase 结束后由环境推进时间，并得到 manager 层 reward。
            r_manager_raw, pinfo = eval_env.finish_phase_and_advance()
            if return_safety_metrics:
                phase_metric_records.append(dict(pinfo))
            if getattr(eval_env, "safe_rl_enabled", False):
                ret_mgr += float(
                    pinfo.get(
                        "total_performance_reward",
                        pinfo.get(
                            "performance_reward",
                            r_manager_raw,
                        ),
                    )
                )
            else:
                ret_mgr += float(r_manager_raw)
            ret_vm_phase_mean += float(np.mean(vm_rewards)) if len(vm_rewards) > 0 else 0.0
            ret_host_phase_mean += float(np.mean(host_rewards)) if len(host_rewards) > 0 else 0.0
            phases += 1

            done = bool(getattr(eval_env, "done_flag", False))
            if done:
                break

            # 如果 episode 未结束，manager 为下一 phase 选择新动作。
            sH = eval_env.get_manager_state()
            m_mask = eval_env.get_manager_action_mask()
            m_act = manager_agent.select_action(sH, m_mask, deterministic=True, count_step=False)
            manager_apply_action(eval_env, m_act)

        avg_vm = ret_vm_phase_mean / max(phases, 1)
        avg_host = ret_host_phase_mean / max(phases, 1)
        avg_mgr = ret_mgr / max(phases, 1)
        total_energy = float(eval_env.total_energy)

        vm_list.append(avg_vm)
        host_list.append(avg_host)
        mgr_list.append(avg_mgr)
        energy_list.append(total_energy)
        if return_safety_metrics:
            seed_metric_records.append(
                build_episode_metric_record(
                    eval_env,
                    seed=int(sd),
                    scheduling_time_seconds=(
                        time.perf_counter() - seed_started_at
                    ),
                    phase_records=phase_metric_records,
                )
            )

    base_result = (
        float(np.mean(vm_list)),
        float(np.mean(host_list)),
        float(np.mean(mgr_list)),
        float(np.mean(energy_list)),
    )
    if not return_safety_metrics:
        return base_result

    safety_metrics = aggregate_safe_metric_records(
        seed_metric_records
    )
    safety_metrics.update(
        {
            "evaluation_violation_budget": (
                FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "zero_violation_pass": bool(
                safety_metrics["fuzzy_ddl_violation_rate"]
                <= FINAL_EVALUATION_VIOLATION_BUDGET
            ),
        }
    )
    return (*base_result, safety_metrics)


def evaluate_and_save_safe_hrl_final_test(
    env_cls,
    env_kwargs,
    vm_agent,
    host_agent,
    manager_agent,
    *,
    training_seeds,
    validation_seeds,
    final_test_seeds,
    metrics_output_directory,
    convergence_window=5,
    q_c_prediction_error=0.0,
    q_c_prediction_error_sample_count=0,
    lambda_current=0.0,
):
    """Run an explicit withheld-seed final test and persist its report.

    The online trainer intentionally never calls this function. Final-test
    seeds must be non-empty and disjoint from both training and validation.
    """

    split = {
        "training": tuple(int(seed) for seed in training_seeds),
        "validation": tuple(int(seed) for seed in validation_seeds),
        "final_test": tuple(int(seed) for seed in final_test_seeds),
    }
    if not split["final_test"]:
        raise ValueError("final_test_seeds must be non-empty")
    for first, second in (
        ("training", "validation"),
        ("training", "final_test"),
        ("validation", "final_test"),
    ):
        overlap = sorted(
            set(split[first]).intersection(split[second])
        )
        if overlap:
            raise ValueError(
                f"{first} and {second} seeds overlap: {overlap}"
            )

    result = evaluate_hrl_three_layer_multi_seed(
        env_cls,
        env_kwargs,
        vm_agent,
        host_agent,
        manager_agent,
        split["final_test"],
        return_safety_metrics=True,
    )
    report = aggregate_safe_metric_records(
        result[-1]["per_seed_metrics"],
        q_c_prediction_error=q_c_prediction_error,
        q_c_prediction_error_sample_count=(
            q_c_prediction_error_sample_count
        ),
        lambda_current=lambda_current,
    )
    report.update(
        {
            "evaluation_violation_budget": (
                FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "zero_violation_pass": bool(
                report["fuzzy_ddl_violation_rate"]
                <= FINAL_EVALUATION_VIOLATION_BUDGET
            ),
            "seed_split": {
                key: list(value)
                for key, value in split.items()
            },
        }
    )
    store = SafeMetricStore(
        metrics_output_directory,
        convergence_window=convergence_window,
    )
    persisted = store.append(
        "final_test",
        report,
        global_step=None,
        episode=None,
    )
    return (*result[:-1], persisted)
