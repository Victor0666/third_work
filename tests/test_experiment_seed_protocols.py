"""Regression tests for the unified comparison and LLM seed protocols."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest

from algorithms.comparisons.drlea_nichgp.config import build_config
from algorithms.comparisons.fcfs import train_fcfs
from algorithms.comparisons.fcfs.policies import (
    FCFSFCFSPolicy,
    FCFSFixedPolicy,
    fcfs_task_order,
    select_vm_fcfs,
)
from algorithms.comparisons.fuzzy_common.environment import FuzzyBaselineEnv
from algorithms.llm_safe_hrl.LLM.problems.cews_task_constructive.eval import (
    _resolve_seeds,
    load_problem_config,
)
from algorithms.llm_safe_hrl.LLM.seevo import SeEvo
from base.hrl_env import HrlFcfsCacheEnv


def test_fcfs_formal_path_uses_shared_fuzzy_environment():
    source = Path(train_fcfs.__file__).read_text(encoding="utf-8")
    assert "env_fcfs" not in source
    assert issubclass(FuzzyBaselineEnv, HrlFcfsCacheEnv)
    protocol = train_fcfs.build_fcfs_protocol("SS", "T")
    assert protocol.test_seeds == (201, 202, 203)
    kwargs = protocol.environment_kwargs(201)
    assert kwargs["deadline_mode"] == "cache_fcfs"
    assert kwargs["deadline_cache_strict"] is True
    assert kwargs["fuzzy_enabled"] is True
    assert kwargs["fuzzy_resource_seed"] == 201


def test_fcfs_task_and_resource_ties_are_deterministic():
    class Workflow:
        def __init__(self, arrival_time):
            self.arrival_time = arrival_time

    env = type("TaskEnv", (), {})()
    env.task_ready_time = {8: 4.0, 3: 4.0, 5: 2.0}
    env.task_meta = [(0, 0)] * 9
    env.task_meta[8] = (0, 0)
    env.task_meta[3] = (1, 0)
    env.task_meta[5] = (1, 1)
    env.workflows = [Workflow(2.0), Workflow(1.0)]
    expected = [5, 3, 8]
    assert fcfs_task_order(env, [8, 3, 5]) == expected

    captured = []
    env.set_task_orderer = lambda orderer, training: captured.append(orderer)
    FCFSFCFSPolicy().begin_episode(env, training=False)
    FCFSFixedPolicy().begin_episode(env, training=False)
    assert captured[0]((8, 3, 5), None, False) == expected
    assert captured[1]((8, 3, 5), None, False) == expected

    class AssignmentEnv:
        vm_available_at = [2.0, 2.0, 3.0]
        vm_ids = [9, 4, 1]

        @staticmethod
        def global_vm_action_mask():
            return [1.0, 1.0, 1.0]

    assert select_vm_fcfs(AssignmentEnv()) == 1


def test_drlea_environment_seeds_do_not_depend_on_algorithm_seed():
    first = build_config("SS", "T", 7, smoke=True)
    second = build_config("SS", "T", 99, smoke=True)
    expected = (
        (1, 2, 3, 4, 5),
        (101, 102, 103),
        (201, 202, 203),
    )
    assert (
        first.train_seeds,
        first.validation_seeds,
        first.test_seeds,
    ) == expected
    assert (
        second.train_seeds,
        second.validation_seeds,
        second.test_seeds,
    ) == expected
    assert (first.algorithm_seed, second.algorithm_seed) == (7, 99)


def test_llm_safe_default_split_and_direct_evaluator_guard():
    config = load_problem_config()
    assert config["dataset"]["train_seeds"] == [1, 2, 3]
    assert config["dataset"]["validation_seeds"] == [4, 5]
    assert _resolve_seeds(config, "train", None) == [1, 2, 3]
    with pytest.raises(ValueError, match="reserved comparison"):
        _resolve_seeds(config, "train", [101])
    with pytest.raises(ValueError, match="reserved comparison"):
        _resolve_seeds(config, "train", [203])


def test_seevo_rejects_reserved_seed_override_before_cma():
    algorithm = object.__new__(SeEvo)
    algorithm.cfg = OmegaConf.create(
        {
            "problem": {
                "dataset": {
                    "train_seeds": [1, 2, 3],
                    "validation_seeds": [4, 5],
                    "test_seeds": [100],
                }
            }
        }
    )
    algorithm.case_num = [201]
    with pytest.raises(ValueError, match="reserved comparison"):
        algorithm._optimization_seed_sets()
