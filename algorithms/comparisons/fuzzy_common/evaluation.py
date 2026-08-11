"""Episode execution and feasibility-first evaluation for fuzzy baselines."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol, Sequence

from . import LLM_SAFE_HRL_ROOT  # noqa: F401
from hrl_mix.model_selection import (
    FeasibilityFirstModelMetrics,
    aggregate_seed_feasibility_metrics,
)
from hrl_mix.safe_metrics import (
    aggregate_safe_metric_records,
    build_episode_metric_record,
)

from .environment import FuzzyAssignmentResult, FuzzyBaselineEnv
from .protocol import FuzzyComparisonProtocol


class ComparisonPolicy(Protocol):
    """Runtime contract shared by the three baseline policies."""

    method_id: str

    def begin_episode(self, env: FuzzyBaselineEnv, *, training: bool) -> None: ...

    def assign(
        self,
        env: FuzzyBaselineEnv,
        host_state: dict,
        *,
        training: bool,
    ) -> FuzzyAssignmentResult: ...

    def observe_assignment(
        self,
        env: FuzzyBaselineEnv,
        result: FuzzyAssignmentResult,
        *,
        training: bool,
    ) -> None: ...

    def end_phase(
        self,
        env: FuzzyBaselineEnv,
        reward: float,
        info: dict,
        *,
        training: bool,
    ) -> None: ...

    def end_episode(self, env: FuzzyBaselineEnv, *, training: bool) -> None: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


@dataclass(frozen=True)
class EvaluationResult:
    records: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]
    model_selection: FeasibilityFirstModelMetrics


def make_environment(
    protocol: FuzzyComparisonProtocol,
    seed: int,
    *,
    reward_config: dict[str, Any] | None = None,
) -> FuzzyBaselineEnv:
    reward = dict(reward_config or {})
    kwargs = protocol.environment_kwargs(int(seed))
    kwargs.update(
        {
            "ddl_reward_fraction": float(reward.get("ddl_fraction", 0.75)),
            "energy_reward_fraction": float(reward.get("energy_fraction", 0.25)),
            "comparison_energy_scale": float(reward.get("energy_scale", 1e-3)),
            "comparison_tardiness_normalizer": float(
                reward.get("tardiness_normalizer", 300.0)
            ),
        }
    )
    return FuzzyBaselineEnv(**kwargs)


def run_episode(
    env: FuzzyBaselineEnv,
    policy: ComparisonPolicy,
    *,
    seed: int,
    training: bool,
    max_assignment_steps: int = 1_000_000,
) -> dict[str, Any]:
    """Run one complete dynamic scheduling episode."""
    started = time.perf_counter()
    env.reset(seed=int(seed))
    policy.begin_episode(env, training=training)
    phase_records: list[dict[str, Any]] = []
    assignment_steps = 0
    while not bool(env.done_flag):
        while True:
            host_state, available = env.get_host_state_for_next_assignment()
            if not available:
                break
            result = policy.assign(env, host_state, training=training)
            policy.observe_assignment(env, result, training=training)
            assignment_steps += 1
            if assignment_steps > int(max_assignment_steps):
                raise RuntimeError(
                    "comparison episode exceeded max_assignment_steps"
                )
        phase_reward, phase_info = env.finish_phase_and_advance()
        phase_info = dict(phase_info or {})
        phase_info["comparison_method_id"] = str(policy.method_id)
        phase_records.append(phase_info)
        policy.end_phase(
            env,
            float(phase_reward),
            phase_info,
            training=training,
        )
    policy.end_episode(env, training=training)
    elapsed = float(time.perf_counter() - started)
    record = build_episode_metric_record(
        env,
        seed=int(seed),
        scheduling_time_seconds=elapsed,
        phase_records=phase_records,
    )
    record.update(
        {
            "comparison_method_id": str(policy.method_id),
            "assignment_steps": int(assignment_steps),
            "training": bool(training),
        }
    )
    return record


def evaluate_policy(
    protocol: FuzzyComparisonProtocol,
    policy: ComparisonPolicy,
    *,
    split: str,
    reward_config: dict[str, Any] | None = None,
    max_assignment_steps: int = 1_000_000,
) -> EvaluationResult:
    key = str(split).strip().lower()
    if key in {"test", "final_test"}:
        seeds = protocol.test_seeds
    elif key == "validation":
        seeds = protocol.validation_seeds
    else:
        raise ValueError("evaluation split must be validation or final_test")
    records = []
    for seed in seeds:
        env = make_environment(protocol, seed, reward_config=reward_config)
        records.append(
            run_episode(
                env,
                policy,
                seed=seed,
                training=False,
                max_assignment_steps=max_assignment_steps,
            )
        )
    aggregate = aggregate_safe_metric_records(records)
    selection_values = aggregate_seed_feasibility_metrics(records)
    selection = FeasibilityFirstModelMetrics.from_mapping(selection_values)
    return EvaluationResult(
        records=tuple(records),
        aggregate=aggregate,
        model_selection=selection,
    )


__all__ = [
    "ComparisonPolicy",
    "EvaluationResult",
    "evaluate_policy",
    "make_environment",
    "run_episode",
]
