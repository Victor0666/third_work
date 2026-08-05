"""Bounded CMA-ES ask/tell optimization with feasibility-first ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .parameter_schema import ParameterDefinition, ParameterSchema


MetricEvaluator = Callable[[dict[str, float], str, Sequence[int]], Mapping[str, Any]]
BatchMetricEvaluator = Callable[
    [Sequence[dict[str, float]], str, Sequence[int]],
    Sequence[Mapping[str, Any]],
]


def _finite(value: Any, default: float = float("inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def constraint_priority_key(
    metrics: Mapping[str, Any],
    *,
    performance_tolerance: float = 0.0,
) -> tuple[float, ...]:
    """Return a strict DDL-first key; no constraint/energy weighted sum is used."""
    feasible = bool(metrics.get("constraint_feasible", False))
    violation = _finite(
        metrics.get(
            "deadline_violation_count",
            metrics.get(
                "max_deadline_violation_rate_across_seeds",
                metrics.get(
                    "deadline_violation_rate",
                    metrics.get("constraint_violation"),
                ),
            ),
        )
    )
    tardiness = _finite(
        metrics.get(
            "total_lateness",
            metrics.get("constraint_secondary_violation"),
        )
    )
    energy = _finite(
        metrics.get(
            "fuzzy_total_energy_score",
            metrics.get("objective", metrics.get("energy")),
        )
    )
    robustness = _finite(
        metrics.get(
            "objective_std_across_seeds",
            metrics.get("objective_cv_across_seeds", 0.0),
        ),
        0.0,
    )
    tolerance = max(0.0, float(performance_tolerance))
    energy_bucket = (
        float(round(energy / tolerance))
        if tolerance > 0.0 and math.isfinite(energy)
        else energy
    )
    if feasible:
        return (0.0, 0.0, 0.0, energy_bucket, robustness, energy)
    return (1.0, violation, tardiness, energy_bucket, robustness, energy)


def rank_fitness(
    results: Sequence[Mapping[str, Any]],
    *,
    performance_tolerance: float = 0.0,
) -> list[float]:
    """Map lexicographic constraint keys to scalar ranks for CMA-ES tell()."""
    keys = [
        constraint_priority_key(
            result,
            performance_tolerance=performance_tolerance,
        )
        for result in results
    ]
    order = sorted(range(len(keys)), key=lambda index: (keys[index], index))
    ranks = [0.0] * len(keys)
    previous_key = None
    previous_rank = 0.0
    for position, index in enumerate(order):
        if previous_key is None or keys[index] != previous_key:
            previous_rank = float(position)
            previous_key = keys[index]
        ranks[index] = previous_rank
    return ranks


@dataclass(frozen=True)
class OptimizerConfig:
    """Central configuration for bounded, staged CMA-ES optimization."""

    enabled: bool = False
    optimizer_seed: int = 0
    max_parameters: int = 12
    population_size: int = 8
    max_generations: int = 6
    initial_sigma: float = 0.25
    stage_seed_counts: Mapping[str, int] = field(
        default_factory=lambda: {"quick": 1, "refine": 3, "confirm": 0}
    )
    elite_fraction: float = 0.25
    boundary_epsilon: float = 1e-6
    sensitivity_epsilon: float = 0.02
    correlation_threshold: float = 0.85
    early_stop_patience: int = 3
    cache_enabled: bool = True
    cache_path: str = "parameter_evaluation_cache.json"
    cache_precision: int = 12
    max_parallel_evaluations: int = 1
    max_branches: int = 6
    max_ast_depth: int = 18
    max_interactions: int = 8
    performance_tolerance: float = 0.0
    diagnostic_perturbations: bool = True
    scenario_ids: Sequence[str] = ()
    auto_admission_enabled: bool = False
    admission_required: bool = True
    admission_config_path: str = ""
    admission_manifest_path: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "OptimizerConfig":
        if value is None:
            return cls()
        defaults = cls()
        known = {name for name in cls.__dataclass_fields__}
        kwargs = {name: value[name] for name in known if name in value}
        config = cls(**kwargs)
        if config.max_parameters < 1:
            raise ValueError("max_parameters must be positive")
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if config.max_generations < 1:
            raise ValueError("max_generations must be positive")
        if not 0.0 < config.initial_sigma <= 1.0:
            raise ValueError("initial_sigma must be in (0, 1]")
        if not 0.0 < config.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 < config.boundary_epsilon < 0.5:
            raise ValueError("boundary_epsilon must be in (0, 0.5)")
        if not 0.0 < config.sensitivity_epsilon < 0.5:
            raise ValueError("sensitivity_epsilon must be in (0, 0.5)")
        if not 0.0 <= config.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in [0, 1]")
        if config.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        if int(config.max_parallel_evaluations) < 1:
            raise ValueError("max_parallel_evaluations must be positive")
        scenario_ids = tuple(
            str(item).strip().upper() for item in config.scenario_ids
        )
        if any(not item for item in scenario_ids):
            raise ValueError("scenario_ids must contain non-empty identifiers")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must not contain duplicates")
        object.__setattr__(config, "scenario_ids", scenario_ids)
        counts = dict(config.stage_seed_counts)
        if any(int(counts.get(stage, 0)) < 0 for stage in ("quick", "refine", "confirm")):
            raise ValueError("stage_seed_counts must be non-negative")
        if int(counts.get("quick", 0)) < 1:
            raise ValueError("quick stage must use at least one training seed")
        if int(counts.get("refine", 0)) < 1:
            raise ValueError("refine stage must use at least one training seed")
        del defaults
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            name: (
                dict(value)
                if isinstance(value, Mapping)
                else list(value)
                if name == "scenario_ids"
                else value
            )
            for name, value in self.__dict__.items()
        }


@dataclass
class OptimizationResult:
    """Complete replayable output of one parameter search."""

    best_parameters: dict[str, float]
    best_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    elite_samples: list[dict[str, Any]]
    local_perturbations: list[dict[str, Any]]
    generations: int
    evaluations: int
    stop_reason: str
    stage_seeds: dict[str, list[int]]
    final_stage: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_parameters": dict(self.best_parameters),
            "best_metrics": dict(self.best_metrics),
            "history": list(self.history),
            "elite_samples": list(self.elite_samples),
            "local_perturbations": list(self.local_perturbations),
            "generations": self.generations,
            "evaluations": self.evaluations,
            "stop_reason": self.stop_reason,
            "stage_seeds": {key: list(value) for key, value in self.stage_seeds.items()},
            "final_stage": self.final_stage,
        }


def _unit_from_actual(definition: ParameterDefinition, value: float) -> float:
    lower, upper = definition.lower_bound, definition.upper_bound
    if definition.transform == "log":
        return float((math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower)))
    if definition.transform == "logit":
        fraction = min(max((value - lower) / (upper - lower), 1e-12), 1.0 - 1e-12)
        logit = math.log(fraction / (1.0 - fraction))
        return float((logit + 12.0) / 24.0)
    return float((value - lower) / (upper - lower))


def _actual_from_unit(definition: ParameterDefinition, unit_value: float) -> float:
    lower, upper = definition.lower_bound, definition.upper_bound
    value = min(max(float(unit_value), 0.0), 1.0)
    if definition.transform == "log":
        return float(math.exp(math.log(lower) + value * (math.log(upper) - math.log(lower))))
    if definition.transform == "logit":
        logit = -12.0 + 24.0 * value
        fraction = 1.0 / (1.0 + math.exp(-logit))
        return float(lower + fraction * (upper - lower))
    return float(lower + value * (upper - lower))


class CMAESOptimizer:
    """Use pycma ask/tell while keeping all scheduling semantics in the evaluator."""

    def __init__(self, config: OptimizerConfig):
        self.config = config

    @staticmethod
    def _import_cma():
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "CMA-ES parameter optimization is enabled but the 'cma' package is "
                "not installed. Install algorithms/llm_safe_hrl/LLM/requirements.txt."
            ) from exc
        return cma

    @staticmethod
    def _select_seeds(seeds: Sequence[int], count: int, stage: str) -> list[int]:
        available = [int(seed) for seed in seeds]
        if count <= 0:
            return []
        if not available:
            raise ValueError(f"{stage} stage has no available seeds")
        return available[: min(int(count), len(available))]

    @staticmethod
    def _safe_evaluate(
        evaluator: MetricEvaluator,
        parameters: dict[str, float],
        stage: str,
        seeds: Sequence[int],
    ) -> dict[str, Any]:
        """Record abnormal vectors as deterministic worst ranks, never cache them here."""
        try:
            return dict(evaluator(parameters, stage, seeds))
        except Exception as exc:
            worst = 1e300
            return {
                "constraint_feasible": False,
                "deadline_violation_rate": worst,
                "max_deadline_violation_rate_across_seeds": worst,
                "constraint_violation": worst,
                "total_lateness": worst,
                "constraint_secondary_violation": worst,
                "objective": worst,
                "fuzzy_total_energy_score": worst,
                "objective_std_across_seeds": worst,
                "objective_cv_across_seeds": worst,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
                "failed_stage": stage,
                "failed_seeds": [int(seed) for seed in seeds],
                "per_seed_metrics": [],
            }

    @classmethod
    def _safe_evaluate_many(
        cls,
        evaluator: MetricEvaluator,
        batch_evaluator: BatchMetricEvaluator | None,
        parameter_maps: Sequence[dict[str, float]],
        stage: str,
        seeds: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Evaluate a batch while preserving the legacy per-vector fallback."""
        if not parameter_maps:
            return []
        if batch_evaluator is None:
            return [
                cls._safe_evaluate(evaluator, parameters, stage, seeds)
                for parameters in parameter_maps
            ]
        try:
            results = list(batch_evaluator(parameter_maps, stage, seeds))
            if len(results) != len(parameter_maps):
                raise ValueError(
                    "batch evaluator must return one result per parameter vector"
                )
            return [dict(result) for result in results]
        except Exception:
            # Isolate a batch transport failure using the established per-vector
            # deterministic error handling instead of invalidating the whole batch.
            return [
                cls._safe_evaluate(evaluator, parameters, stage, seeds)
                for parameters in parameter_maps
            ]

    def optimize(
        self,
        schema: ParameterSchema,
        evaluator: MetricEvaluator,
        *,
        batch_evaluator: BatchMetricEvaluator | None = None,
        train_seeds: Sequence[int],
        validation_seeds: Sequence[int] = (),
        final_test_seeds: Sequence[int] = (),
        warm_start: Mapping[str, float] | None = None,
    ) -> OptimizationResult:
        """Run staged search and deterministic local perturbation diagnostics."""
        if not self.config.enabled:
            raise RuntimeError("CMA-ES optimizer was called while disabled")
        if len(schema.parameters) > self.config.max_parameters:
            raise ValueError("parameter schema exceeds configured max_parameters")
        train = [int(seed) for seed in train_seeds]
        validation = [int(seed) for seed in validation_seeds]
        tests = {int(seed) for seed in final_test_seeds}
        if tests.intersection(train) or tests.intersection(validation):
            raise ValueError("final test seeds must not be used for parameter optimization")
        if set(train).intersection(validation):
            raise ValueError("training and validation seeds must be disjoint")
        counts = dict(self.config.stage_seed_counts)
        stage_seeds = {
            "quick": self._select_seeds(train, int(counts.get("quick", 1)), "quick"),
            "refine": self._select_seeds(train, int(counts.get("refine", len(train))), "refine"),
            # confirm=0 means all available held-out validation seeds.
            "confirm": (
                self._select_seeds(
                    validation,
                    int(counts.get("confirm", 0)) or len(validation),
                    "confirm",
                )
                if validation
                else []
            ),
        }
        cma = self._import_cma()
        initial_values = dict(
            warm_start
            if warm_start is not None
            else zip(schema.names, schema.initial_values)
        )
        x0 = [
            _unit_from_actual(definition, float(initial_values[definition.name]))
            for definition in schema.parameters
        ]
        epsilon = self.config.boundary_epsilon
        x0 = np.clip(np.asarray(x0, dtype=float), epsilon, 1.0 - epsilon).tolist()
        strategy = cma.CMAEvolutionStrategy(
            x0,
            self.config.initial_sigma,
            {
                "bounds": [epsilon, 1.0 - epsilon],
                "popsize": self.config.population_size,
                "seed": self.config.optimizer_seed,
                "verbose": -9,
                "verb_disp": 0,
                "verb_log": 0,
            },
        )
        history: list[dict[str, Any]] = []
        best_key: tuple[float, ...] | None = None
        current_stage: str | None = None
        no_improvement = 0
        stop_reason = "max_generations"
        generations_completed = 0
        quick_generations = (
            min(max(1, self.config.max_generations // 3), self.config.max_generations - 1)
            if self.config.max_generations > 1
            else 0
        )

        for generation in range(self.config.max_generations):
            stage = "quick" if generation < quick_generations else "refine"
            if stage != current_stage:
                current_stage = stage
                best_key = None
                no_improvement = 0
            seeds = stage_seeds[stage]
            asked = strategy.ask()
            repaired = [
                np.clip(np.asarray(vector, dtype=float), epsilon, 1.0 - epsilon).tolist()
                for vector in asked
            ]
            parameter_maps = [
                {
                    definition.name: _actual_from_unit(definition, value)
                    for definition, value in zip(schema.parameters, vector)
                }
                for vector in repaired
            ]
            metrics = self._safe_evaluate_many(
                evaluator,
                batch_evaluator,
                parameter_maps,
                stage,
                seeds,
            )
            fitness = rank_fitness(
                metrics,
                performance_tolerance=self.config.performance_tolerance,
            )
            strategy.tell(repaired, fitness)
            keys = [
                constraint_priority_key(
                    item,
                    performance_tolerance=self.config.performance_tolerance,
                )
                for item in metrics
            ]
            generation_best = min(keys)
            for index, (parameters, result, key, rank) in enumerate(
                zip(parameter_maps, metrics, keys, fitness)
            ):
                history.append(
                    {
                        "generation": generation,
                        "candidate_index": index,
                        "stage": stage,
                        "seeds": list(seeds),
                        "parameters": parameters,
                        "metrics": result,
                        "comparison_key": list(key),
                        "rank": float(rank),
                    }
                )
            generations_completed = generation + 1
            if best_key is None or generation_best < best_key:
                best_key = generation_best
                no_improvement = 0
            else:
                no_improvement += 1
            if (
                stage == "refine"
                and no_improvement >= self.config.early_stop_patience
            ):
                stop_reason = "early_stop_patience"
                break
            if stage == "refine" and strategy.stop():
                stop_reason = "cma_stop:" + ",".join(sorted(strategy.stop()))
                break

        final_search_stage = (
            "refine"
            if any(item["stage"] == "refine" for item in history)
            else "quick"
        )
        final_stage_history = [
            item for item in history if item["stage"] == final_search_stage
        ]
        ordered = sorted(
            final_stage_history,
            key=lambda item: tuple(item["comparison_key"]),
        )
        elite_count = max(1, int(math.ceil(len(ordered) * self.config.elite_fraction)))
        elites = ordered[:elite_count]

        confirm_evaluations = 0
        if stage_seeds["confirm"]:
            confirm_candidates = []
            seen = set()
            confirm_limit = max(1, int(math.ceil(self.config.population_size * self.config.elite_fraction)))
            for elite in ordered:
                vector_key = tuple(round(elite["parameters"][name], 12) for name in schema.names)
                if vector_key in seen:
                    continue
                seen.add(vector_key)
                confirm_candidates.append(elite)
                if len(confirm_candidates) >= confirm_limit:
                    break
            confirm_results = self._safe_evaluate_many(
                evaluator,
                batch_evaluator,
                [item["parameters"] for item in confirm_candidates],
                "confirm",
                stage_seeds["confirm"],
            )
            confirmed = []
            for elite, result in zip(confirm_candidates, confirm_results):
                confirmed.append(
                    {
                        **elite,
                        "stage": "confirm",
                        "seeds": list(stage_seeds["confirm"]),
                        "metrics": result,
                        "comparison_key": list(
                            constraint_priority_key(
                                result,
                                performance_tolerance=self.config.performance_tolerance,
                            )
                        ),
                    }
                )
                confirm_evaluations += 1
            if confirmed:
                ordered = sorted(confirmed, key=lambda item: tuple(item["comparison_key"]))
                elites = ordered
                final_search_stage = "confirm"

        best = ordered[0]
        diagnostic_seeds = stage_seeds["confirm"] or stage_seeds["refine"]
        local_perturbations = []
        if self.config.diagnostic_perturbations:
            perturbation_requests = []
            for definition in schema.parameters:
                center = float(best["parameters"][definition.name])
                delta = self.config.sensitivity_epsilon * (
                    definition.upper_bound - definition.lower_bound
                )
                for direction in (-1, 1):
                    perturbed = dict(best["parameters"])
                    perturbed[definition.name] = min(
                        definition.upper_bound,
                        max(definition.lower_bound, center + direction * delta),
                    )
                    perturbation_requests.append(
                        (definition, direction, center, perturbed)
                    )
            perturbation_results = self._safe_evaluate_many(
                evaluator,
                batch_evaluator,
                [item[3] for item in perturbation_requests],
                "diagnostic",
                diagnostic_seeds,
            )
            for request, result in zip(
                perturbation_requests,
                perturbation_results,
            ):
                definition, direction, center, perturbed = request
                local_perturbations.append(
                    {
                        "parameter": definition.name,
                        "direction": direction,
                        "delta": perturbed[definition.name] - center,
                        "parameters": perturbed,
                        "metrics": result,
                        "comparison_key": list(
                            constraint_priority_key(
                                result,
                                performance_tolerance=self.config.performance_tolerance,
                            )
                        ),
                        "seeds": list(diagnostic_seeds),
                    }
                )

        return OptimizationResult(
            best_parameters=dict(best["parameters"]),
            best_metrics=dict(best["metrics"]),
            history=history,
            elite_samples=elites,
            local_perturbations=local_perturbations,
            generations=generations_completed,
            evaluations=(
                len(history) + confirm_evaluations + len(local_perturbations)
            ),
            stop_reason=stop_reason,
            stage_seeds=stage_seeds,
            final_stage=final_search_stage,
        )
