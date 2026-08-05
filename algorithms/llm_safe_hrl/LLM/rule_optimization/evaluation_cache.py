"""Auditable parameter-level evaluation cache with configuration isolation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .parameter_schema import canonical_json_sha256


def aggregate_seed_evaluations(
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate cached one-seed evaluator outputs using the existing mean semantics."""
    if not results or len(results) != len(seeds):
        raise ValueError("one successful evaluation result is required for every seed")
    common_keys = set(results[0])
    for result in results[1:]:
        common_keys.intersection_update(result)
    numeric_keys = [
        key
        for key in common_keys
        if all(
            not isinstance(result[key], bool)
            and isinstance(result[key], (int, float))
            and math.isfinite(float(result[key]))
            for result in results
        )
    ]
    aggregate = {
        key: float(sum(float(result[key]) for result in results) / len(results))
        for key in numeric_keys
    }
    objectives = [float(result["objective"]) for result in results]
    objective_mean = float(sum(objectives) / len(objectives))
    objective_variance = float(
        sum((value - objective_mean) ** 2 for value in objectives) / len(objectives)
    )
    aggregate["objective_std_across_seeds"] = math.sqrt(objective_variance)
    aggregate["objective_max_across_seeds"] = max(objectives)
    aggregate["objective_cv_across_seeds"] = (
        aggregate["objective_std_across_seeds"]
        / max(abs(objective_mean), 1e-12)
    )
    aggregate["constraint_feasible"] = all(
        bool(result.get("constraint_feasible", False)) for result in results
    )
    aggregate["feasible_seed_rate"] = float(
        sum(bool(result.get("constraint_feasible", False)) for result in results)
        / len(results)
    )
    aggregate["max_deadline_violation_rate_across_seeds"] = max(
        float(
            result.get(
                "max_deadline_violation_rate_across_seeds",
                result.get("deadline_violation_rate", float("inf")),
            )
        )
        for result in results
    )
    aggregate["max_fuzzy_lateness"] = max(
        float(result.get("max_fuzzy_lateness", 0.0)) for result in results
    )
    aggregate["seeds"] = [int(seed) for seed in seeds]
    aggregate["evaluation_seed_count"] = len(results)
    aggregate["completed_seed_count"] = len(results)
    aggregate["all_evaluation_seeds_completed"] = True
    aggregate["per_seed_metrics"] = []
    for seed, result in zip(seeds, results):
        rows = result.get("per_seed_metrics", [])
        if isinstance(rows, list) and rows:
            row = dict(rows[0])
        else:
            row = {
                "seed": int(seed),
                "scenario_id": str(result.get("scenario_id", "unknown")),
                "constraint_feasible": bool(result.get("constraint_feasible", False)),
                "deadline_violation_rate": float(
                    result.get("deadline_violation_rate", float("inf"))
                ),
                "total_lateness": float(result.get("total_lateness", float("inf"))),
                "fuzzy_total_energy_score": float(result.get("objective", float("inf"))),
                "objective": float(result.get("objective", float("inf"))),
            }
        row["seed"] = int(seed)
        aggregate["per_seed_metrics"].append(row)
    passthrough = (
        "scenario_id",
        "structure_hash",
        "parameter_schema_hash",
        "best_parameter_hash",
        "optimizer_config_hash",
        "optimizer_seed",
        "parameter_diagnostics_hash",
        "training_seeds",
        "validation_seeds",
    )
    for key in passthrough:
        if key in results[0]:
            aggregate[key] = results[0][key]
    return aggregate


@dataclass(frozen=True)
class EvaluationCacheKey:
    """Fields that make one simulation result safe to reuse."""

    structure_hash: str
    parameter_vector: tuple[float, ...]
    seed: int
    scenario_id: str
    evaluation_config_hash: str
    resource_config_hash: str

    @classmethod
    def create(
        cls,
        *,
        structure_hash: str,
        parameter_vector: Sequence[float],
        seed: int,
        scenario_id: str,
        evaluation_config_hash: str,
        resource_config_hash: str,
        precision: int = 12,
    ) -> "EvaluationCacheKey":
        vector = tuple(round(float(value), int(precision)) for value in parameter_vector)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("cache parameter vector must contain finite values")
        return cls(
            structure_hash=str(structure_hash),
            parameter_vector=vector,
            seed=int(seed),
            scenario_id=str(scenario_id),
            evaluation_config_hash=str(evaluation_config_hash),
            resource_config_hash=str(resource_config_hash),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_hash": self.structure_hash,
            "parameter_vector": list(self.parameter_vector),
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "evaluation_config_hash": self.evaluation_config_hash,
            "resource_config_hash": self.resource_config_hash,
        }

    @property
    def digest(self) -> str:
        return canonical_json_sha256(self.as_dict())


class EvaluationCache:
    """Small JSON-backed cache; failed evaluations are never stored."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        path: str | Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path).resolve() if path else None
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path is not None and self.path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation cache: {self.path}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported evaluation cache schema: {self.path}")
        entries = payload.get("entries", {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"evaluation cache entries must be a mapping: {self.path}")
        self._entries = {
            str(key): dict(value)
            for key, value in entries.items()
            if isinstance(value, Mapping) and value.get("status") == "ok"
        }

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "entries": self._entries,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def get(self, key: EvaluationCacheKey) -> dict[str, Any] | None:
        if not self.enabled:
            self.misses += 1
            return None
        entry = self._entries.get(key.digest)
        if entry is None or entry.get("key") != key.as_dict():
            self.misses += 1
            return None
        result = entry.get("result")
        if not isinstance(result, Mapping):
            self.misses += 1
            return None
        self.hits += 1
        return dict(result)

    def put(
        self,
        key: EvaluationCacheKey,
        result: Mapping[str, Any],
        *,
        successful: bool = True,
    ) -> None:
        if not self.enabled or not successful:
            return
        if not isinstance(result, Mapping):
            raise TypeError("cached evaluation result must be a mapping")
        self._entries[key.digest] = {
            "status": "ok",
            "key": key.as_dict(),
            "result": dict(result),
        }
        self._persist()

    def put_many(
        self,
        entries: Sequence[tuple[EvaluationCacheKey, Mapping[str, Any]]],
    ) -> None:
        """Store successful results with one atomic persistence operation."""
        if not self.enabled or not entries:
            return
        for key, result in entries:
            if not isinstance(result, Mapping):
                raise TypeError("cached evaluation result must be a mapping")
            self._entries[key.digest] = {
                "status": "ok",
                "key": key.as_dict(),
                "result": dict(result),
            }
        self._persist()

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return float(self.hits / self.requests) if self.requests else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
        }
