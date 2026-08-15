"""Stage-2 Niching GP with clearing and feasibility-first selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

from .config import (
    config_for_scenario,
    protocol_artifact_identity,
    validate_protocol_artifact_identity,
)
from .checkpointing import (
    experiment_manifest,
    portable_path,
    prepare_output,
    read_json,
    source_hash,
    write_csv,
    write_json,
)
from .decision_situations import DecisionSituation
from .env_adapter import CEWSEnvAdapter
from .features import routing_state
from .gp_features import (
    GP_TERMINALS,
    GP_TERMINAL_VERSION,
    choose_task_with_rule,
)
from .gp_primitives import (
    GPProgram,
    crossover,
    mutate,
    random_program,
)
from .metrics import aggregate_seed_metrics, comparison_key, episode_metrics


INFINITE_KEY = (float("inf"),) * 4


@dataclass
class GPIndividual:
    program: GPProgram
    fitness: tuple[float, float, float, float] = INFINITE_KEY
    behavior: tuple[int, ...] = ()
    valid: bool = False
    cleared: bool = False

    def clone(self):
        return GPIndividual(
            self.program,
            tuple(self.fitness),
            tuple(self.behavior),
            bool(self.valid),
            bool(self.cleared),
        )


def behavior_characterization(
    program: GPProgram,
    situations: list[DecisionSituation],
) -> tuple[int, ...]:
    behavior = []
    for situation in situations:
        scores = [
            program(terminals)
            for terminals in situation.terminals
        ]
        if not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
            raise ValueError("GP program produced NaN or Inf")
        selected = min(
            range(len(scores)),
            key=lambda index: (scores[index], index),
        )
        behavior.append(int(selected))
    return tuple(behavior)


def behavioral_distance(first, second) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first, dtype=np.float64)
            - np.asarray(second, dtype=np.float64)
        )
    )


def clear_population(
    population: list[GPIndividual],
    *,
    radius: float,
    capacity: int,
) -> list[GPIndividual]:
    ordered = sorted(population, key=lambda item: item.fitness)
    for item in ordered:
        item.cleared = False
    for index, winner in enumerate(ordered):
        if not winner.valid or winner.cleared:
            continue
        niche_size = 1
        for candidate in ordered[index + 1 :]:
            if not candidate.valid or candidate.cleared:
                continue
            if (
                behavioral_distance(
                    winner.behavior, candidate.behavior
                )
                <= float(radius)
            ):
                if niche_size < int(capacity):
                    niche_size += 1
                else:
                    candidate.cleared = True
    return ordered


def run_rule_episode(config, routing_agent, program, seed: int) -> dict:
    adapter = CEWSEnvAdapter(config, int(seed))
    adapter.reset()
    actionable = adapter.advance_until_actionable()
    while actionable is not None:
        ready = adapter.ready_tasks()
        task_id, _scores = choose_task_with_rule(
            adapter, program, ready
        )
        state, mask = routing_state(
            adapter, task_id, config.normalization
        )
        vm_id, _selection = routing_agent.choose_action(
            state, mask, training=False
        )
        adapter.execute(task_id, vm_id)
        actionable = adapter.advance_until_actionable()
    return episode_metrics(adapter)


def evaluate_program(config, routing_agent, program, seeds) -> dict:
    rows = []
    for scenario in config.training_scenarios:
        scenario_config = config_for_scenario(config, scenario)
        for seed in seeds:
            row = run_rule_episode(
                scenario_config, routing_agent, program, int(seed)
            )
            row["scenario_id"] = scenario
            rows.append(row)
    return aggregate_seed_metrics(rows)


def _tournament(
    population: list[GPIndividual],
    rng: np.random.Generator,
    size: int,
) -> GPIndividual:
    candidates = [
        population[int(rng.integers(len(population)))]
        for _ in range(int(size))
    ]
    return min(candidates, key=lambda item: item.fitness)


def _update_archive(
    archive_by_behavior: dict[tuple[int, ...], GPIndividual],
    candidates: list[GPIndividual],
) -> None:
    for candidate in candidates:
        if not candidate.valid or candidate.cleared:
            continue
        current = archive_by_behavior.get(candidate.behavior)
        if current is None or candidate.fitness < current.fitness:
            archive_by_behavior[candidate.behavior] = candidate.clone()


def evolve_niching_gp(
    config,
    routing_agent,
    situations: list[DecisionSituation],
    *,
    rules_path: str | Path | None = None,
) -> tuple[list[GPProgram], Path, list[dict]]:
    if not situations:
        raise ValueError("Niching GP needs decision situations")
    started = time.perf_counter()
    started_cpu = time.process_time()
    output = prepare_output(config)
    path = Path(rules_path or output / "rules.json")
    rng = np.random.default_rng(config.algorithm_seed + 2)
    population = [
        GPIndividual(
            random_program(
                rng,
                GP_TERMINALS,
                max_depth=config.gp.max_depth,
            )
        )
        for _ in range(config.gp.population_size)
    ]
    archive_by_behavior = {}
    fitness_cache = {}
    history = []
    invalid_count = 0
    failure_count = 0

    def evaluate(individual: GPIndividual) -> None:
        nonlocal invalid_count, failure_count
        key = individual.program.tokens
        try:
            behavior = behavior_characterization(
                individual.program, situations
            )
            if key not in fitness_cache:
                metrics = evaluate_program(
                    config,
                    routing_agent,
                    individual.program,
                    config.train_seeds,
                )
                fitness_cache[key] = (
                    comparison_key(metrics),
                    metrics,
                )
            fitness, _metrics = fitness_cache[key]
            if not all(math.isfinite(value) for value in fitness):
                raise ValueError("non-finite GP fitness")
            individual.behavior = behavior
            individual.fitness = tuple(fitness)
            individual.valid = True
        except (ArithmeticError, RuntimeError, ValueError):
            individual.valid = False
            individual.fitness = INFINITE_KEY
            individual.behavior = ()
            invalid_count += 1
            failure_count += 1

    for generation in range(config.gp.generations):
        for individual in population:
            evaluate(individual)
        ordered = clear_population(
            population,
            radius=config.gp.clearing_radius,
            capacity=config.gp.clearing_capacity,
        )
        available = [
            item for item in ordered if item.valid and not item.cleared
        ]
        if not available:
            raise RuntimeError("all GP individuals are invalid or cleared")
        _update_archive(archive_by_behavior, available)
        best = min(available, key=lambda item: item.fitness)
        history.append(
            {
                "generation": generation + 1,
                "valid_count": sum(item.valid for item in population),
                "uncleared_count": len(available),
                "archive_count": len(archive_by_behavior),
                "deadline_violation_rate": best.fitness[0],
                "max_fuzzy_lateness": best.fitness[1],
                "mean_fuzzy_lateness": best.fitness[2],
                "fuzzy_energy_score": best.fitness[3],
                "best_rule": best.program.expression(),
            }
        )
        if generation + 1 >= config.gp.generations:
            break
        elite_count = min(config.gp.elite_size, len(available))
        next_population = [
            item.clone() for item in available[:elite_count]
        ]
        while len(next_population) < config.gp.population_size:
            first = _tournament(
                available, rng, config.gp.tournament_size
            )
            second = _tournament(
                available, rng, config.gp.tournament_size
            )
            child_a, child_b = first.program, second.program
            if rng.random() < config.gp.crossover_rate:
                child_a, child_b = crossover(
                    child_a,
                    child_b,
                    rng,
                    config.gp.max_depth,
                )
            for child in (child_a, child_b):
                if rng.random() < config.gp.mutation_rate:
                    child = mutate(
                        child, rng, config.gp.max_depth
                    )
                next_population.append(GPIndividual(child))
                if len(next_population) >= config.gp.population_size:
                    break
        population = next_population

    attempts = 0
    while len(archive_by_behavior) < config.gp.archive_size:
        attempts += 1
        if attempts > 256:
            raise RuntimeError(
                "Niching GP could not produce four behaviorally "
                "distinct valid rules"
            )
        candidate = GPIndividual(
            random_program(
                rng,
                GP_TERMINALS,
                max_depth=config.gp.max_depth,
            )
        )
        evaluate(candidate)
        if candidate.valid:
            _update_archive(archive_by_behavior, [candidate])

    train_archive = sorted(
        archive_by_behavior.values(),
        key=lambda item: item.fitness,
    )
    validation_cache = {}
    for individual in train_archive:
        metrics = evaluate_program(
            config,
            routing_agent,
            individual.program,
            config.validation_seeds,
        )
        validation_cache[individual.program.tokens] = metrics
    archive = sorted(
        train_archive,
        key=lambda item: comparison_key(
            validation_cache[item.program.tokens]
        ),
    )[: config.gp.archive_size]
    if len(archive) != 4:
        raise RuntimeError("exactly four GP rules are required")
    payload = {
        "schema_version": 2,
        "method_id": config.method_id,
        "scenario": config.scenario,
        "ddl": config.ddl,
        "algorithm_seed": config.algorithm_seed,
        "protocol_identity": protocol_artifact_identity(config),
        "source_hash": source_hash(),
        "routing_agent": {
            "seed": int(routing_agent.seed),
            "state_dim": int(routing_agent.state_dim),
            "action_dim": int(routing_agent.action_dim),
        },
        "terminal_version": GP_TERMINAL_VERSION,
        "terminal_names": list(GP_TERMINALS),
        "rule_count": 4,
        "selection_order": [
            "deadline_violation_rate",
            "max_fuzzy_lateness",
            "mean_fuzzy_lateness",
            "fuzzy_energy_score",
        ],
        "rules": [
            {
                "rule_id": index,
                **individual.program.to_dict(),
                "fitness": list(individual.fitness),
                "training_fitness": list(individual.fitness),
                "validation_fitness": list(
                    comparison_key(
                        validation_cache[individual.program.tokens]
                    )
                ),
                "behavior": list(individual.behavior),
            }
            for index, individual in enumerate(archive)
        ],
    }
    write_json(path, payload)
    write_json(
        output / "situations.json",
        [item.to_dict() for item in situations],
    )
    write_json(output / "gp_history.json", history)
    write_csv(output / "gp_history.csv", history)
    write_json(
        output / "gp_manifest.json",
        experiment_manifest(
            config,
            stage="gp",
            elapsed_seconds=time.perf_counter() - started,
            failures=failure_count,
            invalid_individuals=invalid_count,
            extra={
                "rules_file": portable_path(path),
                "rule_count": 4,
                "decision_situation_count": len(situations),
                "unique_candidate_evaluation_count": len(
                    fitness_cache
                ),
                "training_fitness_call_count": len(fitness_cache),
                "validation_call_count": len(validation_cache),
                "gp_evolution_seeds": list(config.train_seeds),
                "final_rule_selection_seeds": list(
                    config.validation_seeds
                ),
                "compute_device": str(routing_agent.device),
                "process_cpu_seconds": float(
                    time.process_time() - started_cpu
                ),
            },
        ),
    )
    return [item.program for item in archive], path, history


def load_rules(
    path: str | Path,
    *,
    expected_protocol_identity: dict | None = None,
) -> list[GPProgram]:
    payload = read_json(path)
    if (
        int(payload.get("schema_version", -1)) != 2
        or payload.get("method_id") != "drlea_nichgp"
        or
        int(payload.get("rule_count", -1)) != 4
        or payload.get("terminal_version") != GP_TERMINAL_VERSION
        or tuple(payload.get("terminal_names", ())) != GP_TERMINALS
    ):
        raise ValueError("incompatible GP rule archive")
    if expected_protocol_identity is not None:
        validate_protocol_artifact_identity(
            expected_protocol_identity,
            payload.get("protocol_identity"),
            artifact_name="DRL-EA GP rule archive",
        )
    rules = [
        GPProgram.from_dict(item) for item in payload["rules"]
    ]
    if len(rules) != 4:
        raise ValueError("exactly four serialized rules are required")
    return rules
