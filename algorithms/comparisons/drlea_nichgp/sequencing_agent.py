"""Stage-3 four-rule sequencing D3QN and training loop."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .checkpointing import (
    experiment_manifest,
    portable_path,
    prepare_output,
    write_csv,
    write_json,
)
from .d3qn import MaskedD3QN
from .env_adapter import CEWSEnvAdapter
from .features import routing_state, sequencing_state
from .gp_features import choose_task_with_rule
from .metrics import aggregate_seed_metrics, comparison_key, episode_metrics
from .rewards import sequencing_reward


class SequencingAgent(MaskedD3QN):
    expected_role = "sequencing"

    @classmethod
    def create(cls, config, seed: int):
        return cls(
            16,
            4,
            config.sequencing,
            seed,
            role="sequencing",
        )


def run_sequencing_episode(
    config,
    routing_agent,
    rules,
    sequencing_agent,
    seed: int,
    *,
    training: bool,
) -> tuple[dict, list[dict]]:
    if len(rules) != 4:
        raise ValueError("SA requires exactly four frozen GP rules")
    adapter = CEWSEnvAdapter(config, int(seed))
    adapter.reset()
    actionable = adapter.advance_until_actionable()
    trace = []
    rule_mask = np.ones(4, dtype=np.float32)
    while actionable is not None:
        ready = adapter.ready_tasks()
        state = sequencing_state(
            adapter, config.normalization, ready
        )
        rule_index, selection = sequencing_agent.choose_action(
            state, rule_mask, training=training
        )
        task_id, priorities = choose_task_with_rule(
            adapter, rules[rule_index], ready
        )
        ra_state, ra_mask = routing_state(
            adapter, task_id, config.normalization
        )
        vm_id, _ra_selection = routing_agent.choose_action(
            ra_state, ra_mask, training=False
        )
        before_slack = adapter.aggregate_risk_slack()
        adapter.execute(task_id, vm_id)
        after_slack = adapter.aggregate_risk_slack()
        reward, reward_fields = sequencing_reward(
            before_slack, after_slack, config.reward
        )
        next_task = adapter.advance_until_actionable()
        done = next_task is None
        next_state = (
            np.zeros(16, dtype=np.float32)
            if done
            else sequencing_state(
                adapter, config.normalization
            )
        )
        next_mask = (
            np.zeros(4, dtype=np.float32)
            if done
            else rule_mask
        )
        loss = None
        if training:
            sequencing_agent.remember(
                state,
                rule_index,
                reward,
                next_state,
                done,
                rule_mask,
                next_mask,
            )
            loss = sequencing_agent.learn()
        trace.append(
            {
                "task_id": int(task_id),
                "global_vm_id": int(vm_id),
                "rule_index": int(rule_index),
                "selection": selection,
                "selected_priority": float(
                    priorities[ready.index(task_id)]
                ),
                "reward": reward,
                "loss": loss,
                **reward_fields,
            }
        )
        actionable = next_task
    return episode_metrics(adapter), trace


def evaluate_sequencing(
    config,
    routing_agent,
    rules,
    sequencing_agent,
    seeds,
) -> dict:
    rows = [
        run_sequencing_episode(
            config,
            routing_agent,
            rules,
            sequencing_agent,
            int(seed),
            training=False,
        )[0]
        for seed in seeds
    ]
    return aggregate_seed_metrics(rows)


def train_sequencing(
    config,
    routing_agent,
    rules,
    *,
    checkpoint_path: str | Path | None = None,
) -> tuple[SequencingAgent, Path, list[dict]]:
    started = time.perf_counter()
    started_cpu = time.process_time()
    output = prepare_output(config)
    path = Path(checkpoint_path or output / "sa.pt")
    agent = SequencingAgent.create(config, config.seed + 1)
    best_key = None
    history = []
    interaction_count = 0
    for episode in range(config.sa_episodes):
        train_seed = config.train_seeds[
            episode % len(config.train_seeds)
        ]
        train_metrics, trace = run_sequencing_episode(
            config,
            routing_agent,
            rules,
            agent,
            train_seed,
            training=True,
        )
        interaction_count += len(trace)
        validation = evaluate_sequencing(
            config,
            routing_agent,
            rules,
            agent,
            config.validation_seeds,
        )
        key = comparison_key(validation)
        saved = best_key is None or key < best_key
        if saved:
            best_key = key
            agent.save(path, metrics=validation)
        history.append(
            {
                "episode": episode + 1,
                "train_seed": int(train_seed),
                "mean_reward": float(
                    np.mean([row["reward"] for row in trace])
                )
                if trace
                else 0.0,
                "epsilon": float(agent.epsilon),
                "saved": bool(saved),
                "train_fuzzy_energy_score": train_metrics[
                    "fuzzy_energy_score"
                ],
                **{
                    f"validation_{name}": float(validation[name])
                    for name in (
                        "deadline_violation_rate",
                        "max_fuzzy_lateness",
                        "mean_fuzzy_lateness",
                        "fuzzy_energy_score",
                    )
                },
            }
        )
    best_agent = SequencingAgent.load(path)
    write_json(output / "sa_history.json", history)
    write_csv(output / "sa_history.csv", history)
    write_json(
        output / "sa_manifest.json",
        experiment_manifest(
            config,
            stage="sa",
            elapsed_seconds=time.perf_counter() - started,
            extra={
                "checkpoint": portable_path(path),
                "best_comparison_key": list(best_key or ()),
                "environment_interaction_count": interaction_count,
                "validation_call_count": int(config.sa_episodes),
                "compute_device": str(agent.device),
                "process_cpu_seconds": float(
                    time.process_time() - started_cpu
                ),
            },
        ),
    )
    return best_agent, path, history
