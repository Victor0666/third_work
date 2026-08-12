"""CLI and API for final frozen RA + GP + SA evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from .checkpointing import (
    experiment_manifest,
    portable_path,
    prepare_output,
    write_csv,
    write_json,
)
from .config import build_config, ensure_disjoint_seeds
from .niching_gp import load_rules
from .routing_agent import RoutingAgent
from .sequencing_agent import (
    SequencingAgent,
    evaluate_sequencing,
)


def evaluate_frozen(
    config,
    routing,
    sequencing,
    rules,
    seeds,
    *,
    output_path: str | Path | None = None,
) -> dict:
    ensure_disjoint_seeds(
        config.train_seeds,
        config.validation_seeds,
        tuple(int(seed) for seed in seeds),
    )
    started = time.perf_counter()
    started_cpu = time.process_time()
    output = prepare_output(config)
    metrics = evaluate_sequencing(
        config, routing, rules, sequencing, seeds
    )
    path = Path(output_path or output / "eval.json")
    write_json(path, metrics)
    write_csv(output / "eval.csv", metrics["seed_metrics"])
    workflow_rows = [
        {
            "seed": row["seed"],
            **workflow,
        }
        for row in metrics["seed_metrics"]
        for workflow in row["workflow_metrics"]
    ]
    write_csv(output / "workflows.csv", workflow_rows)
    write_json(
        output / "eval_manifest.json",
        experiment_manifest(
            config,
            stage="evaluation",
            elapsed_seconds=time.perf_counter() - started,
            extra={
                "evaluation_seeds": [int(seed) for seed in seeds],
                "comparison_key": metrics["comparison_key"],
                "all_seed_feasible": metrics[
                    "all_seed_feasible"
                ],
                "feasible_seed_rate": metrics[
                    "feasible_seed_rate"
                ],
                "worst_seed_violation": metrics[
                    "worst_seed_violation"
                ],
                "worst_seed_lateness": metrics[
                    "worst_seed_lateness"
                ],
                "compute_device": str(sequencing.device),
                "process_cpu_seconds": float(
                    time.process_time() - started_cpu
                ),
            },
        ),
    )
    return metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument("--sa-checkpoint", required=True)
    result.add_argument("--rules-file", required=True)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = build_config(
        args.scenario,
        args.ddl,
        args.algorithm_seed,
        smoke=args.smoke,
    )
    metrics = evaluate_frozen(
        config,
        RoutingAgent.load(args.ra_checkpoint),
        SequencingAgent.load(args.sa_checkpoint),
        load_rules(args.rules_file),
        config.test_seeds,
    )
    print(portable_path(config.output_dir / "eval.json"))
    print(tuple(metrics["comparison_key"]))


if __name__ == "__main__":
    main()
