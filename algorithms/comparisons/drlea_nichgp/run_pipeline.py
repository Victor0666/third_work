"""Run RA training, Niching GP, SA training, and final evaluation."""

from __future__ import annotations

import argparse
import time

from .checkpointing import (
    experiment_manifest,
    portable_path,
    prepare_output,
    read_json,
    write_json,
)
from .config import build_config
from .decision_situations import collect_decision_situations
from .evaluate import evaluate_frozen
from .niching_gp import evolve_niching_gp
from .routing_agent import train_routing
from .sequencing_agent import train_sequencing


def run_pipeline(config) -> dict:
    started = time.perf_counter()
    started_cpu = time.process_time()
    output = prepare_output(config)
    routing, ra_path, _ra_history = train_routing(config)
    situations = collect_decision_situations(
        config,
        routing,
        seeds=config.train_seeds,
    )
    rules, rules_path, _gp_history = evolve_niching_gp(
        config, routing, situations
    )
    sequencing, sa_path, _sa_history = train_sequencing(
        config, routing, rules
    )
    evaluation = evaluate_frozen(
        config,
        routing,
        sequencing,
        rules,
        config.test_seeds,
    )
    gp_manifest = read_json(output / "gp_manifest.json")
    manifest = experiment_manifest(
        config,
        stage="complete_pipeline",
        elapsed_seconds=time.perf_counter() - started,
        failures=int(gp_manifest["failure_count"]),
        invalid_individuals=int(
            gp_manifest["invalid_individual_count"]
        ),
        extra={
            "ra_checkpoint": portable_path(ra_path),
            "sa_checkpoint": portable_path(sa_path),
            "rules_file": portable_path(rules_path),
            "instance_fingerprints": [
                row["instance_fingerprint"]
                for row in evaluation["seed_metrics"]
            ],
            "comparison_key": evaluation["comparison_key"],
            "compute_device": str(sequencing.device),
            "process_cpu_seconds": float(
                time.process_time() - started_cpu
            ),
        },
    )
    write_json(output / "manifest.json", manifest)
    return {
        "output_dir": portable_path(output),
        "ra_checkpoint": portable_path(ra_path),
        "rules_file": portable_path(rules_path),
        "sa_checkpoint": portable_path(sa_path),
        "evaluation": evaluation,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    result = run_pipeline(
        build_config(
            args.scenario,
            args.ddl,
            args.algorithm_seed,
            smoke=args.smoke,
        )
    )
    print(result["output_dir"])
    print(tuple(result["evaluation"]["comparison_key"]))


if __name__ == "__main__":
    main()
