"""CLI for stage 2: collect situations and evolve four Niching-GP rules."""

from __future__ import annotations

import argparse

from .config import build_config
from .checkpointing import portable_path
from .decision_situations import collect_decision_situations
from .niching_gp import evolve_niching_gp
from .routing_agent import RoutingAgent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument("--population", type=int)
    result.add_argument("--generations", type=int)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = build_config(
        args.scenario,
        args.ddl,
        args.seed,
        smoke=args.smoke,
    )
    if args.population or args.generations:
        from dataclasses import replace

        config = replace(
            config,
            gp=replace(
                config.gp,
                population_size=args.population
                or config.gp.population_size,
                generations=args.generations
                or config.gp.generations,
            ),
        )
    routing = RoutingAgent.load(args.ra_checkpoint)
    situations = collect_decision_situations(
        config,
        routing,
        seeds=config.train_seeds,
    )
    _rules, path, _history = evolve_niching_gp(
        config, routing, situations
    )
    print(portable_path(path))


if __name__ == "__main__":
    main()
