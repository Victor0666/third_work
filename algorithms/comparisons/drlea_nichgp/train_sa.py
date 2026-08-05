"""CLI for stage 3: train the four-action sequencing agent."""

from __future__ import annotations

import argparse

from .config import build_config
from .checkpointing import portable_path
from .niching_gp import load_rules
from .routing_agent import RoutingAgent
from .sequencing_agent import train_sequencing


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--ra-checkpoint", required=True)
    result.add_argument("--rules-file", required=True)
    result.add_argument("--episodes", type=int)
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = build_config(
        args.scenario,
        args.ddl,
        args.seed,
        sa_episodes=args.episodes or 200,
        smoke=args.smoke,
    )
    routing = RoutingAgent.load(args.ra_checkpoint)
    rules = load_rules(args.rules_file)
    _agent, path, _history = train_sequencing(
        config, routing, rules
    )
    print(portable_path(path))


if __name__ == "__main__":
    main()
