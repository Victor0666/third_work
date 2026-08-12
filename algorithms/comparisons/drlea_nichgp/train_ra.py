"""CLI for stage 1: train the flat global-VM routing agent."""

from __future__ import annotations

import argparse

from .config import build_config
from .checkpointing import portable_path
from .routing_agent import train_routing


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario", default="SS")
    result.add_argument("--ddl", default="T")
    result.add_argument(
        "--algorithm-seed", dest="algorithm_seed", type=int, default=0
    )
    result.add_argument("--episodes", type=int)
    result.add_argument("--reward-mode", default="deadline_energy")
    result.add_argument("--smoke", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = build_config(
        args.scenario,
        args.ddl,
        args.algorithm_seed,
        ra_episodes=args.episodes or 200,
        reward_mode=args.reward_mode,
        smoke=args.smoke,
    )
    _agent, path, _history = train_routing(config)
    print(portable_path(path))


if __name__ == "__main__":
    main()
