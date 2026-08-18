"""Hydra entry point for offline SeEvo rule evolution."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, ListConfig

from protocol_config import (
    configure_seevo_protocol,
    parse_deadline_cache_overrides,
)
from seevo import SeEvo
from utils.utils import init_client


_PROTOCOL_CLI_OPTIONS = {
    "--protocol": "protocol",
    "--source-scenario": "source_scenario",
    "--resource-scale": "resource_scale",
}


def _case_numbers(value) -> list[int]:
    """Normalize Hydra list, scalar, or comma-separated seed values."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [int(item) for item in value]
    if isinstance(value, str):
        return [
            int(item)
            for item in value.replace(",", " ").split()
            if item
        ]
    return [int(value)]


def _translate_protocol_cli_args(args: list[str]) -> list[str]:
    """Translate documented GNU-style flags into Hydra overrides."""
    translated: list[str] = []
    index = 0
    while index < len(args):
        argument = str(args[index])
        if argument == "--deadline-cache" or argument.startswith(
            "--deadline-cache="
        ):
            if argument == "--deadline-cache":
                if index + 1 >= len(args):
                    raise ValueError(
                        "--deadline-cache requires SCENARIO=PATH"
                    )
                value = str(args[index + 1])
                index += 2
            else:
                value = argument.split("=", 1)[1]
                index += 1
            mapping = parse_deadline_cache_overrides([value])
            scenario, path = next(iter(mapping.items()))
            translated.append(
                f"+deadline_cache_paths.{scenario}={path}"
            )
            continue
        matched = False
        for option, hydra_key in _PROTOCOL_CLI_OPTIONS.items():
            if argument == option:
                if (
                    index + 1 >= len(args)
                    or str(args[index + 1]).startswith("--")
                ):
                    raise ValueError(f"{option} requires a value")
                translated.append(f"{hydra_key}={args[index + 1]}")
                index += 2
                matched = True
                break
            prefix = option + "="
            if argument.startswith(prefix):
                value = argument[len(prefix):]
                if not value:
                    raise ValueError(f"{option} requires a value")
                translated.append(f"{hydra_key}={value}")
                index += 1
                matched = True
                break
        if not matched:
            translated.append(argument)
            index += 1
    return translated


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    """Resolve the experiment protocol before any LLM or SeEvo work."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if str(cfg.algorithm).lower() != "seevo":
        raise ValueError(
            "This project snapshot provides the SeEvo implementation only; "
            f"got {cfg.algorithm!r}."
        )
    if str(cfg.mode).strip().lower() != "train":
        raise ValueError(
            "SeEvo is restricted to offline protocol training. "
            "Generalization testing must load frozen rules/checkpoints "
            "without LLM, CMA-ES, counterfactual feedback, or updates."
        )

    llm_root = Path(__file__).resolve().parent
    protocol_context = configure_seevo_protocol(
        cfg,
        llm_root=llm_root,
    )
    logging.info(
        "Experiment protocol=%s training_scenarios=%s test_scenarios=%s",
        protocol_context.protocol,
        list(protocol_context.training_scenarios),
        list(protocol_context.test_scenarios),
    )

    # Configuration and protocol guards run before the first API call.
    init_client(cfg)
    algorithm = SeEvo(
        cfg,
        str(llm_root),
        _case_numbers(cfg.case_num),
    )
    best_code, best_code_path = algorithm.evolve()
    logging.info("Best candidate path: %s", best_code_path)
    logging.info("Best candidate code:\n%s", best_code)


if __name__ == "__main__":
    sys.argv[1:] = _translate_protocol_cli_args(sys.argv[1:])
    main()
