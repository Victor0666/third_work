"""Resolve experiment protocols before SeEvo constructs any component."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from omegaconf import DictConfig, OmegaConf


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from algorithms.llm_safe_hrl.scenario_registry import (  # noqa: E402
    ExperimentProtocolContext,
    apply_scenario_to_problem_config,
    resolve_experiment_protocol,
    validate_component_scenarios,
)


def _mapping(value) -> dict:
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value or {})


def _bind_component_scenarios(
    context: ExperimentProtocolContext,
    component_name: str,
    component_config,
) -> None:
    """Fill an empty protocol placeholder, but reject an explicit conflict."""
    configured = [
        str(value).strip().upper()
        for value in list(getattr(component_config, "scenario_ids", []) or [])
    ]
    if configured:
        validate_component_scenarios(context, component_name, configured)
    component_config.scenario_ids = list(context.training_scenarios)


def configure_seevo_protocol(
    cfg: DictConfig,
    *,
    llm_root: str | Path,
) -> ExperimentProtocolContext:
    """Materialize one protocol domain and fail before any LLM/API call."""
    problem = _mapping(cfg.problem)
    dataset = dict(problem.get("dataset") or {})
    context = resolve_experiment_protocol(
        str(getattr(cfg, "protocol", "single")),
        source_scenario=getattr(cfg, "source_scenario", "SS"),
        resource_scale=getattr(cfg, "resource_scale", None),
        train_seeds=dataset.get("train_seeds", (1, 2, 3)),
        validation_seeds=dataset.get("validation_seeds", (4, 5)),
    )
    primary_scenario = context.training_scenarios[0]
    problem = apply_scenario_to_problem_config(
        problem,
        primary_scenario,
        require_files=True,
    )
    problem["experiment_protocol"] = context.identity()
    problem["dataset"]["train_seeds"] = list(context.llm_train_seeds)
    problem["dataset"]["validation_seeds"] = list(context.llm_validation_seeds)
    cfg.problem = OmegaConf.create(problem)

    training = list(context.training_scenarios)
    _bind_component_scenarios(
        context,
        "CMA-ES",
        cfg.parameter_optimization,
    )
    _bind_component_scenarios(
        context,
        "Counterfactual",
        cfg.counterfactual_feedback,
    )
    for name, scenarios in (
        ("SeEvo", training),
        ("CMA-ES", cfg.parameter_optimization.scenario_ids),
        ("Counterfactual", cfg.counterfactual_feedback.scenario_ids),
        ("Rule Admission", training),
    ):
        validate_component_scenarios(context, name, scenarios)

    output_root = context.artifact_output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generated").mkdir(parents=True, exist_ok=True)
    context.checkpoint_root.mkdir(parents=True, exist_ok=True)
    context.library_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.parameter_optimization.cache_path = str(
        output_root / "parameter_evaluation_cache.json"
    )
    cfg.counterfactual_feedback.cache.path = str(
        output_root / "counterfactual_feedback" / "cache.json"
    )
    cfg.critical_state_replay.archive.archive_path = str(
        output_root / "critical_state_replay" / "archive" / "critical_state_archive.json"
    )
    cfg.critical_state_replay.cache.path = str(
        output_root / "critical_state_replay" / "replay_cache.json"
    )

    llm_root = Path(llm_root).resolve()
    admission_template = (
        llm_root / "cfg" / "problem" / "cews_task_constructive_hrl_ss_admission.yaml"
    )
    admission = _mapping(OmegaConf.load(admission_template))
    admission = apply_scenario_to_problem_config(
        admission,
        primary_scenario,
        require_files=True,
    )
    admission["experiment_protocol"] = context.identity()
    admission["dataset"]["train_seeds"] = list(context.llm_train_seeds)
    admission["dataset"]["validation_seeds"] = list(context.llm_validation_seeds)
    admission_scope = admission.setdefault("admission_scope", {})
    admission_scope["resource_code"] = context.resource_scale
    admission_scope["allowed_scenarios"] = list(context.test_scenarios)
    admission.setdefault("admission", {})["required_evaluation_seeds"] = list(
        context.llm_train_seeds
    )
    admission["admission"]["minimum_evaluation_seed_count"] = len(
        context.llm_train_seeds
    )
    # Runtime scope remains resource-domain-wide for frozen generalization;
    # this separate field restricts actual offline admission simulation.
    admission["admission_evaluation_scenarios"] = training
    admission_path = output_root / "effective_admission_config.yaml"
    OmegaConf.save(OmegaConf.create(admission), admission_path, resolve=True)
    cfg.parameter_optimization.admission_config_path = str(admission_path)
    cfg.parameter_optimization.admission_manifest_path = str(
        context.library_path.resolve()
    )

    OmegaConf.update(cfg, "experiment_protocol", context.identity(), force_add=True)
    OmegaConf.update(
        cfg,
        "protocol_paths",
        {
            "output_root": str(output_root),
            "checkpoint_root": str(context.checkpoint_root.resolve()),
            "library_path": str(context.library_path.resolve()),
        },
        force_add=True,
    )
    manifest = {
        "manifest_version": "llm_safe_drl_experiment_protocol_v1",
        **context.identity(),
        "output_root": str(output_root),
        "checkpoint_root": str(context.checkpoint_root.resolve()),
        "heuristic_library": str(context.library_path.resolve()),
    }
    (output_root / "experiment_protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return context


__all__ = ["configure_seevo_protocol"]
