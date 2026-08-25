"""Stable short-path artifacts, hashes, JSON and CSV output."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from project_paths import PROJECT_ROOT

from . import METHOD_ID
from .config import ComparisonConfig


def prepare_output(config: ComparisonConfig) -> Path:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(
        output / "source.json",
        {
            "method_id": METHOD_ID,
        "artifact_protocol_status": (
            "legacy" if config.protocol == "legacy" else "formal"
        ),
            "source_hash": source_hash(),
        },
    )
    return output


def portable_path(path: str | Path) -> str:
    """Return a project-relative POSIX path for persisted metadata."""

    value = Path(path)
    resolved = value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return value.name


def write_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(
    path: str | Path,
    rows: Iterable[dict],
    *,
    fields: list[str] | None = None,
) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if not isinstance(value, (dict, list, tuple))
            }
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def experiment_manifest(
    config: ComparisonConfig,
    *,
    stage: str,
    elapsed_seconds: float,
    failures: int = 0,
    invalid_individuals: int = 0,
    extra: dict | None = None,
) -> dict:
    return {
        "method_id": METHOD_ID,
        "artifact_protocol_status": (
            "legacy" if config.protocol == "legacy" else "formal"
        ),
        "schema_version": config.schema_version,
        "scenario": config.scenario,
        "ddl": config.ddl,
        "algorithm_seed": config.algorithm_seed,
        "optimizer_seed": int(config.algorithm_seed),
        "deadline_cache_paths": dict(config.deadline_cache_paths),
        "stage": str(stage),
        "source_hash": source_hash(),
        "train_seeds": list(config.train_seeds),
        "validation_seeds": list(config.validation_seeds),
        "test_seeds": list(config.test_seeds),
        "protocol": config.protocol,
        "source_scenario": config.source_scenario,
        "resource_scale": config.resource_scale,
        "training_scenarios": list(config.training_scenarios),
        "test_scenarios": list(config.test_scenarios),
        "validation_interval": int(config.validation_interval),
        "elapsed_seconds": float(elapsed_seconds),
        "failure_count": int(failures),
        "invalid_individual_count": int(invalid_individuals),
        **dict(extra or {}),
    }
