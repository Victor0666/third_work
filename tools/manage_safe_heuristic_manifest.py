"""把已完成的 CEWS 评价报告注册为版本化安全启发式记录。

该工具不会导入或执行候选 Python。候选必须已经由 CEWS evaluator 显式评价，
且源码和评价报告分别位于 manifest 声明的可信目录中。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.heuristic_admission import (  # noqa: E402
    append_admission_record,
    build_admission_record,
)


def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required; install "
            "algorithms/llm_safe_hrl/LLM/requirements.txt"
        ) from exc
    with Path(path).resolve().open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a YAML mapping")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heuristic-id", required=True)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--seevo-iteration", required=True, type=int)
    parser.add_argument("--seevo-individual", required=True, type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    record = build_admission_record(
        heuristic_id=args.heuristic_id,
        display_name=args.display_name,
        source_path=args.candidate,
        evaluation_report_path=args.evaluation_report,
        evaluation_config=_load_yaml(args.config),
        manifest_path=manifest_path,
        seevo_iteration=args.seevo_iteration,
        seevo_individual=args.seevo_individual,
    )
    append_admission_record(manifest_path, record)
    print(
        f"heuristic_id={record['heuristic_id']} "
        f"version={record['version']} "
        f"status={record['admission_status']}"
    )
    if record["rejection_reasons"]:
        print(
            "rejection_reasons="
            + ";".join(record["rejection_reasons"])
        )
    return 0 if record["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
