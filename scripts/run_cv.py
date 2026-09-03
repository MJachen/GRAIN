"""Authorized Phase 5A formal Center A/B full-CV entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import json

from grain.config import load_config
from grain.evaluation.audit import audit_formal_run
from grain.training.formal import run_formal_cv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    output_root = run_formal_cv(config=config, repository_root=REPOSITORY_ROOT)
    audit = audit_formal_run(output_root, REPOSITORY_ROOT)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit("Formal run completed but integrity audit failed")


if __name__ == "__main__":
    main()
