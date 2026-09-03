"""CLI for dataset-specific formal training readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.config import load_config
from grain.training.preflight import run_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run_preflight(load_config(REPOSITORY_ROOT / args.config), REPOSITORY_ROOT)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
