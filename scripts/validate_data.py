"""Validate an approved cohort manifest before split generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.config import load_config
from grain.data.schema import CohortManifest
from grain.data.validation import cohort_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = CohortManifest.from_csv(
        REPOSITORY_ROOT / config["data"]["cohort_manifest"]
    )
    print(json.dumps(cohort_summary(manifest), indent=2))


if __name__ == "__main__":
    main()
