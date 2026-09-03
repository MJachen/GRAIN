"""Reserved full-CV entry point; deliberately gated until Phase 5."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    raise SystemExit(
        "Configuration is valid for experiment "
        f"{config['experiment']['name']!r}, but training is intentionally disabled "
        "until Phase 5 full-CV authorization. Use run_smoke_fold.py only for Phase 4B."
    )


if __name__ == "__main__":
    main()
