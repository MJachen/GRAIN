"""Generate immutable patient-level outer/validation split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.config import load_config
from grain.data.schema import CohortManifest
from grain.data.splits import generate_nested_manifests, write_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = REPOSITORY_ROOT
    manifest_path = repo_root / config["data"]["cohort_manifest"]
    cohort = CohortManifest.from_csv(manifest_path)
    output_dir = Path(args.output_dir or config["split"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    manifests = generate_nested_manifests(
        cohort=cohort,
        dataset_name=config["experiment"]["name"],
        n_outer_folds=int(config["split"]["n_outer_folds"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    write_manifests(manifests, output_dir)
    print(f"Created {len(manifests)} patient-level manifests in {output_dir}")


if __name__ == "__main__":
    main()
