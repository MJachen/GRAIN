"""Create metadata-only fingerprints for configured legacy data candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.config import load_config
from grain.data.legacy import inspect_legacy_table
from grain.data.paths import RepositoryPathPolicy


PAPER_EXPECTED = {
    "grain_center_a": {"total": 160, "positive": 48, "plain": 144, "ce": 22, "both": 6},
    "grain_center_b": {"total": 185, "positive": 39, "plain": 65, "ce": 165, "both": 45},
    "grain_mixed_ab": {"total": 345, "positive": 87},
    "grain_external_center_c": {"total": 83, "positive": 23, "plain": 83, "ce": 36, "both": 36},
}

CODE_EVIDENCE = {
    "grain_center_a": ["attention/trainerv1.py:164", "attention/modloss.py:127"],
    "grain_center_b": ["attention/crossval.py:303", "attention/difffusion.py:98"],
    "grain_mixed_ab": ["attention/search4k.py:100", "attention/trainerv1.py:165 (commented)"],
    "grain_external_center_c": [
        "attention/test4sth.py:272",
        "attention/evaluate.py:68 references a currently missing external file",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/data_inventory.json")
    args = parser.parse_args()
    items = []
    source_root = None
    for name in ("center_a", "center_b", "mixed_ab", "external_center_c"):
        config = load_config(REPOSITORY_ROOT / "configs" / f"{name}.json")
        data = config["data"]
        source_root = data["source_root"]
        policy = RepositoryPathPolicy.create(REPOSITORY_ROOT, source_root)
        layout = data["legacy_layout"]
        item = inspect_legacy_table(
            policy.resolve_source(data["feature_source"]),
            skip_leading_columns=int(layout["skip_leading_columns"]),
            plain_dimension=int(layout["plain_dimension"]),
            ce_dimension=int(layout["ce_dimension"]),
            label_column=data["label_column"],
            confidence=data["provenance_status"],
            paper_expected=PAPER_EXPECTED[config["experiment"]["name"]],
            code_evidence=CODE_EVIDENCE[config["experiment"]["name"]],
        )
        item["dataset"] = config["experiment"]["name"]
        items.append(item)
        for alternative_index, alternative in enumerate(
            data.get("feature_source_alternatives", [])
        ):
            alternative_layout = alternative["legacy_layout"]
            alternative_item = inspect_legacy_table(
                policy.resolve_source(alternative["path"]),
                skip_leading_columns=int(alternative_layout["skip_leading_columns"]),
                plain_dimension=int(alternative_layout["plain_dimension"]),
                ce_dimension=int(alternative_layout["ce_dimension"]),
                label_column=data["label_column"],
                confidence="conflict",
                paper_expected=PAPER_EXPECTED[config["experiment"]["name"]],
                code_evidence=["attention/zhexiantu.py:205"],
            )
            alternative_item["dataset"] = (
                f"{config['experiment']['name']}_alternative_{alternative_index + 1}"
            )
            alternative_item["candidate_role"] = alternative["role"]
            items.append(alternative_item)
    if source_root is None:
        raise RuntimeError("No configured datasets")
    policy = RepositoryPathPolicy.create(REPOSITORY_ROOT, source_root)
    output = policy.resolve_write(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_root": Path(source_root).resolve().as_posix(),
        "source_access": "read_only",
        "patient_identity_provenance": "BLOCKED",
        "patient_identity_reason": (
            "Clinical feature tables do not contain a verified stable patient ID; "
            "row numbers are not accepted as identity."
        ),
        "datasets": items,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
