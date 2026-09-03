from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from grain.evaluation import audit_formal_run, compute_metrics


ROOT = Path(__file__).parents[1]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FormalAuditTests(unittest.TestCase):
    def test_complete_synthetic_artifact_passes(self) -> None:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("immutable-source", encoding="utf-8")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            ids = [f"sample_{index:02d}" for index in range(20)]
            write_json(
                root / "run_manifest.json",
                {
                    "status": "COMPLETE",
                    "expected_samples": 20,
                    "data_sha256": source_sha,
                    "git_commit": commit,
                    "data_source": str(source),
                },
            )
            all_predictions = []
            for fold in range(10):
                fold_dir = root / f"fold_{fold:02d}"
                fold_dir.mkdir()
                test_ids = ids[fold * 2 : fold * 2 + 2]
                remaining = [value for value in ids if value not in test_ids]
                validation_ids = remaining[:2]
                train_ids = remaining[2:]
                write_json(
                    fold_dir / "split.json",
                    {
                        "train": train_ids,
                        "validation": validation_ids,
                        "test": test_ids,
                        "selected_k": 2,
                    },
                )
                write_json(fold_dir / "data_fingerprint.json", {"sha256": source_sha})
                write_json(
                    fold_dir / "resolved_config.json",
                    {
                        "training": {
                            "checkpoint_metric": "validation_auc",
                            "k_selection": {"mode": "validation"},
                        }
                    },
                )
                k_rows = [
                    {
                        "fold": fold,
                        "candidate_k": k,
                        "validation_auc": 1.0 if k == 2 else 0.5,
                        "validation_loss": 1.0,
                        "selected": k == 2,
                    }
                    for k in range(2, 11)
                ]
                write_csv(fold_dir / "k_search.csv", k_rows)
                history = [
                    {
                        "epoch": epoch,
                        "loss": 1.0,
                        "classification": 0.5,
                        "auxiliary": 0.4,
                        "contrastive": 0.3,
                        "balance": 0.2,
                        "gradient_norm": 1.0,
                        "validation_loss": 1.0,
                        "validation_auc": 1.0 if epoch == 0 else 0.5,
                    }
                    for epoch in range(100)
                ]
                write_csv(fold_dir / "train_history.csv", history)
                checkpoint = fold_dir / "best.pt"
                checkpoint.write_bytes(f"checkpoint-{fold}".encode())
                checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                write_json(
                    fold_dir / "checkpoint_metadata.json",
                    {
                        "selection_split": "validation",
                        "selection_metric": "validation_auc",
                        "test_evaluation_count": 1,
                        "git_commit": commit,
                        "data_sha256": source_sha,
                        "checkpoint_sha256": checkpoint_sha,
                        "training_anchor_ids": train_ids[:2],
                        "selected_k": 2,
                        "epoch": 0,
                    },
                )
                test_rows = []
                validation_rows = []
                for split_name, split_ids, destination in (
                    ("test", test_ids, test_rows),
                    ("validation", validation_ids, validation_rows),
                ):
                    for index, sample_id in enumerate(split_ids):
                        probability = 0.1 if index == 0 else 0.9
                        destination.append(
                            {
                                "sample_id": sample_id,
                                "fold": fold,
                                "split": split_name,
                                "label": index,
                                "probability": probability,
                                "prediction": index,
                            }
                        )
                write_csv(fold_dir / "test_predictions.csv", test_rows)
                write_csv(fold_dir / "validation_predictions.csv", validation_rows)
                metrics = compute_metrics([0, 1], [0.1, 0.9]).to_dict()
                write_json(fold_dir / "metrics.json", {"test_metrics": metrics})
                all_predictions.extend(test_rows)
            write_csv(root / "oof_predictions.csv", all_predictions)
            report = audit_formal_run(root, ROOT)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["blockers"], [])


if __name__ == "__main__":
    unittest.main()

