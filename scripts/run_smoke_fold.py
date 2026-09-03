"""Run one explicitly labelled real-data development smoke fold."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from grain.config import load_config
from grain.data import load_legacy_feature_table, generate_nested_manifests_from_arrays
from grain.data.paths import RepositoryPathPolicy
from grain.training import (
    ValidationCheckpointManager,
    build_fold_runtime,
    load_checkpoint,
    select_k_on_validation,
    test,
    train_one_epoch,
    validate,
)


def git_commit() -> str:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Smoke run requires a clean committed code state")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def load_bound_cohort(config: dict):
    data = config["data"]
    policy = RepositoryPathPolicy.create(REPOSITORY_ROOT, data["source_root"])
    layout = data["legacy_layout"]
    return load_legacy_feature_table(
        policy.resolve_source(data["feature_source"]),
        sample_id_prefix=data["sample_id_prefix"],
        skip_leading_columns=int(layout["skip_leading_columns"]),
        plain_dimension=int(layout["plain_dimension"]),
        ce_dimension=int(layout["ce_dimension"]),
        label_column=data["label_column"],
        expected_sha256=data["expected_sha256"],
    )


def distribution(cohort, ids) -> dict[str, int]:
    masks = cohort.select(ids)[1]
    return {
        "plain_only": int(((masks[:, 0]) & (~masks[:, 1])).sum()),
        "ce_only": int(((~masks[:, 0]) & (masks[:, 1])).sum()),
        "both": int(masks.all(dim=1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_center_a.json")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    config = load_config(REPOSITORY_ROOT / args.config)
    if config["experiment"].get("purpose") != "DEVELOPMENT SMOKE TEST - NOT PAPER RESULT":
        raise SystemExit("This entry point accepts only an explicitly marked smoke config")
    cohort = load_bound_cohort(config)
    if cohort.feature_dimension != int(config["model"]["feature_dimension"]):
        raise ValueError("Bound feature dimension does not match model projection input")
    manifests = generate_nested_manifests_from_arrays(
        cohort.sample_ids,
        cohort.labels.tolist(),
        config["experiment"]["name"],
        n_outer_folds=int(config["split"]["n_outer_folds"]),
        validation_fraction=float(config["split"]["validation_fraction"]),
        seed=int(config["split"]["seed"]),
    )
    if not 0 <= args.fold < len(manifests):
        raise ValueError("fold is a zero-based outer-fold index")
    split = manifests[args.fold]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["experiment"]["seed"]) + args.fold
    output_root = REPOSITORY_ROOT / "outputs" / config["experiment"]["name"] / f"fold_{args.fold:02d}"
    output_root.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    inventory = json.loads((REPOSITORY_ROOT / "artifacts" / "data_inventory.json").read_text(encoding="utf-8"))
    fingerprint = next(
        item for item in inventory["datasets"] if item["sha256"] == cohort.source_sha256
    )

    selected_k, candidates = select_k_on_validation(
        config=config,
        cohort=cohort,
        split=split,
        device=device,
        seed=seed,
        epochs=int(config["training"]["k_selection"]["search_epochs"]),
    )
    split = replace(split, selected_k=selected_k)
    split.to_json(output_root / "split.json")
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "k_search.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "candidate_k", "validation_auc", "validation_loss", "selected"])
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "fold": args.fold,
                    "candidate_k": item.k,
                    "validation_auc": item.validation_auc,
                    "validation_loss": item.validation_loss,
                    "selected": item.k == selected_k,
                }
            )

    runtime = build_fold_runtime(
        config=config,
        cohort=cohort,
        split=split,
        selected_k=selected_k,
        device=device,
        seed=seed,
    )
    checkpoint = ValidationCheckpointManager(output_root / "best.pt")
    log_rows = []
    for epoch in range(int(config["training"]["epochs"])):
        training = train_one_epoch(
            model=runtime.model,
            optimizer=runtime.optimizer,
            cohort=cohort,
            training_ids=split.train,
            anchor_bank=runtime.anchor_bank,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            epoch_seed=seed + epoch,
            contrastive_temperature=float(config["training"]["contrastive_temperature"]),
            balance_margin=float(config["training"]["balance_margin"]),
            contrastive_weight=float(config["training"]["loss_weights"]["contrastive"]),
            balance_weight=float(config["training"]["loss_weights"]["balance"]),
            auxiliary_weight=float(config["training"]["loss_weights"]["auxiliary"]),
        )
        validation = validate(
            model=runtime.model,
            cohort=cohort,
            sample_ids=split.validation,
            anchor_bank=runtime.anchor_bank,
            fold=args.fold,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            threshold=float(config["evaluation"]["threshold"]),
        )
        checkpoint.consider(
            validation,
            model=runtime.model,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            epoch=epoch,
            fold=args.fold,
            seed=seed,
            selected_k=selected_k,
            resolved_config=config,
            data_fingerprint=fingerprint,
            git_commit=commit,
        )
        log_rows.append(
            {
                "epoch": epoch,
                **asdict(training),
                "validation_loss": validation.loss,
                "validation_auc": validation.metrics.auc,
            }
        )
        if runtime.scheduler is not None:
            runtime.scheduler.step()
    with (output_root / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)

    payload = load_checkpoint(
        output_root / "best.pt", runtime.model, map_location=device
    )
    best_validation = validate(
        model=runtime.model,
        cohort=cohort,
        sample_ids=split.validation,
        anchor_bank=runtime.anchor_bank,
        fold=args.fold,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        threshold=float(config["evaluation"]["threshold"]),
    )
    threshold = float(config["evaluation"]["threshold"])
    best_validation.write_csv(output_root / "validation_predictions.csv", threshold)
    # The outer test is first accessed here, after K and checkpoint selection.
    held_out_test = test(
        model=runtime.model,
        cohort=cohort,
        sample_ids=split.test,
        anchor_bank=runtime.anchor_bank,
        fold=args.fold,
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        threshold=float(config["evaluation"]["threshold"]),
    )
    held_out_test.write_csv(output_root / "test_predictions.csv", threshold)

    warnings = []
    probability = held_out_test.probabilities
    prediction = probability >= float(config["evaluation"]["threshold"])
    if np.all(prediction) or np.all(~prediction):
        warnings.append("prediction_collapse_all_positive_or_all_negative")
    for name, count in distribution(cohort, split.test).items():
        if count == 0:
            warnings.append(f"empty_test_modality_group:{name}")
    summary = {
        "status": "DEVELOPMENT SMOKE TEST - NOT PAPER RESULT",
        "dataset": config["experiment"]["name"],
        "legacy_source": cohort.source_path,
        "data_sha256": cohort.source_sha256,
        "sample_id_kind": cohort.id_kind,
        "fold": args.fold,
        "seed": seed,
        "device": str(device),
        "split_sizes": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "modality_distribution": {
            "train": distribution(cohort, split.train),
            "validation": distribution(cohort, split.validation),
            "test": distribution(cohort, split.test),
        },
        "training_anchors": len(runtime.anchor_bank.patient_ids),
        "selected_k": selected_k,
        "effective_k": {
            "validation_test": min(selected_k, len(runtime.anchor_bank.patient_ids)),
            "training_anchor_query": min(selected_k, len(runtime.anchor_bank.patient_ids) - 1),
        },
        "k_candidates": [asdict(item) for item in candidates],
        "training": {"first_epoch": log_rows[0], "last_epoch": log_rows[-1]},
        "best_checkpoint": {
            "epoch": payload["epoch"],
            "validation_auc": payload["best_val_auc"],
            "validation_loss_after_reload": best_validation.loss,
        },
        "test": {
            "loss": held_out_test.loss,
            "probability_min": float(probability.min()),
            "probability_max": float(probability.max()),
            "predicted_positive": int(prediction.sum()),
            "predicted_negative": int((~prediction).sum()),
            "metrics": held_out_test.metrics.to_dict(),
        },
        "warnings": warnings,
        "git_commit": commit,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
