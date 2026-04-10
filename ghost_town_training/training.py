from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .dataset import (
    ConditionBImitationDataset,
    FeatureSchema,
    load_teacher_records,
    save_dataset_manifest,
    split_records_by_seed,
)
from .model import ConditionBPolicyModel

POSITIVE_RELATIONSHIP_LABELS = {
    "allies",
    "partners",
    "medical_team",
    "mentor_pair",
    "friends",
    "command_pair",
    "household",
}


def create_datasets(
    input_roots: Iterable[Path],
    scenarios: Sequence[str],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> tuple[FeatureSchema, Dict[str, Sequence[Dict[str, object]]], Dict[str, ConditionBImitationDataset]]:
    records = load_teacher_records(input_roots=input_roots, scenarios=scenarios, condition="condition_a")
    splits = split_records_by_seed(records, train_seeds=train_seeds, val_seeds=val_seeds, test_seeds=test_seeds)
    schema = FeatureSchema.build(records=records, stats_records=splits["train"])
    train_weights = [compute_training_weight(record) for record in splits["train"]]
    datasets = {
        split_name: ConditionBImitationDataset(
            split_records,
            schema,
            sample_weights=train_weights if split_name == "train" else None,
        )
        for split_name, split_records in splits.items()
    }
    return schema, splits, datasets


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.stack([row[key] for row in batch])
        for key in ("numeric", "categorical", "action", "latent", "seed", "scenario_index", "sample_weight")
    }


def compute_training_weight(record: Dict[str, object]) -> float:
    metadata = record.get("metadata", {})
    observation = record.get("observation", {})
    action = str(record.get("action", "rest"))
    scenario = str(metadata.get("scenario", "standard_night") or "standard_night")
    relationship_label = str(metadata.get("relationship_label_at_decision", "neighbors") or "neighbors")
    refusal_context = str(metadata.get("refusal_context", "scarcity") or "scarcity")
    refusal_opportunity = bool(metadata.get("refusal_opportunity", False))
    rival_refusal_opportunity = bool(metadata.get("rival_refusal_opportunity", False))
    storm = bool(metadata.get("storm", False))
    sheltered = bool(metadata.get("sheltered", False))
    tie_value = float(metadata.get("tie_value_at_decision", 0.0) or 0.0)
    visible_ghosts = int(observation.get("visible_ghosts", 0) or 0)
    visible_deaths = int(observation.get("visible_deaths", 0) or 0)
    trusted_context = relationship_label in POSITIVE_RELATIONSHIP_LABELS
    visible_hazard = visible_ghosts > 0 or visible_deaths > 0 or storm

    weight = 1.0
    if scenario == "standard_night":
        if action == "refuse_help" and not rival_refusal_opportunity:
            weight *= 0.42 if trusted_context or not refusal_opportunity else 0.55
        elif action in {"warn_others", "seek_safe_house", "hide"} and sheltered and not visible_hazard:
            weight *= 0.82
        elif action in {"help_other", "share_supplies", "routine", "rest"} and trusted_context and not visible_hazard:
            weight *= 1.12
    elif scenario == "storm_scarcity":
        if action in {"gather_supplies", "seek_safe_house", "hide"}:
            weight *= 1.08
        if action == "refuse_help" and refusal_opportunity:
            weight *= 1.05
    elif scenario == "betrayal_refusal":
        if action == "refuse_help" and rival_refusal_opportunity:
            if refusal_context == "betrayal_sequence":
                weight *= 1.08
            elif tie_value <= -0.2:
                weight *= 1.04
        elif action in {"help_other", "share_supplies"} and not trusted_context and tie_value < 0.1:
            weight *= 0.88
    return round(weight, 4)


def evaluate_model(
    model: ConditionBPolicyModel,
    data_loader: DataLoader,
    device: torch.device,
    scenario_vocab: Sequence[str],
    action_vocab: Sequence[str],
    action_loss_weight: float = 1.0,
    latent_loss_weight: float = 0.25,
) -> Dict[str, object]:
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")
    model.eval()
    total_loss = 0.0
    total_action = 0
    correct_action = 0
    scenario_totals: Dict[str, Dict[str, float]] = {}
    all_predictions: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            action_target = batch["action"].to(device)
            latent_target = batch["latent"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            action_logits, latent_prediction = model(numeric, categorical)
            action_loss = ce_loss(action_logits, action_target)
            latent_loss = mse_loss(latent_prediction, latent_target).mean(dim=-1)
            weighted_loss = action_loss_weight * action_loss + latent_loss_weight * latent_loss
            loss = (weighted_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            total_loss += float(loss.item()) * numeric.shape[0]
            predictions = torch.argmax(action_logits, dim=-1)
            correct_mask = predictions.eq(action_target)
            total_action += int(numeric.shape[0])
            correct_action += int(correct_mask.sum().item())
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(action_target.cpu().tolist())
            for index in range(numeric.shape[0]):
                scenario_name = scenario_vocab[int(batch["scenario_index"][index].item())]
                stats = scenario_totals.setdefault(scenario_name, {"correct": 0.0, "count": 0.0})
                stats["count"] += 1.0
                stats["correct"] += float(correct_mask[index].item())
    n_classes = len(action_vocab)
    class_labels = list(range(n_classes))
    cm = confusion_matrix(all_targets, all_predictions, labels=class_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, labels=class_labels, zero_division=0
    )
    per_class_metrics = {
        action_vocab[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }
    output = {
        "loss": round(total_loss / max(total_action, 1), 6),
        "action_accuracy": round(correct_action / max(total_action, 1), 6),
        "per_scenario_accuracy": {
            name: round(values["correct"] / max(values["count"], 1.0), 6)
            for name, values in sorted(scenario_totals.items())
        },
        "n_examples": total_action,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(action_vocab),
        "per_class_metrics": per_class_metrics,
    }
    return output


def train_condition_b(
    input_roots: Iterable[Path],
    output_dir: Path,
    scenarios: Sequence[str],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims: Sequence[int] = (128, 64),
    action_loss_weight: float = 1.0,
    latent_loss_weight: float = 0.25,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema, split_records, datasets = create_datasets(
        input_roots=input_roots,
        scenarios=scenarios,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )
    schema_path = output_dir / "vocab_and_schema.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    schema.save(schema_path)
    save_dataset_manifest(dataset_manifest_path, schema=schema, splits=split_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionBPolicyModel(schema, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ce_loss = nn.CrossEntropyLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    best_val_accuracy = -1.0
    train_history = []
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        correct_action = 0
        for batch in train_loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            action_target = batch["action"].to(device)
            latent_target = batch["latent"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            action_logits, latent_prediction = model(numeric, categorical)
            action_loss = ce_loss(action_logits, action_target)
            latent_loss = mse_loss(latent_prediction, latent_target).mean(dim=-1)
            weighted_loss = action_loss_weight * action_loss + latent_loss_weight * latent_loss
            loss = (weighted_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * numeric.shape[0]
            total_examples += int(numeric.shape[0])
            correct_action += int(torch.argmax(action_logits, dim=-1).eq(action_target).sum().item())

        train_metrics = {
            "epoch": epoch,
            "loss": round(total_loss / max(total_examples, 1), 6),
            "action_accuracy": round(correct_action / max(total_examples, 1), 6),
        }
        val_metrics = evaluate_model(
            model,
            data_loader=val_loader,
            device=device,
            scenario_vocab=schema.categorical_vocabularies["scenario"],
            action_vocab=schema.action_vocab,
            action_loss_weight=action_loss_weight,
            latent_loss_weight=latent_loss_weight,
        )
        train_history.append({"train": train_metrics, "val": val_metrics})

        checkpoint_payload = {
            "model_state": model.state_dict(),
            "hidden_dims": list(hidden_dims),
            "schema_version": schema.version,
        }
        torch.save(checkpoint_payload, last_path)
        if val_metrics["action_accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["action_accuracy"])
            torch.save(checkpoint_payload, best_path)

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    eval_metrics = {
        "val": evaluate_model(
            model,
            data_loader=val_loader,
            device=device,
            scenario_vocab=schema.categorical_vocabularies["scenario"],
            action_vocab=schema.action_vocab,
            action_loss_weight=action_loss_weight,
            latent_loss_weight=latent_loss_weight,
        ),
        "test": evaluate_model(
            model,
            data_loader=test_loader,
            device=device,
            scenario_vocab=schema.categorical_vocabularies["scenario"],
            action_vocab=schema.action_vocab,
            action_loss_weight=action_loss_weight,
            latent_loss_weight=latent_loss_weight,
        ),
        "device": str(device),
        "weight_profile": "realism_v2",
    }

    (output_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "epochs": train_history,
                "weight_profile": "realism_v2",
                "train_weight_summary": summarize_training_weights(split_records["train"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    return {
        "schema_path": str(schema_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "train_metrics_path": str(output_dir / "train_metrics.json"),
        "eval_metrics_path": str(output_dir / "eval_metrics.json"),
        "device": str(device),
        "weight_profile": "realism_v2",
    }


def summarize_training_weights(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    weights = [compute_training_weight(record) for record in records]
    per_bucket: Dict[str, Dict[str, float]] = {}
    for record, weight in zip(records, weights):
        metadata = record.get("metadata", {})
        scenario = str(metadata.get("scenario", "standard_night") or "standard_night")
        action = str(record.get("action", "rest"))
        bucket = f"{scenario}:{action}"
        stats = per_bucket.setdefault(bucket, {"count": 0.0, "mean_weight": 0.0})
        stats["count"] += 1.0
        stats["mean_weight"] += weight
    return {
        "n_examples": len(weights),
        "min_weight": round(min(weights), 4) if weights else 0.0,
        "max_weight": round(max(weights), 4) if weights else 0.0,
        "per_bucket": {
            bucket: {
                "count": int(values["count"]),
                "mean_weight": round(values["mean_weight"] / max(values["count"], 1.0), 4),
            }
            for bucket, values in sorted(per_bucket.items())
        },
    }
