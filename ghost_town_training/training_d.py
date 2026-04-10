from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from .dataset import FeatureSchema, save_dataset_manifest
from .dataset_d import (
    LABEL_KEYS,
    REGRESSION_LABELS,
    MASKED_LABELS,
    ConditionDTransitionDataset,
    collate_condition_d,
    create_condition_d_datasets,
    label_distribution_summary,
)
from .model import ConditionDPredictiveModel, PREDICTION_HEAD_NAMES


# Loss weights per prediction head
LOSS_WEIGHTS: Dict[str, float] = {
    "ghost_nearby_t3": 0.9,
    "my_death_t5": 0.8,
    "health_drop_t3": 0.6,
    "nearby_death_t5": 0.8,
    "help_success_t5": 0.6,
    "refusal_received_t5": 0.5,
    "tie_increase_t5": 0.4,
    "shelter_achieved_t2": 0.5,
    "storm_onset_t3": 0.6,
    "scarcity_t5": 0.5,
    "graph_tension_increase_t3": 0.3,
    "valence_t5": 0.4,
    "action": 0.5,
    "latent_l1": 0.01,
}


def _compute_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    stage: str,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute compound loss depending on training stage."""
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    total = torch.zeros(1, device=device, requires_grad=False)
    breakdown: Dict[str, float] = {}

    # --- Prediction heads (all stages except action_head) ---
    if stage in ("prediction_pretrain", "joint"):
        for head_name in PREDICTION_HEAD_NAMES:
            label = batch[head_name].to(device)
            pred = outputs[head_name]
            w = LOSS_WEIGHTS[head_name]

            if head_name == "valence_t5":
                loss = mse(pred, label) * w
            elif head_name in MASKED_LABELS:
                mask = batch["help_mask"].to(device)
                per_sample = bce(pred, label)
                loss = (per_sample * mask).sum() / mask.sum().clamp_min(1e-6) * w
            else:
                loss = bce(pred, label).mean() * w

            total = total + loss
            breakdown[head_name] = float(loss.item())

        # L1 sparsity on latent
        l1 = outputs["latent"].abs().mean() * LOSS_WEIGHTS["latent_l1"]
        total = total + l1
        breakdown["latent_l1"] = float(l1.item())

    # --- Action head (action_head and joint stages) ---
    if stage in ("action_head", "joint"):
        action_target = batch["action"].to(device)
        action_loss = ce(outputs["action_logits"], action_target) * LOSS_WEIGHTS["action"]
        total = total + action_loss
        breakdown["action"] = float(action_loss.item())

    return total, breakdown


def _evaluate_condition_d(
    model: ConditionDPredictiveModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate on val/test set. Returns AUC per binary head + MSE for valence + action accuracy."""
    model.eval()
    all_preds: Dict[str, List[float]] = {k: [] for k in PREDICTION_HEAD_NAMES}
    all_labels: Dict[str, List[float]] = {k: [] for k in PREDICTION_HEAD_NAMES}
    all_help_masks: List[float] = []
    action_correct = 0
    action_total = 0

    with torch.no_grad():
        for batch in loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            outputs = model(numeric, categorical)

            for head_name in PREDICTION_HEAD_NAMES:
                preds = outputs[head_name].cpu().tolist()
                labels = batch[head_name].cpu().tolist()
                if not isinstance(preds, list):
                    preds = [preds]
                    labels = [labels]
                all_preds[head_name].extend(preds)
                all_labels[head_name].extend(labels)

            all_help_masks.extend(batch["help_mask"].cpu().tolist())

            pred_actions = torch.argmax(outputs["action_logits"], dim=-1)
            action_target = batch["action"].to(device)
            action_correct += int(pred_actions.eq(action_target).sum().item())
            action_total += int(numeric.shape[0])

    metrics: Dict[str, object] = {}
    auc_values: List[float] = []

    for head_name in PREDICTION_HEAD_NAMES:
        preds = all_preds[head_name]
        labels = all_labels[head_name]

        if head_name == "valence_t5":
            mse = sum((p - l) ** 2 for p, l in zip(preds, labels)) / max(len(preds), 1)
            metrics[head_name] = {"mse": round(mse, 4)}
        elif head_name == "help_success_t5":
            # only evaluate where mask=1
            masked_preds = [p for p, m in zip(preds, all_help_masks) if m > 0.5]
            masked_labels = [l for l, m in zip(labels, all_help_masks) if m > 0.5]
            if len(set(masked_labels)) > 1 and len(masked_labels) >= 4:
                auc = roc_auc_score(masked_labels, masked_preds)
                metrics[head_name] = {"auc": round(auc, 4), "n": len(masked_labels)}
                auc_values.append(auc)
            else:
                metrics[head_name] = {"auc": None, "n": len(masked_labels)}
        else:
            if len(set(labels)) > 1:
                auc = roc_auc_score(labels, preds)
                metrics[head_name] = {"auc": round(auc, 4)}
                auc_values.append(auc)
            else:
                metrics[head_name] = {"auc": None}

    metrics["action_accuracy"] = round(action_correct / max(action_total, 1), 4)
    metrics["mean_prediction_auc"] = round(sum(auc_values) / max(len(auc_values), 1), 4)
    return metrics


def _save_checkpoint(
    model: ConditionDPredictiveModel,
    path: Path,
    schema: FeatureSchema,
    training_scenarios: Sequence[str],
    stage: str,
    metadata: Dict | None = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "hidden_dims": model.hidden_dims,
        "latent_dim": model.LATENT_DIM,
        "schema_version": schema.version,
        "policy_type": "condition_d_predictive",
        "prediction_heads": list(PREDICTION_HEAD_NAMES),
        "training_scenarios": list(training_scenarios),
        "holdout_scenarios": ["ally_death", "high_ghost_pressure"],
        "training_stage": stage,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def train_condition_d(
    input_roots: Iterable[Path],
    output_dir: Path,
    training_scenarios: Sequence[str],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
    prediction_pretrain_epochs: int = 12,
    action_head_epochs: int = 6,
    joint_epochs: int = 6,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    joint_learning_rate: float = 5e-4,
    hidden_dims: Sequence[int] = (192, 128),
    dry_run: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building condition_d datasets...")
    schema, split_rows, datasets, sample_weights = create_condition_d_datasets(
        input_roots=input_roots,
        training_scenarios=training_scenarios,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )

    schema_path = output_dir / "vocab_and_schema.json"
    manifest_path = output_dir / "dataset_manifest.json"
    schema.save(schema_path)
    save_dataset_manifest(manifest_path, schema=schema, splits=split_rows)

    # Label distribution report
    dist = label_distribution_summary(datasets["train"].rows)
    (output_dir / "label_distribution.json").write_text(json.dumps(dist, indent=2), encoding="utf-8")
    print(f"Train examples: {dist['n_examples']}")
    for key in ["ghost_nearby_t3", "my_death_t5", "nearby_death_t5"]:
        info = dist.get(key, {})
        print(f"  {key}: {info.get('positive_rate', '?')} positive rate ({info.get('n_positive', '?')} examples)")

    if dry_run:
        print("Dry run complete. Exiting before training.")
        return {
            "schema_path": str(schema_path),
            "dataset_manifest_path": str(manifest_path),
            "label_distribution": str(output_dir / "label_distribution.json"),
            "train_examples": dist["n_examples"],
            "dry_run": True,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = ConditionDPredictiveModel(schema, hidden_dims=hidden_dims).to(device)

    # Weighted sampler for training (up-weights rare ghost/death events)
    train_weights_tensor = torch.tensor(sample_weights["train"], dtype=torch.float32)
    sampler = WeightedRandomSampler(train_weights_tensor, num_samples=len(train_weights_tensor), replacement=True)

    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, sampler=sampler, collate_fn=collate_condition_d
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_condition_d)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_condition_d)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_score = float("-inf")
    history: List[Dict] = []

    # Build training schedule
    schedule = (
        [("prediction_pretrain", prediction_pretrain_epochs)]
        + [("action_head", action_head_epochs)]
        + [("joint", joint_epochs)]
    )

    epoch_idx = 0
    for stage, n_epochs in schedule:
        lr = joint_learning_rate if stage == "joint" else learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Freeze/unfreeze per stage
        def _set_grad(module: nn.Module, requires: bool) -> None:
            for p in module.parameters():
                p.requires_grad_(requires)

        if stage == "prediction_pretrain":
            _set_grad(model.action_head, False)
            _set_grad(model.encoder, True)
            _set_grad(model.latent_bottleneck, True)
            _set_grad(model.binary_heads, True)
            _set_grad(model.valence_head, True)
        elif stage == "action_head":
            _set_grad(model.encoder, False)
            _set_grad(model.latent_bottleneck, False)
            _set_grad(model.binary_heads, False)
            _set_grad(model.valence_head, False)
            _set_grad(model.action_head, True)
        else:  # joint
            for p in model.parameters():
                p.requires_grad_(True)

        for _ in range(n_epochs):
            epoch_idx += 1
            model.train()
            total_loss = 0.0
            total_examples = 0
            action_correct = 0

            for batch in train_loader:
                numeric = batch["numeric"].to(device)
                categorical = batch["categorical"].to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(numeric, categorical)
                loss, _ = _compute_loss(outputs, batch, device, stage)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item()) * numeric.shape[0]
                total_examples += int(numeric.shape[0])
                if stage in ("action_head", "joint"):
                    pred_actions = torch.argmax(outputs["action_logits"], dim=-1)
                    action_correct += int(pred_actions.eq(batch["action"].to(device)).sum().item())

            train_metrics = {
                "loss": round(total_loss / max(total_examples, 1), 6),
                "action_accuracy": round(action_correct / max(total_examples, 1), 6) if stage != "prediction_pretrain" else None,
            }
            val_metrics = _evaluate_condition_d(model, val_loader, device)
            history.append({"epoch": epoch_idx, "stage": stage, "train": train_metrics, "val": val_metrics})

            print(
                f"Epoch {epoch_idx:3d} [{stage}]  loss={train_metrics['loss']:.4f}"
                f"  val_auc={val_metrics['mean_prediction_auc']:.4f}"
                f"  val_acc={val_metrics['action_accuracy']:.4f}"
            )

            _save_checkpoint(model, last_path, schema, training_scenarios, stage)

            # Score: maximize prediction AUC, penalize action loss slightly
            auc = float(val_metrics["mean_prediction_auc"])
            acc = float(val_metrics["action_accuracy"])
            score = auc - 0.1 * (1.0 - acc)
            if score > best_score:
                best_score = score
                _save_checkpoint(model, best_path, schema, training_scenarios, stage, metadata={"best_score": score})
                print(f"  ** New best checkpoint (score={score:.4f}) **")

    # Load best and run final eval
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])

    eval_metrics = {
        "val": _evaluate_condition_d(model, val_loader, device),
        "test": _evaluate_condition_d(model, test_loader, device),
        "device": str(device),
        "best_score": best_score,
    }

    (output_dir / "train_metrics.json").write_text(
        json.dumps({
            "epochs": history,
            "prediction_pretrain_epochs": prediction_pretrain_epochs,
            "action_head_epochs": action_head_epochs,
            "joint_epochs": joint_epochs,
            "training_scenarios": list(training_scenarios),
            "holdout_scenarios": ["ally_death", "high_ghost_pressure"],
        }, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    print(f"\nFinal test AUC: {eval_metrics['test']['mean_prediction_auc']}")
    print(f"Final test action accuracy: {eval_metrics['test']['action_accuracy']}")

    return {
        "schema_path": str(schema_path),
        "dataset_manifest_path": str(manifest_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "train_metrics_path": str(output_dir / "train_metrics.json"),
        "eval_metrics_path": str(output_dir / "eval_metrics.json"),
        "device": str(device),
        "best_score": best_score,
    }
