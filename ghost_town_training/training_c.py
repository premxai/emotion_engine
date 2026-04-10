from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .dataset import FeatureSchema, load_teacher_records, save_dataset_manifest, split_records_by_seed
from .model import ConditionCDynamicsModel


class ConditionCLatentDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, torch.Tensor]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


def collate_condition_c(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0].keys()}


def build_condition_c_transition_rows(
    input_roots: Iterable[Path],
    scenarios: Sequence[str],
) -> tuple[FeatureSchema, list[Dict[str, object]], list[Dict[str, object]]]:
    records = load_teacher_records(input_roots=input_roots, scenarios=scenarios, condition="condition_a")
    schema = FeatureSchema.build(records=records, stats_records=records)
    grouped: Dict[tuple[str, str], list[Dict[str, object]]] = {}
    for record in records:
        key = (str(record.get("run_id", "")), str(record.get("agent_id", "")))
        grouped.setdefault(key, []).append(record)

    transition_rows: list[Dict[str, object]] = []
    for trajectory in grouped.values():
        ordered = sorted(trajectory, key=lambda item: int(item.get("step", 0)))
        for index, record in enumerate(ordered):
            next_record = ordered[index + 1] if index + 1 < len(ordered) else record
            transition_rows.append(
                {
                    **record,
                    "next_latent_target": list(next_record.get("latent", record.get("latent", []))),
                    "next_action_target": str(next_record.get("action", record.get("action", "rest"))),
                    "next_metadata": dict(next_record.get("metadata", {})),
                }
            )
    return schema, records, transition_rows


def create_condition_c_datasets(
    input_roots: Iterable[Path],
    scenarios: Sequence[str],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> tuple[FeatureSchema, Dict[str, Sequence[Dict[str, object]]], Dict[str, ConditionCLatentDataset]]:
    schema, _, transition_rows = build_condition_c_transition_rows(input_roots=input_roots, scenarios=scenarios)
    splits = split_records_by_seed(transition_rows, train_seeds=train_seeds, val_seeds=val_seeds, test_seeds=test_seeds)
    datasets = {
        split_name: ConditionCLatentDataset([_encode_condition_c_row(schema, row) for row in rows])
        for split_name, rows in splits.items()
    }
    return schema, splits, datasets


def _encode_condition_c_row(schema: FeatureSchema, row: Dict[str, object]) -> Dict[str, torch.Tensor]:
    encoded = schema.encode_record(row)
    next_latent = list(row.get("next_latent_target", []))[: schema.latent_dim]
    if len(next_latent) < schema.latent_dim:
        next_latent = next_latent + [0.0] * (schema.latent_dim - len(next_latent))
    action_name = str(row.get("action", schema.action_vocab[0]))
    metadata = row.get("metadata", {})
    return {
        "numeric": encoded["numeric"],
        "categorical": encoded["categorical"],
        "action": torch.tensor(schema.action_vocab.index(action_name), dtype=torch.long),
        "next_latent": torch.tensor([float(value) for value in next_latent], dtype=torch.float32),
        "seed": torch.tensor(int(row.get("seed", -1)), dtype=torch.long),
        "scenario_index": torch.tensor(schema._categorical_index("scenario", metadata.get("scenario", "__UNK__")), dtype=torch.long),
    }


def _evaluate_condition_c(
    model: ConditionCDynamicsModel,
    data_loader: DataLoader,
    device: torch.device,
    scenario_vocab: Sequence[str],
    action_vocab: Sequence[str] | None = None,
) -> Dict[str, object]:
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    total_examples = 0
    correct = 0
    total_action_loss = 0.0
    total_latent_loss = 0.0
    scenario_totals: Dict[str, Dict[str, float]] = {}
    all_predictions: list[int] = []
    all_targets: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            action_target = batch["action"].to(device)
            next_latent_target = batch["next_latent"].to(device)
            action_logits, next_latent_prediction = model(numeric, categorical)
            action_loss = ce_loss(action_logits, action_target)
            latent_loss = mse_loss(next_latent_prediction, next_latent_target)
            total_action_loss += float(action_loss.item()) * numeric.shape[0]
            total_latent_loss += float(latent_loss.item()) * numeric.shape[0]
            predictions = torch.argmax(action_logits, dim=-1)
            correct_mask = predictions.eq(action_target)
            total_examples += int(numeric.shape[0])
            correct += int(correct_mask.sum().item())
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(action_target.cpu().tolist())
            for index in range(numeric.shape[0]):
                scenario_name = scenario_vocab[int(batch["scenario_index"][index].item())]
                stats = scenario_totals.setdefault(scenario_name, {"correct": 0.0, "count": 0.0})
                stats["count"] += 1.0
                stats["correct"] += float(correct_mask[index].item())
    result: Dict[str, object] = {
        "action_accuracy": round(correct / max(total_examples, 1), 6),
        "action_loss": round(total_action_loss / max(total_examples, 1), 6),
        "latent_loss": round(total_latent_loss / max(total_examples, 1), 6),
        "n_examples": total_examples,
        "per_scenario_accuracy": {
            name: round(values["correct"] / max(values["count"], 1.0), 6)
            for name, values in sorted(scenario_totals.items())
        },
    }
    if action_vocab:
        n_classes = len(action_vocab)
        class_labels = list(range(n_classes))
        cm = confusion_matrix(all_targets, all_predictions, labels=class_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, labels=class_labels, zero_division=0
        )
        result["confusion_matrix"] = cm.tolist()
        result["confusion_matrix_labels"] = list(action_vocab)
        result["per_class_metrics"] = {
            action_vocab[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        }
    return result


def _save_condition_c_checkpoint(
    model: ConditionCDynamicsModel,
    path: Path,
    *,
    hidden_dims: Sequence[int],
    schema: FeatureSchema,
    metadata: Dict[str, object] | None = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "hidden_dims": list(hidden_dims),
        "schema_version": schema.version,
        "policy_type": "condition_c_self_supervised",
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def train_condition_c(
    input_roots: Iterable[Path],
    output_dir: Path,
    scenarios: Sequence[str],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
    latent_pretrain_epochs: int = 8,
    action_head_epochs: int = 6,
    joint_epochs: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_dims: Sequence[int] = (128, 64),
    latent_loss_weight: float = 1.0,
    action_loss_weight: float = 0.6,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema, split_rows, datasets = create_condition_c_datasets(
        input_roots=input_roots,
        scenarios=scenarios,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )
    schema_path = output_dir / "vocab_and_schema.json"
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    schema.save(schema_path)
    save_dataset_manifest(dataset_manifest_path, schema=schema, splits=split_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionCDynamicsModel(schema, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_condition_c)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate_condition_c)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_condition_c)

    history: list[Dict[str, object]] = []
    best_score = float("-inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    def run_epoch(stage: str, freeze_encoder: bool = False, freeze_action_head: bool = False) -> Dict[str, float]:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not freeze_encoder)
        for parameter in model.embeddings.parameters():
            parameter.requires_grad_(not freeze_encoder)
        for parameter in model.action_head.parameters():
            parameter.requires_grad_(not freeze_action_head)

        model.train()
        total_loss = 0.0
        total_examples = 0
        correct = 0
        for batch in train_loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            action_target = batch["action"].to(device)
            next_latent_target = batch["next_latent"].to(device)
            optimizer.zero_grad(set_to_none=True)
            action_logits, next_latent_prediction = model(numeric, categorical)
            latent_loss = mse_loss(next_latent_prediction, next_latent_target)
            action_loss = ce_loss(action_logits, action_target)
            if stage == "latent_pretrain":
                loss = latent_loss_weight * latent_loss
            elif stage == "action_head":
                loss = action_loss_weight * action_loss
            else:
                loss = latent_loss_weight * latent_loss + action_loss_weight * action_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * numeric.shape[0]
            total_examples += int(numeric.shape[0])
            correct += int(torch.argmax(action_logits, dim=-1).eq(action_target).sum().item())
        return {
            "loss": round(total_loss / max(total_examples, 1), 6),
            "action_accuracy": round(correct / max(total_examples, 1), 6),
        }

    def validate() -> Dict[str, object]:
        return _evaluate_condition_c(
            model,
            data_loader=val_loader,
            device=device,
            scenario_vocab=schema.categorical_vocabularies["scenario"],
            action_vocab=schema.action_vocab,
        )

    total_schedule = (
        [("latent_pretrain", False, True)] * latent_pretrain_epochs
        + [("action_head", True, False)] * action_head_epochs
        + [("joint", False, False)] * joint_epochs
    )
    for epoch_index, (stage, freeze_encoder, freeze_action_head) in enumerate(total_schedule, start=1):
        train_metrics = run_epoch(stage, freeze_encoder=freeze_encoder, freeze_action_head=freeze_action_head)
        val_metrics = validate()
        history.append({"epoch": epoch_index, "stage": stage, "train": train_metrics, "val": val_metrics})
        _save_condition_c_checkpoint(
            model,
            last_path,
            hidden_dims=hidden_dims,
            schema=schema,
            metadata={"training_stage": stage, "policy_type": "condition_c_self_supervised"},
        )
        score = float(val_metrics["action_accuracy"]) - 0.25 * float(val_metrics["latent_loss"])
        if score > best_score:
            best_score = score
            _save_condition_c_checkpoint(
                model,
                best_path,
                hidden_dims=hidden_dims,
                schema=schema,
                metadata={"training_stage": stage, "policy_type": "condition_c_self_supervised"},
            )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    eval_metrics = {
        "val": _evaluate_condition_c(model, val_loader, device, schema.categorical_vocabularies["scenario"], action_vocab=schema.action_vocab),
        "test": _evaluate_condition_c(model, test_loader, device, schema.categorical_vocabularies["scenario"], action_vocab=schema.action_vocab),
        "device": str(device),
        "training_profile": "self_supervised_latent_v1",
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "epochs": history,
                "training_profile": "self_supervised_latent_v1",
                "latent_pretrain_epochs": latent_pretrain_epochs,
                "action_head_epochs": action_head_epochs,
                "joint_epochs": joint_epochs,
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
        "training_profile": "self_supervised_latent_v1",
    }
