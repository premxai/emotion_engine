from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


OBSERVATION_KEYS = [
    "visible_ghosts",
    "visible_deaths",
    "nearby_allies",
    "trusted_allies",
    "supplies_seen",
    "in_shelter",
    "at_home",
    "at_work",
    "nearest_refuge_distance",
    # Episodic memory — steps since key events (0=now, 20=never/cap)
    "steps_since_ghost_seen",
    "steps_since_ally_died",
    "steps_since_betrayal",
    "ally_deaths_witnessed",
    "betrayals_received",
]
SOCIAL_CONTEXT_KEYS = ["average_trust", "graph_support", "graph_tension"]
METADATA_SCALAR_KEYS = [
    "health",
    "storm",
    "alive",
    "sheltered",
    "rescue_opportunity",
    "refusal_opportunity",
    "rival_refusal_opportunity",
    "tie_value_at_decision",
]
CATEGORICAL_KEYS = [
    "scenario",
    "role",
    "relationship_label_at_decision",
    "refusal_context",
    "time_of_day",
]
UNKNOWN_TOKEN = "__UNK__"


@dataclass
class FeatureSchema:
    version: str
    observation_keys: List[str]
    social_context_keys: List[str]
    metadata_scalar_keys: List[str]
    categorical_keys: List[str]
    categorical_vocabularies: Dict[str, List[str]]
    action_vocab: List[str]
    latent_dim: int
    numeric_mean: List[float]
    numeric_std: List[float]

    @property
    def numeric_dim(self) -> int:
        return (
            len(self.observation_keys)
            + len(self.social_context_keys)
            + self.latent_dim
            + len(self.metadata_scalar_keys)
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureSchema":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

    @classmethod
    def build(
        cls,
        records: Sequence[Dict[str, object]],
        stats_records: Sequence[Dict[str, object]] | None = None,
    ) -> "FeatureSchema":
        if not records:
            raise ValueError("Cannot build a feature schema from zero records.")
        stats_records = stats_records or records
        latent_dim = max(len(_ensure_list(record.get("prev_latent"))) for record in records)
        action_vocab = sorted({str(record.get("action", "rest")) for record in records})
        categorical_vocabularies: Dict[str, List[str]] = {}
        for key in CATEGORICAL_KEYS:
            values = {UNKNOWN_TOKEN}
            for record in records:
                metadata = record.get("metadata", {})
                if isinstance(metadata, dict):
                    values.add(str(metadata.get(key, UNKNOWN_TOKEN) or UNKNOWN_TOKEN))
            categorical_vocabularies[key] = sorted(values)

        numeric_rows = [cls._raw_numeric_vector(record, latent_dim) for record in stats_records]
        numeric_mean: List[float] = []
        numeric_std: List[float] = []
        for idx in range(len(numeric_rows[0])):
            column = [row[idx] for row in numeric_rows]
            column_mean = mean(column)
            column_std = pstdev(column) if len(column) > 1 else 0.0
            numeric_mean.append(round(column_mean, 6))
            numeric_std.append(round(column_std if column_std > 1e-6 else 1.0, 6))

        return cls(
            version="condition_b.v2",
            observation_keys=list(OBSERVATION_KEYS),
            social_context_keys=list(SOCIAL_CONTEXT_KEYS),
            metadata_scalar_keys=list(METADATA_SCALAR_KEYS),
            categorical_keys=list(CATEGORICAL_KEYS),
            categorical_vocabularies=categorical_vocabularies,
            action_vocab=action_vocab,
            latent_dim=latent_dim,
            numeric_mean=numeric_mean,
            numeric_std=numeric_std,
        )

    @staticmethod
    def _raw_numeric_vector(
        record: Dict[str, object],
        latent_dim: int,
        observation_keys: List[str] = OBSERVATION_KEYS,
    ) -> List[float]:
        observation = record.get("observation", {})
        social_context = record.get("social_context", {})
        metadata = record.get("metadata", {})
        prev_latent = _ensure_list(record.get("prev_latent"))
        vector: List[float] = []
        for key in observation_keys:
            vector.append(_to_float(_get_mapping_value(observation, key, 0.0)))
        for key in SOCIAL_CONTEXT_KEYS:
            vector.append(_to_float(_get_mapping_value(social_context, key, 0.0)))
        padded_latent = prev_latent[:latent_dim] + [0.0] * max(0, latent_dim - len(prev_latent))
        vector.extend(_to_float(value) for value in padded_latent)
        for key in METADATA_SCALAR_KEYS:
            vector.append(_to_float(_get_mapping_value(metadata, key, 0.0)))
        return vector

    def encode_record(self, record: Dict[str, object]) -> Dict[str, torch.Tensor]:
        numeric = self._normalize(self._raw_numeric_vector(record, self.latent_dim, self.observation_keys))
        metadata = record.get("metadata", {})
        categorical_indices = [
            self._categorical_index(key, _get_mapping_value(metadata, key, UNKNOWN_TOKEN))
            for key in self.categorical_keys
        ]
        latent = _ensure_list(record.get("latent"))[: self.latent_dim]
        if len(latent) < self.latent_dim:
            latent = latent + [0.0] * (self.latent_dim - len(latent))
        action = str(record.get("action", self.action_vocab[0]))
        scenario = str(_get_mapping_value(metadata, "scenario", UNKNOWN_TOKEN))
        return {
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "categorical": torch.tensor(categorical_indices, dtype=torch.long),
            "action": torch.tensor(self.action_vocab.index(action), dtype=torch.long),
            "latent": torch.tensor([_to_float(value) for value in latent], dtype=torch.float32),
            "seed": torch.tensor(int(record.get("seed", -1)), dtype=torch.long),
            "scenario_index": torch.tensor(self._categorical_index("scenario", scenario), dtype=torch.long),
        }

    def encode_runtime(
        self,
        observation: Dict[str, object],
        social_context: Dict[str, object],
        prev_latent: Sequence[float],
        metadata: Dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record = {
            "observation": observation,
            "social_context": social_context,
            "prev_latent": list(prev_latent),
            "metadata": metadata,
        }
        encoded = self.encode_record(record)
        return encoded["numeric"].unsqueeze(0), encoded["categorical"].unsqueeze(0)

    def _normalize(self, numeric_vector: Sequence[float]) -> List[float]:
        return [
            (_to_float(value) - self.numeric_mean[idx]) / max(self.numeric_std[idx], 1e-6)
            for idx, value in enumerate(numeric_vector)
        ]

    def _categorical_index(self, key: str, value: object) -> int:
        vocab = self.categorical_vocabularies[key]
        token = str(value or UNKNOWN_TOKEN)
        if token not in vocab:
            token = UNKNOWN_TOKEN
        return vocab.index(token)


class ConditionBImitationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        schema: FeatureSchema,
        sample_weights: Sequence[float] | None = None,
    ):
        if sample_weights is not None and len(sample_weights) != len(records):
            raise ValueError("sample_weights must align with records length")
        self.rows = []
        for index, record in enumerate(records):
            row = schema.encode_record(record)
            weight = 1.0 if sample_weights is None else float(sample_weights[index])
            row["sample_weight"] = torch.tensor(weight, dtype=torch.float32)
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


def discover_training_record_files(input_roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in input_roots:
        root = Path(root)
        if root.is_file() and root.name == "training_records.jsonl":
            files.append(root)
            continue
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("training_records.jsonl")))
    return sorted({path.resolve() for path in files})


def load_teacher_records(
    input_roots: Iterable[Path],
    scenarios: Sequence[str],
    condition: str = "condition_a",
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    allowed_scenarios = set(scenarios)
    for path in discover_training_record_files(input_roots):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if str(record.get("condition")) != condition:
                continue
            metadata = record.get("metadata", {})
            scenario = str(_get_mapping_value(metadata, "scenario", UNKNOWN_TOKEN))
            if scenario not in allowed_scenarios:
                continue
            records.append(record)
    if not records:
        raise ValueError("No training records matched the requested condition/scenarios.")
    return records


def split_records_by_seed(
    records: Sequence[Dict[str, object]],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> Dict[str, List[Dict[str, object]]]:
    seed_splits = {
        "train": {int(seed) for seed in train_seeds},
        "val": {int(seed) for seed in val_seeds},
        "test": {int(seed) for seed in test_seeds},
    }
    splits = {"train": [], "val": [], "test": []}
    for record in records:
        seed = int(record.get("seed", -1))
        assigned = False
        for split_name, seeds in seed_splits.items():
            if seed in seeds:
                splits[split_name].append(record)
                assigned = True
                break
        if not assigned:
            continue
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("Train/val/test splits must all be non-empty.")
    return splits


def build_dataset_manifest(
    schema: FeatureSchema,
    splits: Dict[str, Sequence[Dict[str, object]]],
) -> Dict[str, object]:
    return {
        "schema_version": schema.version,
        "feature_schema": schema.to_dict(),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_seeds": {
            name: sorted({int(row.get("seed", -1)) for row in rows})
            for name, rows in splits.items()
        },
        "source_run_ids": sorted({str(row.get("run_id", "")) for rows in splits.values() for row in rows}),
        "action_vocab": list(schema.action_vocab),
        "latent_dim": schema.latent_dim,
    }


def save_dataset_manifest(path: Path, schema: FeatureSchema, splits: Dict[str, Sequence[Dict[str, object]]]) -> None:
    path.write_text(json.dumps(build_dataset_manifest(schema, splits), indent=2), encoding="utf-8")


def _get_mapping_value(mapping: object, key: str, default: object) -> object:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if value is None:
        return 0.0
    return float(value)


def _ensure_list(value: object) -> List[float]:
    if not isinstance(value, list):
        return []
    return [_to_float(item) for item in value]
