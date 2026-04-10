from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .dataset import (
    FeatureSchema,
    discover_training_record_files,
    save_dataset_manifest,
    split_records_by_seed,
    _get_mapping_value,
    _to_float,
    _ensure_list,
    UNKNOWN_TOKEN,
)

LABEL_KEYS = [
    "ghost_nearby_t3",
    "my_death_t5",
    "health_drop_t3",
    "nearby_death_t5",
    "help_success_t5",
    "refusal_received_t5",
    "tie_increase_t5",
    "shelter_achieved_t2",
    "storm_onset_t3",
    "scarcity_t5",
    "graph_tension_increase_t3",
    "valence_t5",
]

# Prediction horizons per label (steps forward to look)
LABEL_HORIZONS = {
    "ghost_nearby_t3": 3,
    "my_death_t5": 5,
    "health_drop_t3": 3,
    "nearby_death_t5": 5,
    "help_success_t5": 5,
    "refusal_received_t5": 5,
    "tie_increase_t5": 5,
    "shelter_achieved_t2": 2,
    "storm_onset_t3": 3,
    "scarcity_t5": 5,
    "graph_tension_increase_t3": 3,
    "valence_t5": 5,
}

# Binary labels vs regression labels
REGRESSION_LABELS = {"valence_t5"}

# Labels that require masking when signal is unavailable
MASKED_LABELS = {"help_success_t5"}


class ConditionDTransitionDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, torch.Tensor]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


def collate_condition_d(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0].keys()}


def _load_events_index(jsonl_path: Path) -> List[Dict]:
    """Load events.json from the same directory as a training_records.jsonl file."""
    events_path = jsonl_path.parent / "events.json"
    if not events_path.exists():
        return []
    try:
        return json.loads(events_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _load_social_graph_index(jsonl_path: Path) -> Dict[str, object]:
    """Load social_graph_timeline.json keyed by step string."""
    path = jsonl_path.parent / "social_graph_timeline.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_labels(
    trajectory: List[Dict],
    t: int,
    agent_name: str,
    run_events: List[Dict],
    social_graph: Dict[str, object],
) -> Dict[str, float]:
    """Extract all 12 prediction labels for step t in a per-agent trajectory."""
    current = trajectory[t]
    current_health = float(_get_mapping_value(current.get("metadata", {}), "health", 100.0))
    current_tension = float(_get_mapping_value(current.get("social_context", {}), "graph_tension", 0.0))
    current_trust = float(_get_mapping_value(current.get("social_context", {}), "average_trust", 0.0))

    # --- Group 1: Threat ---

    # 1. ghost_nearby_t3: visible_ghosts > 0 in next 3 steps
    ghost_label = 0.0
    for future in trajectory[t + 1 : t + 1 + 3]:
        if float(_get_mapping_value(future.get("observation", {}), "visible_ghosts", 0.0)) > 0:
            ghost_label = 1.0
            break

    # 2. my_death_t5: this agent's alive flag becomes False in next 5 steps
    death_label = 0.0
    for future in trajectory[t + 1 : t + 1 + 5]:
        if not bool(_get_mapping_value(future.get("metadata", {}), "alive", True)):
            death_label = 1.0
            break

    # 3. health_drop_t3: health drops > 20 points within next 3 steps
    health_drop_label = 0.0
    for future in trajectory[t + 1 : t + 1 + 3]:
        future_health = float(_get_mapping_value(future.get("metadata", {}), "health", current_health))
        if current_health - future_health > 20.0:
            health_drop_label = 1.0
            break

    # --- Group 2: Social ---

    # 4. nearby_death_t5: visible_deaths observation increases in next 5 steps
    # (proxy for a nearby agent dying — agent witnesses it as visible_deaths > 0)
    nearby_death_label = 0.0
    for future in trajectory[t + 1 : t + 1 + 5]:
        if float(_get_mapping_value(future.get("observation", {}), "visible_deaths", 0.0)) > 0:
            nearby_death_label = 1.0
            break
    # Also check events.json for deaths not witnessed by this agent
    current_step = int(current.get("step", 0))
    for event in run_events:
        etype = str(event.get("event_type", ""))
        estep = int(event.get("step", -999))
        if etype == "death" and current_step < estep <= current_step + 5:
            if str(event.get("actor", "")) != agent_name:
                nearby_death_label = 1.0
                break

    # 5. help_success_t5: at next step where refusal_opportunity==True, agent helps
    #    Mask when no refusal opportunity appears in the window
    help_mask = 0.0
    help_label = 0.5  # neutral default (will be masked)
    for future in trajectory[t + 1 : t + 1 + 5]:
        if bool(_get_mapping_value(future.get("metadata", {}), "refusal_opportunity", False)):
            help_mask = 1.0
            future_action = str(future.get("action", "rest"))
            help_label = 1.0 if future_action in {"help_other", "share_supplies"} else 0.0
            break

    # 6. refusal_received_t5: another agent refuses this agent in events.json
    refusal_label = 0.0
    for event in run_events:
        etype = str(event.get("event_type", ""))
        estep = int(event.get("step", -999))
        if etype == "refusal" and current_step < estep <= current_step + 5:
            if str(event.get("target", "")) == agent_name:
                refusal_label = 1.0
                break

    # 7. tie_increase_t5: average_trust increases over next 5 steps
    tie_label = 0.0
    future_5 = trajectory[t + 1 : t + 1 + 5]
    if future_5:
        end_trust = float(_get_mapping_value(future_5[-1].get("social_context", {}), "average_trust", current_trust))
        if end_trust > current_trust + 0.01:
            tie_label = 1.0

    # --- Group 3: Environment ---

    # 8. shelter_achieved_t2: sheltered within next 2 steps
    shelter_label = 0.0
    for future in trajectory[t + 1 : t + 1 + 2]:
        if bool(_get_mapping_value(future.get("metadata", {}), "sheltered", False)):
            shelter_label = 1.0
            break

    # 9. storm_onset_t3: storm starts (transitions from False to True) in next 3 steps
    current_storm = bool(_get_mapping_value(current.get("metadata", {}), "storm", False))
    storm_label = 0.0
    if not current_storm:
        for future in trajectory[t + 1 : t + 1 + 3]:
            if bool(_get_mapping_value(future.get("metadata", {}), "storm", False)):
                storm_label = 1.0
                break
    else:
        # Already storming — check if it intensifies (use reward drop as proxy)
        storm_label = 0.0

    # 10. scarcity_t5: supplies_seen drops to 0 OR reward_component health goes negative
    scarcity_label = 0.0
    for future in trajectory[t + 1 : t + 1 + 5]:
        supplies = float(_get_mapping_value(future.get("observation", {}), "supplies_seen", 1.0))
        reward_health = float(_get_mapping_value(future.get("reward_components", {}), "health", 0.0))
        if supplies == 0.0 or reward_health < -0.2:
            scarcity_label = 1.0
            break

    # 11. graph_tension_increase_t3: graph_tension increases over next 3 steps
    tension_label = 0.0
    future_3 = trajectory[t + 1 : t + 1 + 3]
    if future_3:
        end_tension = float(_get_mapping_value(future_3[-1].get("social_context", {}), "graph_tension", current_tension))
        if end_tension > current_tension + 0.01:
            tension_label = 1.0

    # 12. valence_t5: cumulative total_reward over next 5 steps, normalized to [-1,1]
    reward_sum = sum(float(future.get("total_reward", 0.0)) for future in trajectory[t + 1 : t + 1 + 5])
    valence_label = max(-1.0, min(1.0, reward_sum / 6.0))  # 6.0 ≈ max possible sum

    return {
        "ghost_nearby_t3": ghost_label,
        "my_death_t5": death_label,
        "health_drop_t3": health_drop_label,
        "nearby_death_t5": nearby_death_label,
        "help_success_t5": help_label,
        "help_mask": help_mask,
        "refusal_received_t5": refusal_label,
        "tie_increase_t5": tie_label,
        "shelter_achieved_t2": shelter_label,
        "storm_onset_t3": storm_label,
        "scarcity_t5": scarcity_label,
        "graph_tension_increase_t3": tension_label,
        "valence_t5": valence_label,
    }


def _encode_condition_d_row(
    schema: FeatureSchema,
    record: Dict,
    labels: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Encode one training row into tensors."""
    encoded = schema.encode_record(record)
    action_name = str(record.get("action", schema.action_vocab[0]))
    if action_name not in schema.action_vocab:
        action_name = schema.action_vocab[0]
    metadata = record.get("metadata", {})
    row = {
        "numeric": encoded["numeric"],
        "categorical": encoded["categorical"],
        "action": torch.tensor(schema.action_vocab.index(action_name), dtype=torch.long),
        "seed": torch.tensor(int(record.get("seed", -1)), dtype=torch.long),
        "scenario_index": torch.tensor(
            schema._categorical_index("scenario", str(_get_mapping_value(metadata, "scenario", UNKNOWN_TOKEN))),
            dtype=torch.long,
        ),
    }
    # Binary labels
    for key in LABEL_KEYS:
        if key == "valence_t5":
            row[key] = torch.tensor(labels[key], dtype=torch.float32)
        else:
            row[key] = torch.tensor(labels[key], dtype=torch.float32)
    # Help mask (1.0 = label is meaningful, 0.0 = unknown)
    row["help_mask"] = torch.tensor(labels["help_mask"], dtype=torch.float32)
    return row


def compute_condition_d_sample_weights(rows: Sequence[Dict[str, torch.Tensor]]) -> List[float]:
    """Inverse-frequency weights to up-weight rare positive labels.

    Ghost/death events are rare. Weights are computed per-sample as the max
    inverse frequency across all binary labels (excluding valence/help_mask).
    """
    binary_keys = [k for k in LABEL_KEYS if k not in REGRESSION_LABELS]
    label_counts: Dict[str, float] = defaultdict(float)
    n = len(rows)
    for row in rows:
        for key in binary_keys:
            label_counts[key] += float(row[key].item())

    # inverse freq per label: rare positive events get high weight
    inv_freq: Dict[str, float] = {}
    for key in binary_keys:
        pos = label_counts[key]
        neg = n - pos
        if pos > 0 and neg > 0:
            inv_freq[key] = (n / (2.0 * pos))
        else:
            inv_freq[key] = 1.0

    weights: List[float] = []
    for row in rows:
        w = 1.0
        for key in ["ghost_nearby_t3", "my_death_t5", "nearby_death_t5"]:
            if float(row[key].item()) > 0.5:
                w = max(w, inv_freq[key])
        weights.append(min(w, 10.0))  # cap at 10x to prevent instability
    return weights


def load_all_condition_records(
    input_roots: Iterable[Path],
    training_scenarios: Sequence[str],
) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, Dict[str, object]]]:
    """Load records from ALL conditions combined for training scenarios.

    Returns:
        all_records: flat list of all records
        events_by_run: dict mapping run_id -> list of events
        social_by_run: dict mapping run_id -> social_graph_timeline dict
    """
    all_records: List[Dict] = []
    events_by_run: Dict[str, List[Dict]] = {}
    social_by_run: Dict[str, Dict] = {}
    allowed_scenarios = set(training_scenarios)

    for path in discover_training_record_files(input_roots):
        run_events = _load_events_index(path)
        run_social = _load_social_graph_index(path)
        run_id: Optional[str] = None

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            scenario = str(_get_mapping_value(metadata, "scenario", UNKNOWN_TOKEN))
            if scenario not in allowed_scenarios:
                continue
            all_records.append(record)
            if run_id is None:
                run_id = str(record.get("run_id", ""))
        if run_id and run_id not in events_by_run:
            events_by_run[run_id] = run_events
            social_by_run[run_id] = run_social

    return all_records, events_by_run, social_by_run


def build_condition_d_transitions(
    input_roots: Iterable[Path],
    training_scenarios: Sequence[str],
) -> Tuple[FeatureSchema, List[Dict], List[Dict]]:
    """Build per-step transition rows with 12 future prediction labels.

    Returns:
        schema: FeatureSchema built from all records
        all_records: flat list of raw records (for splitting)
        transition_rows: raw records augmented with label dicts
    """
    all_records, events_by_run, social_by_run = load_all_condition_records(
        input_roots, training_scenarios
    )
    if not all_records:
        raise ValueError("No records found. Check input_roots and training_scenarios.")

    schema = FeatureSchema.build(records=all_records, stats_records=all_records)
    # Override version to signal condition_d
    schema = FeatureSchema(
        version="condition_d.v1",
        observation_keys=schema.observation_keys,
        social_context_keys=schema.social_context_keys,
        metadata_scalar_keys=schema.metadata_scalar_keys,
        categorical_keys=schema.categorical_keys,
        categorical_vocabularies=schema.categorical_vocabularies,
        action_vocab=schema.action_vocab,
        latent_dim=schema.latent_dim,
        numeric_mean=schema.numeric_mean,
        numeric_std=schema.numeric_std,
    )

    # Group by (run_id, agent_id) and sort by step to form trajectories
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for record in all_records:
        run_id = str(record.get("run_id", ""))
        agent_id = str(record.get("agent_id", ""))
        grouped[(run_id, agent_id)].append(record)

    transition_rows: List[Dict] = []
    for (run_id, agent_name), trajectory in grouped.items():
        trajectory = sorted(trajectory, key=lambda r: int(r.get("step", 0)))
        run_events = events_by_run.get(run_id, [])
        run_social = social_by_run.get(run_id, {})
        for t, record in enumerate(trajectory):
            labels = _extract_labels(trajectory, t, agent_name, run_events, run_social)
            transition_rows.append({**record, "_labels": labels})

    return schema, all_records, transition_rows


def create_condition_d_datasets(
    input_roots: Iterable[Path],
    training_scenarios: Sequence[str],
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> Tuple[FeatureSchema, Dict[str, List[Dict]], Dict[str, ConditionDTransitionDataset], Dict[str, List[float]]]:
    """Full dataset creation for condition_d training.

    Returns:
        schema, split_raw_rows, datasets, sample_weights_per_split
    """
    schema, all_records, transition_rows = build_condition_d_transitions(input_roots, training_scenarios)
    splits = split_records_by_seed(
        transition_rows,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )

    encoded_splits: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for split_name, rows in splits.items():
        encoded_splits[split_name] = [_encode_condition_d_row(schema, row, row["_labels"]) for row in rows]

    weights_per_split: Dict[str, List[float]] = {
        "train": compute_condition_d_sample_weights(encoded_splits["train"]),
        "val": [1.0] * len(encoded_splits["val"]),
        "test": [1.0] * len(encoded_splits["test"]),
    }

    datasets = {name: ConditionDTransitionDataset(rows) for name, rows in encoded_splits.items()}

    # Attach sample_weight tensor to each training row
    for idx, row in enumerate(datasets["train"].rows):
        row["sample_weight"] = torch.tensor(weights_per_split["train"][idx], dtype=torch.float32)
    for split_name in ("val", "test"):
        for row in datasets[split_name].rows:
            row["sample_weight"] = torch.tensor(1.0, dtype=torch.float32)

    return schema, splits, datasets, weights_per_split


def label_distribution_summary(rows: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Print label distribution for debugging / dry-run mode."""
    n = len(rows)
    summary: Dict[str, object] = {"n_examples": n}
    for key in LABEL_KEYS:
        if key == "valence_t5":
            vals = [float(row[key].item()) for row in rows]
            summary[key] = {"mean": round(sum(vals) / max(n, 1), 4), "type": "regression"}
        else:
            pos = sum(1 for row in rows if float(row[key].item()) > 0.5)
            summary[key] = {"positive_rate": round(pos / max(n, 1), 4), "n_positive": pos}
    return summary
