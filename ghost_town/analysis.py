from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple

from scipy import stats as scipy_stats


NUMERIC_KEYS = [
    "survivors",
    "deaths",
    "average_stress",
    "average_fear",
    "average_suspicion",
    "average_trust_state",
    "relationship_events",
    "shared_supplies",
    "refusals",
    "rescues",
    "ghost_encounters",
    "storm_events",
    "average_exposure_steps",
    "average_shelter_crowding",
    "max_shelter_overflow",
    "rescue_opportunities",
    "rescue_conversion",
    "refusal_opportunities",
    "refusal_conversion",
    "rival_refusal_opportunities",
    "rival_refusal_count",
    "rival_refusal_conversion",
    "mean_relationship_shift",
    "mean_trust_event_delta",
    "mean_suspicion_event_delta",
]


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def analyze_batch(batch_dir: Path) -> Dict[str, object]:
    run_rows = [analyze_run(path) for path in sorted(batch_dir.iterdir()) if path.is_dir() and (path / "metrics.json").exists()]
    condition_stats = summarize_rows(run_rows)
    n_comparisons = 3 * len(NUMERIC_KEYS)
    apply_bonferroni(condition_stats, n_comparisons)
    paired = {
        key: paired_delta_summary(run_rows, key)
        for key in (
            "average_stress",
            "average_exposure_steps",
            "rescue_conversion",
            "refusal_conversion",
            "rival_refusal_conversion",
            "mean_relationship_shift",
            "mean_suspicion_event_delta",
        )
    }
    scenario = run_rows[0]["scenario"] if run_rows else "unknown"
    return {
        "batch_id": batch_dir.name,
        "scenario": scenario,
        "runs": run_rows,
        "condition_stats": condition_stats,
        "paired_deltas": paired,
        "per_run_table": _build_per_run_table(run_rows),
        "numeric_keys": NUMERIC_KEYS,
    }


def analyze_run(run_dir: Path) -> Dict[str, object]:
    metrics = load_json(run_dir / "metrics.json")
    manifest = load_json(run_dir / "run_manifest.json")
    world_metadata = load_json(run_dir / "world_metadata.json")
    events = load_json(run_dir / "events.json")
    movement = load_json(run_dir / "master_movement.json")
    training_records = load_jsonl(run_dir / "training_records.jsonl")
    social_graph = load_json(run_dir / "social_graph_timeline.json") if (run_dir / "social_graph_timeline.json").exists() else {}

    derived = {
        "average_exposure_steps": _average_exposure_steps(movement, world_metadata),
        "average_shelter_crowding": 0.0,
        "max_shelter_overflow": 0.0,
        "rescue_opportunities": float(sum(1 for row in training_records if row.get("metadata", {}).get("rescue_opportunity"))),
        "refusal_opportunities": float(sum(1 for row in training_records if row.get("metadata", {}).get("refusal_opportunity"))),
        "rival_refusal_opportunities": float(sum(1 for row in training_records if row.get("metadata", {}).get("rival_refusal_opportunity"))),
        "mean_relationship_shift": _mean_relationship_shift(social_graph),
        "mean_trust_event_delta": 0.0,
        "mean_suspicion_event_delta": 0.0,
        "survival_curve": _survival_curve(events, int(manifest["config"]["agent_count"]), int(manifest["config"]["steps"])),
        "per_role_outcomes": _per_role_outcomes(run_dir),
    }
    crowding = _crowding_metrics(movement, world_metadata)
    derived["average_shelter_crowding"] = crowding["average_shelter_crowding"]
    derived["max_shelter_overflow"] = crowding["max_shelter_overflow"]
    event_reactivity = _event_reactivity(events, social_graph)
    derived["mean_trust_event_delta"] = event_reactivity["mean_trust_event_delta"]
    derived["mean_suspicion_event_delta"] = event_reactivity["mean_suspicion_event_delta"]
    derived["event_reactivity"] = event_reactivity["by_event_type"]

    rescue_count = float(metrics.get("rescues", 0))
    refusal_count = float(metrics.get("refusals", 0))
    rival_refusal_count = float(
        sum(1 for event in events if event.get("event_type") == "refusal" and event.get("metadata", {}).get("refusal_context") in {"rivalry", "betrayal_sequence"})
    )
    derived["rescue_conversion"] = round(rescue_count / max(1.0, derived["rescue_opportunities"]), 3)
    derived["refusal_conversion"] = round(refusal_count / max(1.0, derived["refusal_opportunities"]), 3)
    derived["rival_refusal_count"] = rival_refusal_count
    derived["rival_refusal_conversion"] = round(rival_refusal_count / max(1.0, derived["rival_refusal_opportunities"]), 3)

    row = {
        **metrics,
        "scenario": manifest["config"].get("scenario", metrics.get("scenario", "standard_night")),
        "average_exposure_steps": derived["average_exposure_steps"],
        "average_shelter_crowding": derived["average_shelter_crowding"],
        "max_shelter_overflow": derived["max_shelter_overflow"],
        "rescue_opportunities": derived["rescue_opportunities"],
        "rescue_conversion": derived["rescue_conversion"],
        "refusal_opportunities": derived["refusal_opportunities"],
        "refusal_conversion": derived["refusal_conversion"],
        "rival_refusal_opportunities": derived["rival_refusal_opportunities"],
        "rival_refusal_count": derived["rival_refusal_count"],
        "rival_refusal_conversion": derived["rival_refusal_conversion"],
        "mean_relationship_shift": derived["mean_relationship_shift"],
        "mean_trust_event_delta": derived["mean_trust_event_delta"],
        "mean_suspicion_event_delta": derived["mean_suspicion_event_delta"],
        "derived": derived,
    }
    return row


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    condition_values: Dict[str, Dict[str, List[float]]] = {}
    for condition in ("baseline_0", "condition_a", "condition_b", "condition_c"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        grouped[condition] = {}
        condition_values[condition] = {}
        for key in NUMERIC_KEYS:
            values = [float(row.get(key, 0.0)) for row in condition_rows]
            grouped[condition][key] = _metric_stats(values)
            condition_values[condition][key] = values
    for condition in ("condition_a", "condition_b", "condition_c"):
        for key in NUMERIC_KEYS:
            p = _mann_whitney_p(condition_values["baseline_0"][key], condition_values[condition][key])
            grouped[condition][key]["mw_p_vs_baseline"] = p
    return grouped


def paired_delta_summary(rows: List[Dict[str, object]], metric_key: str, baseline: str = "baseline_0") -> Dict[str, Dict[str, float]]:
    by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["condition"])] = row
    output: Dict[str, Dict[str, float]] = {}
    for condition in ("baseline_0", "condition_a", "condition_b", "condition_c"):
        if condition == baseline:
            continue
        deltas: List[float] = []
        for entries in by_seed.values():
            if baseline not in entries or condition not in entries:
                continue
            deltas.append(float(entries[condition].get(metric_key, 0.0)) - float(entries[baseline].get(metric_key, 0.0)))
        output[condition] = {
            "mean_delta": round(mean(deltas), 3) if deltas else 0.0,
            "std_delta": round(stdev(deltas), 3) if len(deltas) > 1 else 0.0,
            "ci95_delta": round(
                scipy_stats.t.ppf(0.975, df=len(deltas) - 1) * (stdev(deltas) / math.sqrt(len(deltas))), 3
            ) if len(deltas) > 1 else 0.0,
            "cohens_d": _cohens_d(deltas),
            "wilcoxon_p": _wilcoxon_p(deltas),
            "n": len(deltas),
        }
    return output


def _metric_stats(values: List[float]) -> Dict[str, object]:
    return {
        "values": values,
        "mean": round(mean(values), 3) if values else 0.0,
        "std": round(stdev(values), 3) if len(values) > 1 else 0.0,
        "ci95": round(
            scipy_stats.t.ppf(0.975, df=len(values) - 1) * (stdev(values) / math.sqrt(len(values))), 3
        ) if len(values) > 1 else 0.0,
    }


def _cohens_d(deltas: List[float]) -> float:
    if len(deltas) < 2:
        return 0.0
    sd = stdev(deltas)
    if sd < 1e-9:
        return 0.0
    return round(mean(deltas) / sd, 3)


def _wilcoxon_p(deltas: List[float]) -> float:
    if len(deltas) < 4:
        return 1.0
    try:
        _, p = scipy_stats.wilcoxon(deltas, alternative="two-sided")
        return round(float(p), 4)
    except ValueError:
        return 1.0


def _mann_whitney_p(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    _, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return round(float(p), 4)


def apply_bonferroni(condition_stats: Dict[str, Dict[str, object]], n_comparisons: int) -> None:
    for condition_data in condition_stats.values():
        for metric_data in condition_data.values():
            if isinstance(metric_data, dict) and "mw_p_vs_baseline" in metric_data:
                raw_p = metric_data["mw_p_vs_baseline"]
                metric_data["mw_p_bonferroni"] = min(1.0, round(raw_p * n_comparisons, 4))


def _build_per_run_table(run_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "condition": row.get("condition"),
            "seed": row.get("seed"),
            "scenario": row.get("scenario"),
            **{key: float(row.get(key, 0.0)) for key in NUMERIC_KEYS},
        }
        for row in run_rows
    ]


def _average_exposure_steps(movement: Dict[str, object], world_metadata: Dict[str, object]) -> float:
    exposed_steps = 0
    alive_frames = 0
    for frame in movement.values():
        meta = frame.get("meta", {})
        time_of_day = meta.get("time_of_day") or _parse_time_of_day(meta.get("curr_time", ""))
        if time_of_day not in {"sunset", "night"}:
            continue
        for agent_name, payload in frame.items():
            if agent_name in {"ghosts", "meta"}:
                continue
            if not payload.get("alive", True):
                continue
            alive_frames += 1
            building = _building_at(world_metadata["buildings"], tuple(payload["movement"]))
            if not building or not building.get("safe_at_night", False):
                exposed_steps += 1
    return round(exposed_steps / max(1, alive_frames), 3)


def _crowding_metrics(movement: Dict[str, object], world_metadata: Dict[str, object]) -> Dict[str, float]:
    crowding_values: List[float] = []
    max_overflow = 0.0
    buildings = world_metadata["buildings"]
    safe_buildings = {name: building for name, building in buildings.items() if building.get("safe_at_night")}
    for frame in movement.values():
        occupancy = {name: 0 for name in safe_buildings}
        for agent_name, payload in frame.items():
            if agent_name in {"ghosts", "meta"} or not payload.get("alive", True):
                continue
            building = _building_at(buildings, tuple(payload["movement"]))
            if building and building.get("safe_at_night", False):
                occupancy[_building_name_for(buildings, tuple(payload["movement"]))] += 1
        for name, count in occupancy.items():
            capacity = max(1, int(safe_buildings[name]["capacity"]))
            overflow = max(0, count - capacity)
            crowding_values.append(round(overflow / capacity, 3))
            max_overflow = max(max_overflow, float(overflow))
    return {
        "average_shelter_crowding": round(mean(crowding_values), 3) if crowding_values else 0.0,
        "max_shelter_overflow": round(max_overflow, 3),
    }


def _survival_curve(events: List[Dict[str, object]], agent_count: int, steps: int) -> Dict[str, int]:
    survivors = agent_count
    deaths_by_step: Dict[int, int] = {}
    for event in events:
        if event.get("event_type") == "death":
            step = int(event.get("step", 0))
            deaths_by_step[step] = deaths_by_step.get(step, 0) + 1
    curve: Dict[str, int] = {}
    for step in range(steps):
        survivors -= deaths_by_step.get(step, 0)
        curve[str(step)] = max(0, survivors)
    return curve


def _per_role_outcomes(run_dir: Path) -> Dict[str, Dict[str, float]]:
    personas_root = run_dir / "personas"
    role_rows: Dict[str, List[Tuple[float, float, float]]] = {}
    if not personas_root.exists():
        return {}
    for state_path in personas_root.glob("*/state.json"):
        state = load_json(state_path)
        role_rows.setdefault(str(state["role"]), []).append(
            (
                1.0 if state["alive"] else 0.0,
                float(state["stress"]),
                float(state["health"]),
            )
        )
    output: Dict[str, Dict[str, float]] = {}
    for role, values in role_rows.items():
        output[role] = {
            "count": float(len(values)),
            "survival_rate": round(mean(value[0] for value in values), 3),
            "average_stress": round(mean(value[1] for value in values), 3),
            "average_health": round(mean(value[2] for value in values), 3),
        }
    return output


def _mean_relationship_shift(social_graph: Dict[str, object]) -> float:
    if not social_graph:
        return 0.0
    keys = sorted(social_graph.keys(), key=int)
    if len(keys) < 2:
        return 0.0
    first_edges = _edge_map(social_graph[keys[0]])
    last_edges = _edge_map(social_graph[keys[-1]])
    shifts = []
    for edge_key, first_weight in first_edges.items():
        if edge_key in last_edges:
            shifts.append(abs(last_edges[edge_key] - first_weight))
    return round(mean(shifts), 3) if shifts else 0.0


def _edge_map(graph_step: Dict[str, object]) -> Dict[Tuple[str, str], float]:
    edges = {}
    for edge in graph_step.get("edges", []):
        edges[(str(edge["source"]), str(edge["target"]))] = float(edge["weight"])
    return edges


def _event_reactivity(events: List[Dict[str, object]], social_graph: Dict[str, object]) -> Dict[str, object]:
    if not social_graph:
        return {"mean_trust_event_delta": 0.0, "mean_suspicion_event_delta": 0.0, "by_event_type": {}}
    keys = sorted(social_graph.keys(), key=int)
    by_step = {int(key): social_graph[key] for key in keys}
    trust_deltas: List[float] = []
    suspicion_deltas: List[float] = []
    by_event_type: Dict[str, Dict[str, List[float]]] = {}
    for event in events:
        event_type = str(event.get("event_type", ""))
        step = int(event.get("step", 0))
        next_step = step + 1
        if step not in by_step or next_step not in by_step:
            continue
        target = event.get("target") or event.get("actor")
        if not target:
            continue
        current_node = by_step[step].get("nodes", {}).get(target)
        next_node = by_step[next_step].get("nodes", {}).get(target)
        if not current_node or not next_node:
            continue
        trust_delta = float(next_node.get("trust_state", 0.0)) - float(current_node.get("trust_state", 0.0))
        suspicion_delta = float(next_node.get("suspicion", 0.0)) - float(current_node.get("suspicion", 0.0))
        trust_deltas.append(trust_delta)
        suspicion_deltas.append(suspicion_delta)
        by_event_type.setdefault(event_type, {"trust": [], "suspicion": []})
        by_event_type[event_type]["trust"].append(trust_delta)
        by_event_type[event_type]["suspicion"].append(suspicion_delta)
    return {
        "mean_trust_event_delta": round(mean(trust_deltas), 3) if trust_deltas else 0.0,
        "mean_suspicion_event_delta": round(mean(suspicion_deltas), 3) if suspicion_deltas else 0.0,
        "by_event_type": {
            event_type: {
                "mean_trust_delta": round(mean(values["trust"]), 3) if values["trust"] else 0.0,
                "mean_suspicion_delta": round(mean(values["suspicion"]), 3) if values["suspicion"] else 0.0,
            }
            for event_type, values in by_event_type.items()
        },
    }


def _parse_time_of_day(curr_time: str) -> str:
    if "/" not in curr_time:
        return "day"
    return curr_time.split("/")[-1].strip()


def _building_at(buildings: Dict[str, Dict[str, object]], location: Tuple[int, int]) -> Dict[str, object] | None:
    best_name = _building_name_for(buildings, location)
    return buildings.get(best_name) if best_name else None


def _building_name_for(buildings: Dict[str, Dict[str, object]], location: Tuple[int, int]) -> str | None:
    best_match = None
    best_distance = None
    for name, building in buildings.items():
        radius = 1 if building["category"] in {"house", "forest_edge"} else 2
        distance = abs(int(building["location"][0]) - location[0]) + abs(int(building["location"][1]) - location[1])
        if distance <= radius and (best_distance is None or distance < best_distance):
            best_match = name
            best_distance = distance
    return best_match
