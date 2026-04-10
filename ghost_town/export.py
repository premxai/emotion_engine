from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from .types import AgentState, EventLog, RunManifest, TrainingStepRecord, WorldState


def export_simulation(
    output_dir: Path,
    world: WorldState,
    timeline: Dict[int, Dict[str, object]],
    metrics: Dict[str, object],
    affect_timeline: Optional[Dict[int, Dict[str, object]]] = None,
    social_graph_timeline: Optional[Dict[int, Dict[str, object]]] = None,
    training_records: Optional[Iterable[TrainingStepRecord]] = None,
    run_manifest: Optional[RunManifest] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "master_movement.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    meta = {
        "sim_code": output_dir.name,
        "maze_name": "ghost_town",
        "sec_per_step": 300,
        "day_index": world.day_index,
        "weather": world.weather,
        "condition": metrics["condition"],
        "event_count": len(world.events),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "world_metadata.json").write_text(json.dumps(serialize_world(world), indent=2), encoding="utf-8")
    (output_dir / "events.json").write_text(json.dumps([serialize_event(event) for event in world.events], indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if affect_timeline is not None:
        (output_dir / "affect_timeline.json").write_text(json.dumps(affect_timeline, indent=2), encoding="utf-8")
    if social_graph_timeline is not None:
        (output_dir / "social_graph_timeline.json").write_text(json.dumps(social_graph_timeline, indent=2), encoding="utf-8")
    with (output_dir / "training_records.jsonl").open("w", encoding="utf-8") as handle:
        if training_records is not None:
            for record in training_records:
                handle.write(json.dumps(serialize_training_record(record)) + "\n")
    manifest = run_manifest or build_run_manifest(output_dir, metrics)
    (output_dir / "run_manifest.json").write_text(json.dumps(serialize_manifest(manifest), indent=2), encoding="utf-8")
    personas_dir = output_dir / "personas"
    personas_dir.mkdir(exist_ok=True)
    for agent in world.agents.values():
        agent_dir = personas_dir / agent.name
        agent_dir.mkdir(exist_ok=True)
        (agent_dir / "state.json").write_text(json.dumps(serialize_agent(agent), indent=2), encoding="utf-8")


def serialize_event(event: EventLog) -> Dict[str, object]:
    return {
        "step": event.step,
        "time_of_day": event.time_of_day,
        "weather": event.weather,
        "event_type": event.event_type,
        "description": event.description,
        "actor": event.actor,
        "target": event.target,
        "location": event.location,
        "metadata": event.metadata,
    }


def serialize_training_record(record: TrainingStepRecord) -> Dict[str, object]:
    return {
        "run_id": record.run_id,
        "step": record.step,
        "agent_id": record.agent_id,
        "condition": record.condition,
        "seed": record.seed,
        "observation": record.observation,
        "social_context": record.social_context,
        "prev_latent": record.prev_latent,
        "latent": record.latent,
        "action": record.action,
        "goal": record.goal,
        "reward_components": record.reward_components,
        "total_reward": record.total_reward,
        "done": record.done,
        "metadata": record.metadata,
    }


def serialize_agent(agent: AgentState) -> Dict[str, object]:
    return {
        "name": agent.name,
        "role": agent.role,
        "home": agent.home,
        "workplace": agent.workplace,
        "personality": agent.personality,
        "skills": agent.skills,
        "traits": agent.traits,
        "social_ties": agent.social_ties,
        "relationship_labels": agent.relationship_labels,
        "day_goals": agent.day_goals,
        "night_goals": agent.night_goals,
        "survival_style": agent.survival_style,
        "alive": agent.alive,
        "location": list(agent.location),
        "health": round(agent.health, 2),
        "stamina": round(agent.stamina, 2),
        "stress": round(agent.stress, 2),
        "fear": round(agent.fear, 2),
        "trust": round(agent.trust, 2),
        "grief": round(agent.grief, 2),
        "suspicion": round(agent.suspicion, 2),
        "inventory": agent.inventory,
        "sheltered": agent.sheltered,
        "current_goal": agent.current_goal,
        "current_action": agent.current_action,
        "destination_name": agent.destination_name,
        "affect_embedding": agent.affect_embedding,
        "prompt_context": agent.prompt_context,
        "probe_data": agent.probe_data,
        "known_deaths": agent.memory.known_deaths,
        "trust_memory": agent.memory.trust_memory,
        "relationship_events": agent.memory.relationship_events,
    }


def serialize_world(world: WorldState) -> Dict[str, object]:
    return {
        "width": world.width,
        "height": world.height,
        "day_index": world.day_index,
        "buildings": {
            name: {
                "category": building.category,
                "location": list(building.location),
                "district": building.district,
                "map_label": building.map_label,
                "shelter_quality": building.shelter_quality,
                "capacity": building.capacity,
                "safe_at_night": building.safe_at_night,
                "tags": building.tags,
                "supply_spawn": building.supply_spawn,
                "treatment_zone": building.treatment_zone,
                "ghost_spawn": building.ghost_spawn,
            }
            for name, building in world.buildings.items()
        },
    }


def serialize_manifest(manifest: RunManifest) -> Dict[str, object]:
    return {
        "run_id": manifest.run_id,
        "schema_version": manifest.schema_version,
        "timestamp_utc": manifest.timestamp_utc,
        "git_commit": manifest.git_commit,
        "runtime_backend": manifest.runtime_backend,
        "tmx_path": manifest.tmx_path,
        "semantics_manifest": manifest.semantics_manifest,
        "transfer_source_manifest": manifest.transfer_source_manifest,
        "policy_source": manifest.policy_source,
        "condition_b_checkpoint": manifest.condition_b_checkpoint,
        "condition_b_schema": manifest.condition_b_schema,
        "condition_b_policy_mode": manifest.condition_b_policy_mode,
        "condition_b_temperature": manifest.condition_b_temperature,
        "config": manifest.config,
        "metrics": manifest.metrics,
        "outputs": manifest.outputs,
    }


def build_run_manifest(output_dir: Path, metrics: Dict[str, object]) -> RunManifest:
    return RunManifest(
        run_id=output_dir.name,
        schema_version="ghost_town.v2",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(output_dir),
        config={
            "condition": metrics.get("condition", ""),
            "scenario": metrics.get("scenario", "standard_night"),
            "seed": metrics.get("seed", ""),
            "steps": metrics.get("steps", ""),
        },
        metrics=metrics,
        outputs={
            "movement": "master_movement.json",
            "events": "events.json",
            "metrics": "metrics.json",
            "world_metadata": "world_metadata.json",
            "training_records": "training_records.jsonl",
            "manifest": "run_manifest.json",
        },
        policy_source="",
        condition_b_checkpoint="",
        condition_b_schema="",
        condition_b_policy_mode="",
        condition_b_temperature=1.0,
        runtime_backend="native",
        tmx_path="",
        semantics_manifest="",
        transfer_source_manifest="",
    )


def get_git_commit(path_hint: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path_hint.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
