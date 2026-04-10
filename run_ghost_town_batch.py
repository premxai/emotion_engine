from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

from ghost_town.scenario import list_scenarios, resolve_mode_for_scenario
from ghost_town.simulator import GhostTownSimulator
from ghost_town.types import BatchExperimentResult


CONDITIONS = ["baseline_0", "condition_a", "condition_b", "condition_c", "condition_d"]
CURRICULUM_PRESETS = {
    "8-agent": {"agent_count": 8, "steps": 24},
    "12-agent": {"agent_count": 12, "steps": 24},
    "25-agent": {"agent_count": 25, "steps": 24},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched batch experiments for ghost town.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7], help="List of random seeds.")
    parser.add_argument("--curriculum", choices=sorted(CURRICULUM_PRESETS), default="12-agent")
    parser.add_argument("--agents", type=int, default=None, help="Override preset agent count.")
    parser.add_argument("--steps", type=int, default=None, help="Override preset step count.")
    parser.add_argument("--scenario", choices=list_scenarios(), default="standard_night")
    parser.add_argument("--storm-mode", choices=["stochastic", "none", "scheduled"], default=None)
    parser.add_argument("--ghost-mode", choices=["standard", "none", "high_pressure"], default=None)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=None)
    parser.add_argument("--condition-b-checkpoint", default="", help="Optional trained condition_b checkpoint.")
    parser.add_argument("--condition-b-schema", default="", help="Optional trained condition_b schema bundle.")
    parser.add_argument("--condition-b-policy-mode", choices=["deterministic", "sample"], default="deterministic")
    parser.add_argument("--condition-b-temperature", type=float, default=1.0)
    parser.add_argument("--condition-c-checkpoint", default="", help="Optional trained condition_c checkpoint.")
    parser.add_argument("--condition-c-schema", default="", help="Optional trained condition_c schema bundle.")
    parser.add_argument("--condition-d-checkpoint", default="", help="Optional trained condition_d checkpoint.")
    parser.add_argument("--condition-d-schema", default="", help="Optional trained condition_d schema bundle.")
    parser.add_argument("--output", default="outputs/batch_experiment", help="Batch output directory.")
    return parser.parse_args()


def run_batch(
    output_dir: Path,
    seeds: List[int],
    curriculum: str = "12-agent",
    agent_count: int | None = None,
    steps: int | None = None,
    scenario: str = "standard_night",
    storm_mode: str | None = None,
    ghost_mode: str | None = None,
    conditions: List[str] | None = None,
    condition_b_checkpoint: str = "",
    condition_b_schema: str = "",
    condition_b_policy_mode: str = "deterministic",
    condition_b_temperature: float = 1.0,
    condition_c_checkpoint: str = "",
    condition_c_schema: str = "",
    condition_d_checkpoint: str = "",
    condition_d_schema: str = "",
    runtime_backend: str = "native",
    tmx_path: str = "",
    semantics_manifest: str = "",
    transfer_source_manifest: str = "",
    world_width_override: int | None = None,
    world_height_override: int | None = None,
    building_location_overrides: Dict[str, tuple[int, int]] | None = None,
) -> BatchExperimentResult:
    preset = CURRICULUM_PRESETS[curriculum]
    final_agent_count = agent_count or preset["agent_count"]
    final_steps = steps or preset["steps"]
    resolved_modes = resolve_mode_for_scenario(scenario, storm_mode, ghost_mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_conditions = conditions or CONDITIONS

    rows: List[Dict[str, object]] = []
    run_paths: Dict[str, List[str]] = {condition: [] for condition in selected_conditions}

    for seed in seeds:
        for condition in selected_conditions:
            run_name = f"{condition}_{scenario}_seed{seed}_{curriculum}"
            run_dir = output_dir / run_name
            simulator = GhostTownSimulator(
                condition=condition,
                agent_count=final_agent_count,
                steps=final_steps,
                seed=seed,
                scenario=scenario,
                storm_mode=resolved_modes["storm_mode"],
                ghost_mode=resolved_modes["ghost_mode"],
                run_id=run_name,
                curriculum_stage=curriculum,
                condition_b_checkpoint=condition_b_checkpoint if condition == "condition_b" else "",
                condition_b_schema=condition_b_schema if condition == "condition_b" else "",
                condition_b_policy_mode=condition_b_policy_mode if condition == "condition_b" else "deterministic",
                condition_b_temperature=condition_b_temperature if condition == "condition_b" else 1.0,
                condition_c_checkpoint=condition_c_checkpoint if condition == "condition_c" else "",
                condition_c_schema=condition_c_schema if condition == "condition_c" else "",
                condition_d_checkpoint=condition_d_checkpoint if condition == "condition_d" else "",
                condition_d_schema=condition_d_schema if condition == "condition_d" else "",
                runtime_backend=runtime_backend,
                tmx_path=tmx_path,
                semantics_manifest=semantics_manifest,
                transfer_source_manifest=transfer_source_manifest,
                world_width_override=world_width_override,
                world_height_override=world_height_override,
                building_location_overrides=building_location_overrides,
            )
            result = simulator.export(run_dir)
            run_paths[condition].append(str(run_dir))
            rows.append(result.metrics)

    aggregate_json = output_dir / "aggregate_metrics.json"
    aggregate_csv = output_dir / "aggregate_metrics.csv"
    summary_json = output_dir / "condition_summary.json"
    manifest_json = output_dir / "batch_manifest.json"

    aggregate_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(rows, aggregate_csv)
    condition_summary = summarize_by_condition(rows)
    summary_json.write_text(json.dumps(condition_summary, indent=2), encoding="utf-8")

    batch = BatchExperimentResult(
        batch_id=output_dir.name,
        runs=run_paths,
        aggregate_summary_path=str(summary_json),
        aggregate_metrics_path=str(aggregate_json),
    )
    manifest_json.write_text(
        json.dumps(
            {
                "batch_id": batch.batch_id,
                "curriculum": curriculum,
                "scenario": scenario,
                "agent_count": final_agent_count,
                "steps": final_steps,
                "seeds": seeds,
                "conditions": selected_conditions,
                "storm_mode": resolved_modes["storm_mode"],
                "ghost_mode": resolved_modes["ghost_mode"],
                "condition_b_policy_source": (
                    "trained_checkpoint_sampled_rl"
                    if condition_b_checkpoint and condition_b_schema and condition_b_policy_mode == "sample"
                    else "trained_checkpoint"
                    if condition_b_checkpoint and condition_b_schema
                    else "scaffold_fallback"
                ),
                "condition_b_checkpoint": condition_b_checkpoint,
                "condition_b_schema": condition_b_schema,
                "condition_b_policy_mode": condition_b_policy_mode,
                "condition_b_temperature": condition_b_temperature,
                "condition_c_policy_source": "trained_checkpoint" if condition_c_checkpoint and condition_c_schema else "hybrid_fallback",
                "condition_c_checkpoint": condition_c_checkpoint,
                "condition_c_schema": condition_c_schema,
                "condition_d_policy_source": "trained_checkpoint" if condition_d_checkpoint and condition_d_schema else "scaffold_fallback",
                "condition_d_checkpoint": condition_d_checkpoint,
                "condition_d_schema": condition_d_schema,
                "runtime_backend": runtime_backend,
                "tmx_path": tmx_path,
                "semantics_manifest": semantics_manifest,
                "transfer_source_manifest": transfer_source_manifest,
                "runs": batch.runs,
                "aggregate_summary_path": batch.aggregate_summary_path,
                "aggregate_metrics_path": batch.aggregate_metrics_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return batch


def summarize_by_condition(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    numeric_keys = [
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
    ]
    for condition in sorted({str(row["condition"]) for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            key: round(mean(float(row[key]) for row in condition_rows), 3) for key in numeric_keys
        }
    return summary


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    result = run_batch(
        output_dir=Path(args.output),
        seeds=args.seeds,
        curriculum=args.curriculum,
        agent_count=args.agents,
        steps=args.steps,
        scenario=args.scenario,
        storm_mode=args.storm_mode,
        ghost_mode=args.ghost_mode,
        conditions=args.conditions,
        condition_b_checkpoint=args.condition_b_checkpoint,
        condition_b_schema=args.condition_b_schema,
        condition_b_policy_mode=args.condition_b_policy_mode,
        condition_b_temperature=args.condition_b_temperature,
        condition_c_checkpoint=args.condition_c_checkpoint,
        condition_c_schema=args.condition_c_schema,
        condition_d_checkpoint=args.condition_d_checkpoint,
        condition_d_schema=args.condition_d_schema,
    )
    print(f"completed batch {result.batch_id}: summary={result.aggregate_summary_path}")


if __name__ == "__main__":
    main()
