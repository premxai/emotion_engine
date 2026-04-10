from __future__ import annotations

import argparse
from pathlib import Path

from ghost_town.scenario import list_scenarios
from ghost_town.simulator import GhostTownSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ghost-stress town simulator.")
    parser.add_argument("--condition", default="condition_a", choices=["baseline_0", "condition_a", "condition_b", "condition_c", "condition_d"])
    parser.add_argument("--agents", type=int, default=12, help="Number of agents to simulate.")
    parser.add_argument("--steps", type=int, default=72, help="Number of five-minute steps to simulate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic runs.")
    parser.add_argument("--scenario", default="standard_night", choices=list_scenarios(), help="Scenario preset to run.")
    parser.add_argument("--storm-mode", default=None, choices=["stochastic", "none", "scheduled"])
    parser.add_argument("--ghost-mode", default=None, choices=["standard", "none", "high_pressure"])
    parser.add_argument("--curriculum-stage", default="", help="Optional label for curriculum stage.")
    parser.add_argument("--condition-b-checkpoint", default="", help="Optional trained condition_b checkpoint.")
    parser.add_argument("--condition-b-schema", default="", help="Optional trained condition_b schema bundle.")
    parser.add_argument("--condition-b-policy-mode", choices=["deterministic", "sample"], default="deterministic")
    parser.add_argument("--condition-b-temperature", type=float, default=1.0)
    parser.add_argument("--condition-c-checkpoint", default="", help="Optional trained condition_c checkpoint.")
    parser.add_argument("--condition-c-schema", default="", help="Optional trained condition_c schema bundle.")
    parser.add_argument("--condition-d-checkpoint", default="", help="Optional trained condition_d checkpoint.")
    parser.add_argument("--condition-d-schema", default="", help="Optional trained condition_d schema bundle.")
    parser.add_argument(
        "--output",
        default="outputs/ghost_town_run",
        help="Output directory for movement logs, events, metrics, and per-agent state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    simulator = GhostTownSimulator(
        condition=args.condition,
        agent_count=args.agents,
        steps=args.steps,
        seed=args.seed,
        scenario=args.scenario,
        storm_mode=args.storm_mode,
        ghost_mode=args.ghost_mode,
        run_id=output_path.name,
        curriculum_stage=args.curriculum_stage,
        condition_b_checkpoint=args.condition_b_checkpoint,
        condition_b_schema=args.condition_b_schema,
        condition_b_policy_mode=args.condition_b_policy_mode,
        condition_b_temperature=args.condition_b_temperature,
        condition_c_checkpoint=args.condition_c_checkpoint,
        condition_c_schema=args.condition_c_schema,
        condition_d_checkpoint=args.condition_d_checkpoint,
        condition_d_schema=args.condition_d_schema,
    )
    result = simulator.export(output_path)
    print(
        f"completed {args.condition}: survivors={result.metrics['survivors']} "
        f"deaths={result.metrics['deaths']} avg_stress={result.metrics['average_stress']}"
    )


if __name__ == "__main__":
    main()
