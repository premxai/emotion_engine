#!/usr/bin/env python3
"""Train condition_d — predictive emotion engine.

Collects training data from all conditions across training scenarios,
builds 12 future-event prediction labels, and trains ConditionDPredictiveModel
via three-stage curriculum (prediction_pretrain → action_head → joint).

Usage:
    python train_condition_d.py --input-dirs outputs/condition_d_training_data --output outputs/condition_d_v1
    python train_condition_d.py --input-dirs outputs/ --output outputs/condition_d_smoke --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train condition_d predictive emotion engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more directories containing training_records.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/condition_d_smoke"),
        help="Output directory for checkpoints, metrics, and schema.",
    )
    parser.add_argument(
        "--training-scenarios",
        nargs="+",
        default=["standard_night", "storm_scarcity", "betrayal_refusal"],
        help="Scenarios to include in training (ally_death and high_ghost_pressure are held out).",
    )
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Seeds for training split.",
    )
    parser.add_argument(
        "--val-seeds",
        nargs="+",
        type=int,
        default=[6],
        help="Seeds for validation split.",
    )
    parser.add_argument(
        "--test-seeds",
        nargs="+",
        type=int,
        default=[7],
        help="Seeds for test split.",
    )
    parser.add_argument("--prediction-pretrain-epochs", type=int, default=12)
    parser.add_argument("--action-head-epochs", type=int, default=6)
    parser.add_argument("--joint-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--joint-learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[192, 128],
        help="Hidden layer sizes for encoder backbone.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset and print label distribution, then exit without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate input directories
    for d in args.input_dirs:
        if not d.exists():
            print(f"ERROR: Input directory does not exist: {d}", file=sys.stderr)
            sys.exit(1)

    from ghost_town_training.training_d import train_condition_d

    result = train_condition_d(
        input_roots=args.input_dirs,
        output_dir=args.output,
        training_scenarios=args.training_scenarios,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        prediction_pretrain_epochs=args.prediction_pretrain_epochs,
        action_head_epochs=args.action_head_epochs,
        joint_epochs=args.joint_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        joint_learning_rate=args.joint_learning_rate,
        hidden_dims=tuple(args.hidden_dims),
        dry_run=args.dry_run,
    )

    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
