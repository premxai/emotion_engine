from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghost_town_training.training import train_condition_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first imitation-based condition_b checkpoint.")
    parser.add_argument("--input-dir", action="append", required=True, help="Root directory containing teacher run outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory to write checkpoints and metrics.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["standard_night", "storm_scarcity", "betrayal_refusal"],
        help="Scenario names to include from the teacher dataset.",
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(0, 8)))
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[8])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[9])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_condition_b(
        input_roots=[Path(path) for path in args.input_dir],
        output_dir=Path(args.output_dir),
        scenarios=args.scenarios,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

