from __future__ import annotations

import argparse
import json
from pathlib import Path

from ghost_town_training.training_c import train_condition_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train condition_c using self-supervised latent dynamics plus an action head.")
    parser.add_argument("--input-dir", action="append", required=True, help="Teacher data roots containing training_records.jsonl files.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--scenarios", nargs="+", default=["standard_night", "storm_scarcity", "betrayal_refusal"])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(0, 8)))
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[8])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[9])
    parser.add_argument("--latent-pretrain-epochs", type=int, default=8)
    parser.add_argument("--action-head-epochs", type=int, default=6)
    parser.add_argument("--joint-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_condition_c(
        input_roots=[Path(path) for path in args.input_dir],
        output_dir=Path(args.output_dir),
        scenarios=args.scenarios,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        test_seeds=args.test_seeds,
        latent_pretrain_epochs=args.latent_pretrain_epochs,
        action_head_epochs=args.action_head_epochs,
        joint_epochs=args.joint_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
