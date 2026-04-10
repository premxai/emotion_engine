import json
import shutil
import unittest
from pathlib import Path

from ghost_town.simulator import GhostTownSimulator
from ghost_town_training.dataset import FeatureSchema, load_teacher_records, split_records_by_seed
from ghost_town_training.training import compute_training_weight, summarize_training_weights, train_condition_b


def _write_teacher_run(root: Path, scenario: str, seed: int, action: str) -> None:
    run_dir = root / f"condition_a_{scenario}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(4):
        rows.append(
            {
                "run_id": run_dir.name,
                "step": step,
                "agent_id": f"agent_{seed}_{step}",
                "condition": "condition_a",
                "seed": seed,
                "observation": {
                    "visible_ghosts": 2 if scenario == "storm_scarcity" else 0,
                    "visible_deaths": 1 if scenario == "betrayal_refusal" else 0,
                    "nearby_allies": 1,
                    "trusted_allies": 1 if scenario == "standard_night" else 0,
                    "supplies_seen": 1 if scenario == "storm_scarcity" else 3,
                    "in_shelter": scenario == "standard_night",
                    "at_home": scenario == "standard_night",
                    "at_work": scenario == "storm_scarcity",
                    "nearest_refuge_distance": 1 if scenario == "standard_night" else 4,
                },
                "social_context": {
                    "average_trust": 0.8 if scenario == "standard_night" else 0.2,
                    "graph_support": 0.7 if scenario == "standard_night" else 0.1,
                    "graph_tension": 0.8 if scenario == "betrayal_refusal" else 0.2,
                },
                "prev_latent": [0.1 * step] * 8,
                "latent": (
                    [0.1, 0.15, 0.8, 0.0, 0.05, 0.6, 0.1, 0.3]
                    if scenario == "standard_night"
                    else [0.6, 0.7, 0.2, 0.1, 0.2, 0.1, 0.8, 0.2]
                    if scenario == "storm_scarcity"
                    else [0.2, 0.4, 0.1, 0.5, 0.8, 0.0, 0.4, 0.9]
                ),
                "action": action,
                "goal": action,
                "reward_components": {"survival": 1.0, "social": 0.2},
                "total_reward": 1.2,
                "done": step == 3,
                "metadata": {
                    "health": 80.0 if scenario != "storm_scarcity" else 55.0,
                    "storm": scenario == "storm_scarcity",
                    "alive": True,
                    "sheltered": scenario == "standard_night",
                    "rescue_opportunity": False,
                    "refusal_opportunity": scenario == "betrayal_refusal",
                    "rival_refusal_opportunity": scenario == "betrayal_refusal",
                    "scenario": scenario,
                    "role": "steward" if scenario == "standard_night" else "runner",
                    "refusal_context": "betrayal_sequence" if scenario == "betrayal_refusal" else "scarcity",
                    "relationship_label_at_decision": "rivals" if scenario == "betrayal_refusal" else "household",
                    "tie_value_at_decision": -0.5 if scenario == "betrayal_refusal" else 0.8,
                    "time_of_day": "night" if scenario == "storm_scarcity" else "day",
                },
            }
        )
    (run_dir / "training_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )


class ConditionBVerticalSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("outputs") / "test_condition_b_vertical_slice"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.teacher_root = self.root / "teacher"
        for seed in range(0, 10):
            _write_teacher_run(self.teacher_root, "standard_night", seed, "seek_safe_house")
            _write_teacher_run(self.teacher_root, "storm_scarcity", seed, "gather_supplies")
            _write_teacher_run(self.teacher_root, "betrayal_refusal", seed, "refuse_help")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_teacher_records_split_cleanly(self):
        records = load_teacher_records([self.teacher_root], ["standard_night", "storm_scarcity", "betrayal_refusal"])
        splits = split_records_by_seed(records, train_seeds=range(0, 8), val_seeds=[8], test_seeds=[9])
        self.assertEqual(sorted({row["seed"] for row in splits["train"]}), list(range(0, 8)))
        self.assertEqual(sorted({row["seed"] for row in splits["val"]}), [8])
        self.assertEqual(sorted({row["seed"] for row in splits["test"]}), [9])

    def test_training_writes_schema_checkpoint_and_eval(self):
        output_dir = self.root / "trained_condition_b"
        result = train_condition_b(
            input_roots=[self.teacher_root],
            output_dir=output_dir,
            scenarios=["standard_night", "storm_scarcity", "betrayal_refusal"],
            train_seeds=list(range(0, 8)),
            val_seeds=[8],
            test_seeds=[9],
            epochs=6,
            batch_size=16,
            learning_rate=1e-3,
        )
        self.assertTrue((output_dir / "best.pt").exists())
        self.assertTrue((output_dir / "last.pt").exists())
        self.assertTrue((output_dir / "vocab_and_schema.json").exists())
        self.assertTrue((output_dir / "eval_metrics.json").exists())
        eval_metrics = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
        self.assertGreater(eval_metrics["test"]["action_accuracy"], 0.5)
        self.assertEqual(eval_metrics["weight_profile"], "realism_v2")
        schema = FeatureSchema.load(output_dir / "vocab_and_schema.json")
        self.assertIn("refuse_help", schema.action_vocab)
        self.assertEqual(schema.latent_dim, 8)
        self.assertEqual(schema.version, "condition_b.v2")
        self.assertIn("tie_value_at_decision", schema.metadata_scalar_keys)
        self.assertIn("refusal_context", schema.categorical_keys)
        self.assertIn("time_of_day", schema.categorical_keys)
        self.assertEqual(result["best_checkpoint"], str(output_dir / "best.pt"))
        self.assertEqual(result["weight_profile"], "realism_v2")
        train_metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(train_metrics["weight_profile"], "realism_v2")
        self.assertIn("standard_night:seek_safe_house", train_metrics["train_weight_summary"]["per_bucket"])

    def test_weighting_downweights_calm_refusals_and_preserves_rival_refusals(self):
        records = load_teacher_records([self.teacher_root], ["standard_night", "storm_scarcity", "betrayal_refusal"])
        standard_refusal = next(
            {
                **record,
                "action": "refuse_help",
                "metadata": {**record["metadata"], "refusal_opportunity": True, "rival_refusal_opportunity": False},
            }
            for record in records
            if record["metadata"]["scenario"] == "standard_night"
        )
        betrayal_refusal = next(
            record for record in records if record["metadata"]["scenario"] == "betrayal_refusal"
        )
        calm_weight = compute_training_weight(standard_refusal)
        rivalry_weight = compute_training_weight(betrayal_refusal)
        self.assertLess(calm_weight, 1.0)
        self.assertGreater(rivalry_weight, 1.0)
        self.assertLess(calm_weight, rivalry_weight)

    def test_weight_summary_reports_weight_range(self):
        records = load_teacher_records([self.teacher_root], ["standard_night", "storm_scarcity", "betrayal_refusal"])
        summary = summarize_training_weights(records)
        self.assertEqual(summary["n_examples"], len(records))
        self.assertLess(summary["min_weight"], summary["max_weight"])

    def test_condition_b_runtime_loads_trained_checkpoint(self):
        output_dir = self.root / "runtime_model"
        train_condition_b(
            input_roots=[self.teacher_root],
            output_dir=output_dir,
            scenarios=["standard_night", "storm_scarcity", "betrayal_refusal"],
            train_seeds=list(range(0, 8)),
            val_seeds=[8],
            test_seeds=[9],
            epochs=4,
            batch_size=16,
            learning_rate=1e-3,
        )
        simulator = GhostTownSimulator(
            condition="condition_b",
            scenario="standard_night",
            agent_count=12,
            steps=2,
            seed=7,
            condition_b_checkpoint=str(output_dir / "best.pt"),
            condition_b_schema=str(output_dir / "vocab_and_schema.json"),
        )
        result = simulator.run()
        self.assertEqual(result.config.condition_b_policy_source, "trained_checkpoint")
        first_step = result.affect_timeline[0]
        first_agent = next(iter(first_step.values()))
        self.assertEqual(first_agent["probe_data"]["policy_source"], "trained_checkpoint")
        manifest = simulator._build_run_manifest(self.root / "dummy_run", result.metrics)
        self.assertEqual(manifest.policy_source, "trained_checkpoint")


if __name__ == "__main__":
    unittest.main()
