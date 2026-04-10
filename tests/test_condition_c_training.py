import json
import shutil
import unittest
from pathlib import Path

from ghost_town.simulator import GhostTownSimulator
from ghost_town_training.training_c import train_condition_c
from tests.test_condition_b_training import _write_teacher_run


class ConditionCTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("outputs") / "test_condition_c_training"
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

    def test_training_writes_checkpoint_and_eval(self):
        output_dir = self.root / "trained_condition_c"
        result = train_condition_c(
            input_roots=[self.teacher_root],
            output_dir=output_dir,
            scenarios=["standard_night", "storm_scarcity", "betrayal_refusal"],
            train_seeds=list(range(0, 8)),
            val_seeds=[8],
            test_seeds=[9],
            latent_pretrain_epochs=3,
            action_head_epochs=2,
            joint_epochs=1,
            batch_size=16,
            learning_rate=1e-3,
        )
        self.assertTrue((output_dir / "best.pt").exists())
        self.assertTrue((output_dir / "last.pt").exists())
        self.assertTrue((output_dir / "vocab_and_schema.json").exists())
        self.assertTrue((output_dir / "eval_metrics.json").exists())
        eval_metrics = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
        self.assertGreater(eval_metrics["test"]["action_accuracy"], 0.45)
        self.assertLess(eval_metrics["test"]["latent_loss"], 0.5)
        self.assertEqual(result["training_profile"], "self_supervised_latent_v1")

    def test_condition_c_runtime_loads_trained_checkpoint(self):
        output_dir = self.root / "runtime_model"
        train_condition_c(
            input_roots=[self.teacher_root],
            output_dir=output_dir,
            scenarios=["standard_night", "storm_scarcity", "betrayal_refusal"],
            train_seeds=list(range(0, 8)),
            val_seeds=[8],
            test_seeds=[9],
            latent_pretrain_epochs=2,
            action_head_epochs=2,
            joint_epochs=1,
            batch_size=16,
            learning_rate=1e-3,
        )
        simulator = GhostTownSimulator(
            condition="condition_c",
            scenario="standard_night",
            agent_count=12,
            steps=2,
            seed=7,
            condition_c_checkpoint=str(output_dir / "best.pt"),
            condition_c_schema=str(output_dir / "vocab_and_schema.json"),
        )
        result = simulator.run()
        self.assertEqual(result.config.condition_c_policy_source, "trained_checkpoint")
        first_step = result.affect_timeline[0]
        first_agent = next(iter(first_step.values()))
        self.assertEqual(first_agent["probe_data"]["policy_source"], "trained_condition_c_checkpoint")


if __name__ == "__main__":
    unittest.main()
