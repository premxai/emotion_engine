import json
import shutil
import unittest
from pathlib import Path

from ghost_town.comparison import build_condition_ab_comparison, render_condition_ab_markdown, write_condition_ab_comparison
from ghost_town.diagnosis import diagnose_condition_b_batches
from tests.test_condition_b_diagnosis import _make_run


class ConditionABComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("outputs") / "test_condition_ab_comparison"
        if self.root.exists():
            shutil.rmtree(self.root)
        scenarios = {
            "standard_night": {
                "condition_a": {"survivors": 1, "deaths": 0, "average_stress": 4.0, "average_fear": 1.0, "average_suspicion": 40.0, "relationship_events": 10, "refusals": 2, "ghost_encounters": 0},
                "condition_b": {"survivors": 1, "deaths": 0, "average_stress": 6.0, "average_fear": 2.0, "average_suspicion": 42.0, "relationship_events": 8, "refusals": 1, "ghost_encounters": 0},
            },
            "storm_scarcity": {
                "condition_a": {"survivors": 1, "deaths": 0, "average_stress": 8.0, "average_fear": 4.0, "average_suspicion": 55.0, "relationship_events": 14, "refusals": 4, "ghost_encounters": 1},
                "condition_b": {"survivors": 1, "deaths": 0, "average_stress": 7.5, "average_fear": 4.1, "average_suspicion": 50.0, "relationship_events": 12, "refusals": 3, "ghost_encounters": 1},
            },
            "betrayal_refusal": {
                "condition_a": {"survivors": 1, "deaths": 0, "average_stress": 5.0, "average_fear": 1.5, "average_suspicion": 70.0, "relationship_events": 20, "refusals": 3, "ghost_encounters": 0},
                "condition_b": {"survivors": 1, "deaths": 0, "average_stress": 4.0, "average_fear": 1.6, "average_suspicion": 65.0, "relationship_events": 18, "refusals": 3, "ghost_encounters": 0},
            },
        }
        self.batch_dirs = []
        for scenario, payload in scenarios.items():
            batch_dir = self.root / scenario
            self.batch_dirs.append(batch_dir)
            _make_run(batch_dir / f"condition_a_{scenario}_seed8_12-agent", "condition_a", scenario, 8, payload["condition_a"], rival_refusal=scenario == "betrayal_refusal")
            _make_run(batch_dir / f"condition_b_{scenario}_seed8_12-agent", "condition_b", scenario, 8, payload["condition_b"], rival_refusal=scenario == "betrayal_refusal")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_comparison_report_is_generated(self):
        diagnosis = diagnose_condition_b_batches(self.batch_dirs)
        report = build_condition_ab_comparison(self.batch_dirs, diagnosis_report=diagnosis)
        self.assertEqual(report["report_type"], "condition_a_vs_condition_b")
        self.assertEqual(len(report["scenario_reports"]), 3)
        self.assertEqual(report["overall_summary"]["assessment"], "promising_with_calm_context_gap")
        markdown = render_condition_ab_markdown(report)
        self.assertIn("Condition A vs Condition B Comparison", markdown)
        output_paths = write_condition_ab_comparison(report, self.root / "comparison")
        self.assertTrue(output_paths["json"].exists())
        self.assertTrue(output_paths["markdown"].exists())
        payload = json.loads(output_paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["overall_summary"]["assessment"], "promising_with_calm_context_gap")


if __name__ == "__main__":
    unittest.main()
