import json
import shutil
import unittest
from pathlib import Path

from ghost_town.comparison_matrix import build_condition_matrix_report, render_condition_matrix_markdown, write_condition_matrix_report
from tests.test_condition_b_diagnosis import _make_run


class ConditionMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("outputs") / "test_condition_matrix"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.batch_dirs = []
        scenarios = {
            "standard_night": {
                "baseline_0": {"survivors": 1, "deaths": 0, "average_stress": 2.0, "average_fear": 0.2, "average_suspicion": 20.0, "relationship_events": 2, "refusals": 0, "ghost_encounters": 0},
                "condition_a": {"survivors": 1, "deaths": 0, "average_stress": 4.0, "average_fear": 1.0, "average_suspicion": 40.0, "relationship_events": 10, "refusals": 2, "ghost_encounters": 0},
                "condition_b": {"survivors": 1, "deaths": 0, "average_stress": 6.0, "average_fear": 2.0, "average_suspicion": 45.0, "relationship_events": 9, "refusals": 1, "ghost_encounters": 0},
                "condition_c": {"survivors": 1, "deaths": 0, "average_stress": 5.0, "average_fear": 1.8, "average_suspicion": 42.0, "relationship_events": 8, "refusals": 1, "ghost_encounters": 0},
            },
            "betrayal_refusal": {
                "baseline_0": {"survivors": 1, "deaths": 0, "average_stress": 1.0, "average_fear": 0.5, "average_suspicion": 30.0, "relationship_events": 1, "refusals": 0, "ghost_encounters": 0},
                "condition_a": {"survivors": 1, "deaths": 0, "average_stress": 5.0, "average_fear": 1.5, "average_suspicion": 70.0, "relationship_events": 20, "refusals": 3, "ghost_encounters": 0},
                "condition_b": {"survivors": 1, "deaths": 0, "average_stress": 4.0, "average_fear": 1.7, "average_suspicion": 60.0, "relationship_events": 18, "refusals": 3, "ghost_encounters": 0},
                "condition_c": {"survivors": 1, "deaths": 0, "average_stress": 4.5, "average_fear": 1.9, "average_suspicion": 62.0, "relationship_events": 17, "refusals": 2, "ghost_encounters": 0},
            },
        }
        for scenario, payload in scenarios.items():
            batch_dir = self.root / scenario
            self.batch_dirs.append(batch_dir)
            for condition, metrics in payload.items():
                _make_run(batch_dir / f"{condition}_{scenario}_seed8_12-agent", condition, scenario, 8, metrics, rival_refusal=scenario == "betrayal_refusal" and condition != "baseline_0")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_matrix_report_is_generated(self):
        report = build_condition_matrix_report(self.batch_dirs)
        self.assertEqual(report["report_type"], "condition_matrix")
        self.assertEqual(len(report["scenario_reports"]), 2)
        markdown = render_condition_matrix_markdown(report)
        self.assertIn("Condition Matrix", markdown)
        paths = write_condition_matrix_report(report, self.root / "matrix")
        self.assertTrue(paths["json"].exists())
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["overall_summary"]["conditions"], ["baseline_0", "condition_a", "condition_b", "condition_c"])


if __name__ == "__main__":
    unittest.main()
