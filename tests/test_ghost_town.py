import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from ghost_town.analysis import analyze_batch
from ghost_town.emotions import EmergentEmotionEngine, Observation, ProgrammedEmotionEngine
from ghost_town.content import create_buildings
from ghost_town.scenario import create_world
from ghost_town.simulator import GhostTownSimulator
from ghost_town_training.inference import ConditionBInferenceBackend
from run_ghost_town_batch import run_batch


class ProgrammedEmotionEngineTests(unittest.TestCase):
    def test_programmed_emotion_escalates_fear_at_night(self):
        world = create_world(2)
        world.time_of_day = "night"
        world.weather = "storm"
        world.storm_severity = 0.8
        agent = next(iter(world.agents.values()))
        engine = ProgrammedEmotionEngine()
        affect = engine.compute_internal_state(
            agent,
            Observation(
                visible_ghosts=2,
                visible_deaths=1,
                nearby_allies=0,
                trusted_allies=0,
                supplies_seen=0,
                in_shelter=False,
                at_home=False,
                at_work=False,
                nearest_refuge_distance=5,
            ),
            {"average_trust": 0.1},
            world,
        )
        self.assertGreater(affect.affect_vector["fear"], 0.7)
        self.assertGreater(affect.action_bias["seek_safe_house"], 0.8)

    def test_emergent_emotion_uses_persistent_latent_state(self):
        world = create_world(2)
        agent = next(iter(world.agents.values()))
        engine = EmergentEmotionEngine()
        observation = Observation(
            visible_ghosts=1,
            visible_deaths=0,
            nearby_allies=1,
            trusted_allies=1,
            supplies_seen=3,
            in_shelter=False,
            at_home=False,
            at_work=True,
            nearest_refuge_distance=3,
        )
        first = engine.compute_internal_state(agent, observation, {"average_trust": 0.4, "graph_support": 0.5, "graph_tension": 0.0}, world)
        agent.affect_embedding = first.latent_vector
        second = engine.compute_internal_state(agent, observation, {"average_trust": 0.4, "graph_support": 0.5, "graph_tension": 0.0}, world)
        self.assertNotEqual(first.latent_vector, second.latent_vector)
        self.assertIn("probe_no_death", second.prompt_context["summary"])
        self.assertIn("counterfactuals", second.probe_data)
        self.assertIn("with_shelter", second.probe_data["counterfactuals"])

    def test_condition_b_inference_dampens_non_hostile_overreaction(self):
        backend = ConditionBInferenceBackend.__new__(ConditionBInferenceBackend)
        backend.schema = SimpleNamespace(
            action_vocab=[
                "seek_safe_house",
                "hide",
                "seek_hospital",
                "help_other",
                "share_supplies",
                "refuse_help",
                "warn_others",
                "patrol",
                "gather_supplies",
            ]
        )
        calibration = backend._build_calibration_context(
            observation={
                "visible_ghosts": 0,
                "visible_deaths": 0,
                "in_shelter": False,
                "nearest_refuge_distance": 4,
            },
            social_context={"graph_tension": 0.05, "graph_support": 0.45},
            metadata={
                "storm": False,
                "time_of_day": "day",
                "rescue_opportunity": False,
                "refusal_opportunity": False,
                "rival_refusal_opportunity": False,
                "relationship_label_at_decision": "household",
                "tie_value_at_decision": 0.8,
            },
        )
        action_bias = backend._action_bias_from_probs(
            [0.2, 0.2, 0.0, 0.1, 0.05, 0.45, 0.0, 0.0, 0.0],
            calibration,
        )
        affect = backend._affect_from_latent([0.8, 0.7, 0.1, 0.0, 0.9, 0.1, 0.0, 0.0], calibration)
        self.assertTrue(calibration["non_hostile"])
        self.assertGreater(action_bias["help"], -0.1)
        self.assertLess(affect["fear"], 0.35)
        self.assertLess(affect["stress"], 0.35)
        self.assertLess(affect["suspicion"], 0.5)

    def test_condition_b_inference_preserves_rival_refusal_pressure(self):
        backend = ConditionBInferenceBackend.__new__(ConditionBInferenceBackend)
        backend.schema = SimpleNamespace(
            action_vocab=[
                "seek_safe_house",
                "hide",
                "seek_hospital",
                "help_other",
                "share_supplies",
                "refuse_help",
                "warn_others",
                "patrol",
                "gather_supplies",
            ]
        )
        calibration = backend._build_calibration_context(
            observation={
                "visible_ghosts": 0,
                "visible_deaths": 0,
                "in_shelter": False,
                "nearest_refuge_distance": 2,
            },
            social_context={"graph_tension": 0.4, "graph_support": 0.0},
            metadata={
                "storm": False,
                "time_of_day": "day",
                "rescue_opportunity": False,
                "refusal_opportunity": True,
                "rival_refusal_opportunity": True,
                "relationship_label_at_decision": "rivals",
                "tie_value_at_decision": -0.4,
            },
        )
        action_bias = backend._action_bias_from_probs(
            [0.05, 0.05, 0.0, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
            calibration,
        )
        affect = backend._affect_from_latent([0.5, 0.45, 0.1, 0.0, 0.8, 0.1, 0.0, 0.0], calibration)
        self.assertFalse(calibration["non_hostile"])
        self.assertLess(action_bias["help"], -0.2)
        self.assertGreater(affect["suspicion"], 0.7)


class GhostTownSimulationTests(unittest.TestCase):
    def _condition_a_decision(self, simulator: GhostTownSimulator, agent_name: str):
        simulator._update_occupancy()
        agent = simulator.world.agents[agent_name]
        observation = simulator._observe(agent)
        social_context = simulator._social_context(agent)
        affect = simulator.engine.compute_internal_state(agent, observation, social_context, simulator.world)
        return simulator.action_resolver.resolve(agent, observation, affect.affect_vector, affect.action_bias, simulator.world)

    def test_exact_map_metadata_contains_key_locations_and_forest_spawns(self):
        world = create_world(12)
        self.assertEqual((world.width, world.height), (36, 28))
        self.assertIn("Town House", world.buildings)
        self.assertIn("Clinic", world.buildings)
        self.assertIn("Police Station", world.buildings)
        forest_spawns = [building for building in world.buildings.values() if building.ghost_spawn]
        self.assertGreaterEqual(len(forest_spawns), 4)
        self.assertTrue(all(building.category == "forest_edge" for building in forest_spawns))

    def test_roster_has_home_work_and_goal_assignments(self):
        world = create_world(12)
        self.assertEqual(len(world.agents), 12)
        for agent in world.agents.values():
            self.assertTrue(agent.home in world.buildings)
            self.assertTrue(agent.workplace in world.buildings)
            self.assertTrue(agent.day_goals)
            self.assertTrue(agent.night_goals)
            self.assertTrue(agent.role)
            self.assertTrue(agent.skills)
            self.assertTrue(agent.relationship_labels)

    def test_role_assignments_include_medical_and_defense_behaviors(self):
        world = create_world(12)
        self.assertEqual(world.agents["Dr. Mira Chen"].workplace, "Clinic")
        self.assertIn("medicine", world.agents["Dr. Mira Chen"].skills)
        self.assertEqual(world.agents["Sheriff Elias Boone"].workplace, "Police Station")
        self.assertIn("combat", world.agents["Sheriff Elias Boone"].skills)

    def test_crowded_shelter_scenario_repositions_agents_and_reduces_capacity(self):
        world = create_world(12, scenario="crowded_shelter")
        self.assertEqual(world.buildings["Town House"].capacity, 4)
        crowded = [agent.name for agent in world.agents.values() if agent.location == world.buildings["Town House"].location]
        self.assertGreaterEqual(len(crowded), 6)

    def test_warning_updates_trust_memory_for_allied_neighbor(self):
        simulator = GhostTownSimulator(condition="condition_a", agent_count=12, steps=1, seed=7)
        alma = simulator.world.agents["Alma Ward"]
        june = simulator.world.agents["June Carter"]
        june.location = alma.location
        before = june.social_ties["Alma Ward"]
        simulator._warn_neighbors(alma)
        self.assertGreater(june.social_ties["Alma Ward"], before)
        self.assertIn("Alma Ward", june.memory.trust_memory)
        self.assertGreater(june.memory.relationship_events.get("trusted_warning", 0), 0)

    def test_share_supplies_transfers_food_and_builds_trust(self):
        simulator = GhostTownSimulator(condition="condition_a", agent_count=12, steps=1, seed=7)
        rosa = simulator.world.agents["Rosa Mercer"]
        caleb = simulator.world.agents["Caleb Dunn"]
        rosa.location = caleb.location
        rosa.inventory["food"] = 3
        caleb.inventory["food"] = 0
        before = caleb.social_ties["Rosa Mercer"]
        simulator._share_supplies(rosa, "Caleb Dunn")
        self.assertEqual(rosa.inventory["food"], 2)
        self.assertEqual(caleb.inventory["food"], 1)
        self.assertGreater(caleb.social_ties["Rosa Mercer"], before)

    def test_refusal_increases_target_suspicion(self):
        simulator = GhostTownSimulator(condition="condition_a", agent_count=12, steps=1, seed=7)
        nora = simulator.world.agents["Nora Vale"]
        rosa = simulator.world.agents["Rosa Mercer"]
        nora.location = rosa.location
        before = rosa.suspicion
        simulator._refuse_help(nora, "Rosa Mercer")
        self.assertGreater(rosa.suspicion, before)
        self.assertGreater(rosa.memory.relationship_events.get("refusal", 0), 0)

    def test_help_other_creates_rescue_event(self):
        simulator = GhostTownSimulator(condition="condition_a", agent_count=12, steps=1, seed=7)
        mira = simulator.world.agents["Dr. Mira Chen"]
        owen = simulator.world.agents["Owen Pike"]
        owen.location = mira.location
        owen.health = 24
        simulator._help_other(mira, "Owen Pike")
        rescue_events = [event for event in simulator.world.events if event.event_type == "rescue"]
        self.assertTrue(rescue_events)
        self.assertGreater(owen.health, 24)

    def test_rivalry_plus_nearby_need_triggers_refusal(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="betrayal_refusal", agent_count=12, steps=1, seed=7)
        nora = simulator.world.agents["Nora Vale"]
        rosa = simulator.world.agents["Rosa Mercer"]
        ivy = simulator.world.agents["Ivy Hart"]
        ivy.inventory["food"] = 1
        nora.location = simulator.world.buildings["Diner"].location
        rosa.location = simulator.world.buildings["Diner"].location
        nora.inventory["food"] = 1
        rosa.inventory["food"] = 0
        decision = self._condition_a_decision(simulator, "Nora Vale")
        self.assertEqual(decision.action, "refuse_help")
        self.assertEqual(decision.social_target, "Rosa Mercer")

    def test_low_inventory_suppresses_sharing(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="storm_scarcity", agent_count=12, steps=1, seed=7)
        rosa = simulator.world.agents["Rosa Mercer"]
        caleb = simulator.world.agents["Caleb Dunn"]
        rosa.location = caleb.location
        rosa.inventory["food"] = 1
        caleb.inventory["food"] = 0
        decision = self._condition_a_decision(simulator, "Rosa Mercer")
        self.assertNotEqual(decision.action, "share_supplies")

    def test_household_tie_still_allows_sharing_in_non_hostile_context(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="standard_night", agent_count=12, steps=1, seed=7)
        rosa = simulator.world.agents["Rosa Mercer"]
        caleb = simulator.world.agents["Caleb Dunn"]
        rosa.location = caleb.location
        rosa.inventory["food"] = 4
        caleb.inventory["food"] = 0
        simulator.world.time_of_day = "day"
        simulator.world.weather = "clear"
        simulator.world.storm_severity = 0.0
        decision = self._condition_a_decision(simulator, "Rosa Mercer")
        self.assertEqual(decision.action, "share_supplies")
        self.assertEqual(decision.social_target, "Caleb Dunn")

    def test_household_tie_in_same_physical_setup_does_not_trigger_betrayal_refusal(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="betrayal_refusal", agent_count=12, steps=1, seed=7)
        rosa = simulator.world.agents["Rosa Mercer"]
        caleb = simulator.world.agents["Caleb Dunn"]
        rosa.location = simulator.world.buildings["Diner"].location
        caleb.location = simulator.world.buildings["Diner"].location
        rosa.inventory["food"] = 4
        caleb.inventory["food"] = 0
        simulator.world.time_of_day = "day"
        decision = self._condition_a_decision(simulator, "Rosa Mercer")
        self.assertNotEqual(decision.action, "refuse_help")

    def test_storm_pressure_increases_early_shelter_seeking(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="storm_scarcity", agent_count=12, steps=1, seed=7)
        gideon = simulator.world.agents["Gideon Marsh"]
        gideon.location = simulator.world.buildings["Farm Land"].location
        simulator.world.time_of_day = "day"
        simulator.world.weather = "storm"
        simulator.world.storm_severity = 0.75
        decision = self._condition_a_decision(simulator, "Gideon Marsh")
        self.assertIn(decision.action, {"hide", "seek_safe_house"})

    def test_repeated_refusal_marks_betrayal_sequence_context(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="betrayal_refusal", agent_count=12, steps=1, seed=7)
        nora = simulator.world.agents["Nora Vale"]
        rosa = simulator.world.agents["Rosa Mercer"]
        nora.location = simulator.world.buildings["Diner"].location
        rosa.location = simulator.world.buildings["Diner"].location
        simulator._refuse_help(nora, "Rosa Mercer")
        simulator._refuse_help(nora, "Rosa Mercer")
        refusal_events = [event for event in simulator.world.events if event.event_type == "refusal"]
        self.assertEqual(refusal_events[-1].metadata["refusal_context"], "betrayal_sequence")

    def test_training_record_includes_refusal_metadata(self):
        simulator = GhostTownSimulator(condition="condition_a", scenario="betrayal_refusal", agent_count=12, steps=4, seed=7)
        simulator.run()
        refusal_rows = [row for row in simulator.training_records if row.action == "refuse_help"]
        self.assertTrue(refusal_rows)
        metadata = refusal_rows[0].metadata
        self.assertIn("refusal_context", metadata)
        self.assertIn("relationship_label_at_decision", metadata)
        self.assertIn("tie_value_at_decision", metadata)

    def test_simulation_records_permanent_deaths(self):
        simulator = GhostTownSimulator(condition="condition_a", agent_count=12, steps=1, seed=7)
        target = simulator.world.agents["Caleb Dunn"]
        simulator._kill_agent(target, "test fatality", target.location)
        metrics = simulator._compute_metrics()
        death_events = [event for event in simulator.world.events if event.event_type == "death"]
        self.assertTrue(death_events)
        self.assertEqual(metrics["deaths"], len([agent for agent in simulator.world.agents.values() if not agent.alive]))

    def test_export_writes_movement_logs_and_metrics(self):
        simulator = GhostTownSimulator(condition="condition_c", agent_count=12, steps=24, seed=4)
        output = Path("outputs") / "test_export"
        if output.exists():
            shutil.rmtree(output)
        try:
            result = simulator.export(output)
            self.assertTrue((output / "master_movement.json").exists())
            self.assertTrue((output / "metrics.json").exists())
            self.assertTrue((output / "world_metadata.json").exists())
            self.assertTrue((output / "affect_timeline.json").exists())
            self.assertTrue((output / "social_graph_timeline.json").exists())
            self.assertTrue((output / "training_records.jsonl").exists())
            self.assertTrue((output / "run_manifest.json").exists())
            movement = json.loads((output / "master_movement.json").read_text(encoding="utf-8"))
            world_metadata = json.loads((output / "world_metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            affect_timeline = json.loads((output / "affect_timeline.json").read_text(encoding="utf-8"))
            social_graph_timeline = json.loads((output / "social_graph_timeline.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            training_line = (output / "training_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
            training_record = json.loads(training_line)
            self.assertIn("0", {str(key) for key in movement.keys()})
            self.assertIn("Town House", world_metadata["buildings"])
            self.assertIn("mean_social_tie", metrics)
            self.assertIn("relationship_events", metrics)
            self.assertIn("shared_supplies", metrics)
            self.assertIn("refusals", metrics)
            self.assertIn("rescues", metrics)
            self.assertEqual(manifest["schema_version"], "ghost_town.v2")
            self.assertIn("config", manifest)
            self.assertTrue(affect_timeline)
            first_affect_step = affect_timeline[sorted(affect_timeline.keys(), key=int)[0]]
            first_agent = next(iter(first_affect_step.values()))
            self.assertIn("observation", first_agent)
            self.assertIn("probe_data", first_agent)
            self.assertIn("prev_latent", first_agent)
            self.assertTrue(social_graph_timeline)
            first_graph_step = social_graph_timeline[sorted(social_graph_timeline.keys(), key=int)[0]]
            self.assertIn("nodes", first_graph_step)
            self.assertIn("edges", first_graph_step)
            self.assertIn("reward_components", training_record)
            self.assertIn("done", training_record)
        finally:
            if output.exists():
                shutil.rmtree(output)

    def test_invalid_agent_count_is_rejected(self):
        with self.assertRaises(ValueError):
            GhostTownSimulator(condition="condition_a", agent_count=25, steps=1, seed=7)

    def test_batch_runner_creates_aggregate_outputs(self):
        output = Path("outputs") / "test_batch"
        if output.exists():
            shutil.rmtree(output)
        try:
            result = run_batch(output_dir=output, seeds=[3], curriculum="8-agent", storm_mode="none", ghost_mode="standard")
            self.assertTrue((output / "aggregate_metrics.json").exists())
            self.assertTrue((output / "aggregate_metrics.csv").exists())
            self.assertTrue((output / "condition_summary.json").exists())
            self.assertTrue((output / "batch_manifest.json").exists())
            summary = json.loads((output / "condition_summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("condition_a", summary)
            self.assertEqual(manifest["scenario"], "standard_night")
            self.assertEqual(result.batch_id, "test_batch")
        finally:
            if output.exists():
                shutil.rmtree(output)

    def test_analysis_batch_derives_behavior_metrics(self):
        output = Path("outputs") / "test_analysis_batch"
        if output.exists():
            shutil.rmtree(output)
        try:
            run_batch(output_dir=output, seeds=[3], curriculum="8-agent", scenario="storm_scarcity")
            analysis = analyze_batch(output)
            self.assertEqual(analysis["scenario"], "storm_scarcity")
            self.assertIn("condition_stats", analysis)
            self.assertIn("condition_a", analysis["condition_stats"])
            self.assertIn("average_exposure_steps", analysis["condition_stats"]["condition_a"])
            self.assertIn("rival_refusal_conversion", analysis["condition_stats"]["condition_a"])
            self.assertIn("paired_deltas", analysis)
            self.assertTrue(analysis["runs"])
        finally:
            if output.exists():
                shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
