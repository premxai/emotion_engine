from __future__ import annotations

import random
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .emotions import (
    EmotionEngine,
    EmergentEmotionEngine,
    HybridEmotionEngine,
    NeutralEmotionEngine,
    Observation,
    ProgrammedEmotionEngine,
    build_condition_b_engine,
    build_condition_c_engine,
    build_condition_d_engine,
)
from .export import export_simulation
from .pathfinding import find_path
from .planner import ActionResolver, manhattan, step_toward
from .scenario import create_world, get_scenario_preset, resolve_mode_for_scenario
from .types import (
    ActionDecision,
    EventLog,
    ExperimentConfig,
    GhostState,
    RunManifest,
    TrainingStepRecord,
    WorldState,
)

SUPPORTED_CONDITIONS = {"baseline_0", "condition_a", "condition_b", "condition_c", "condition_d"}


@dataclass
class SimulationResult:
    condition: str
    world: WorldState
    timeline: Dict[int, Dict[str, object]]
    metrics: Dict[str, object]
    affect_timeline: Dict[int, Dict[str, object]]
    social_graph_timeline: Dict[int, Dict[str, object]]
    training_records: List[TrainingStepRecord]
    config: ExperimentConfig


class GhostTownSimulator:
    def __init__(
        self,
        condition: str,
        agent_count: int = 12,
        steps: int = 72,
        seed: int = 7,
        storm_mode: str | None = None,
        ghost_mode: str | None = None,
        scenario: str = "standard_night",
        run_id: str = "",
        curriculum_stage: str = "",
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
        building_location_overrides: Dict[str, Tuple[int, int]] | None = None,
    ):
        if condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported condition: {condition}")
        self.scenario = scenario
        self.scenario_preset = get_scenario_preset(scenario)
        resolved_modes = resolve_mode_for_scenario(scenario, storm_mode, ghost_mode)
        self.condition_b_checkpoint = str(condition_b_checkpoint or "")
        self.condition_b_schema = str(condition_b_schema or "")
        self.condition_b_policy_mode = str(condition_b_policy_mode or "deterministic")
        self.condition_b_temperature = float(condition_b_temperature or 1.0)
        self.condition_c_checkpoint = str(condition_c_checkpoint or "")
        self.condition_c_schema = str(condition_c_schema or "")
        self.condition_d_checkpoint = str(condition_d_checkpoint or "")
        self.condition_d_schema = str(condition_d_schema or "")
        self.condition_b_policy_source = (
            "trained_checkpoint"
            if condition == "condition_b" and self.condition_b_checkpoint and self.condition_b_schema
            else "scaffold_fallback"
        )
        if condition == "condition_b" and self.condition_b_policy_source == "trained_checkpoint" and self.condition_b_policy_mode == "sample":
            self.condition_b_policy_source = "trained_checkpoint_sampled_rl"
        self.condition_c_policy_source = (
            "trained_checkpoint"
            if condition == "condition_c" and self.condition_c_checkpoint and self.condition_c_schema
            else "hybrid_fallback"
        )
        self.condition_d_policy_source = (
            "trained_checkpoint"
            if condition == "condition_d" and self.condition_d_checkpoint and self.condition_d_schema
            else "scaffold_fallback"
        )
        self.runtime_backend = str(runtime_backend or "native")
        self.tmx_path = str(tmx_path or "")
        self.semantics_manifest = str(semantics_manifest or "")
        self.transfer_source_manifest = str(transfer_source_manifest or "")
        self.config = ExperimentConfig(
            condition=condition,
            seed=seed,
            steps=steps,
            agent_count=agent_count,
            scenario=scenario,
            storm_mode=resolved_modes["storm_mode"],
            ghost_mode=resolved_modes["ghost_mode"],
            run_id=run_id or f"{condition}_seed{seed}_a{agent_count}_s{steps}",
            curriculum_stage=curriculum_stage or f"{agent_count}-agent",
            condition_b_policy_source=self.condition_b_policy_source,
            condition_b_checkpoint=self.condition_b_checkpoint,
            condition_b_schema=self.condition_b_schema,
            condition_b_policy_mode=self.condition_b_policy_mode,
            condition_b_temperature=self.condition_b_temperature,
            condition_c_policy_source=self.condition_c_policy_source,
            condition_c_checkpoint=self.condition_c_checkpoint,
            condition_c_schema=self.condition_c_schema,
            condition_d_policy_source=self.condition_d_policy_source,
            condition_d_checkpoint=self.condition_d_checkpoint,
            condition_d_schema=self.condition_d_schema,
            runtime_backend=self.runtime_backend,
            tmx_path=self.tmx_path,
            semantics_manifest=self.semantics_manifest,
            transfer_source_manifest=self.transfer_source_manifest,
        )
        self.condition = condition
        self.engine = self._build_engine()
        self.world = create_world(agent_count, scenario=scenario)
        self.world.scenario = scenario
        self._apply_runtime_overrides(
            width_override=world_width_override,
            height_override=world_height_override,
            building_location_overrides=building_location_overrides,
        )
        self._validate_world(agent_count)
        self.steps = steps
        self.random = random.Random(seed)
        self.action_resolver = ActionResolver()
        self.timeline: Dict[int, Dict[str, object]] = {}
        self.affect_timeline: Dict[int, Dict[str, object]] = {}
        self.social_graph_timeline: Dict[int, Dict[str, object]] = {}
        self.training_records: List[TrainingStepRecord] = []
        self.step_events: List[EventLog] = []
        self.ghost_counter = 0
        self.phase_offset = int(self.scenario_preset.get("phase_offset", 0))
        self._scenario_script_steps_run: set[int] = set()

    def run(self) -> SimulationResult:
        _prev_time_of_day = "day"
        for step in range(self.steps):
            self.world.step = step
            self._update_time_and_weather(step)
            # Dawn transition: consolidate last night's memory into persistent state
            if _prev_time_of_day == "night" and self.world.time_of_day == "sunrise":
                self._dawn_memory_consolidation()
            _prev_time_of_day = self.world.time_of_day
            self._spawn_ghosts()
            self._update_occupancy()
            self._apply_scenario_script(step)
            decisions = self._decide_actions()
            self._apply_actions(decisions)
            self._move_ghosts()
            self._resolve_hazards()
            self._recover_and_decay()
            self._record_training_step(decisions)
            self._record_timeline()
        metrics = self._compute_metrics()
        return SimulationResult(
            self.condition,
            self.world,
            self.timeline,
            metrics,
            self.affect_timeline,
            self.social_graph_timeline,
            self.training_records,
            self.config,
        )

    def export(self, destination: Path) -> SimulationResult:
        result = self.run()
        export_simulation(
            destination,
            result.world,
            result.timeline,
            result.metrics,
            affect_timeline=result.affect_timeline,
            social_graph_timeline=result.social_graph_timeline,
            training_records=result.training_records,
            run_manifest=self._build_run_manifest(destination, result.metrics),
        )
        return result

    def _build_engine(self) -> EmotionEngine:
        if self.condition == "baseline_0":
            return NeutralEmotionEngine()
        if self.condition == "condition_a":
            return ProgrammedEmotionEngine()
        if self.condition == "condition_b":
            return build_condition_b_engine(
                checkpoint_path=self.condition_b_checkpoint,
                schema_path=self.condition_b_schema,
                policy_mode=self.condition_b_policy_mode,
                temperature=self.condition_b_temperature,
                rng_seed=self.config.seed,
            )
        if self.condition == "condition_c":
            return build_condition_c_engine(
                checkpoint_path=self.condition_c_checkpoint,
                schema_path=self.condition_c_schema,
            )
        if self.condition == "condition_d":
            return build_condition_d_engine(
                checkpoint_path=self.condition_d_checkpoint,
                schema_path=self.condition_d_schema,
            )
        raise ValueError(f"Unsupported condition: {self.condition}")

    def _update_time_and_weather(self, step: int) -> None:
        self.step_events = []
        total_phase = step + self.phase_offset
        phase = total_phase % 24
        if phase < 10:
            self.world.time_of_day = "day"
        elif phase < 13:
            self.world.time_of_day = "sunset"
        elif phase < 21:
            self.world.time_of_day = "night"
        else:
            self.world.time_of_day = "sunrise"
        self.world.day_index = total_phase // 24 + 1

        storm_roll = self.random.random()
        if self.config.storm_mode == "none":
            self.world.weather = "clear"
            self.world.storm_severity = 0.0
            return
        if self.config.storm_mode == "scheduled":
            if phase in {9, 18}:
                self.world.weather = "storm"
                self.world.storm_severity = 0.75
                self._log("storm", "a scheduled storm front rolls through town", metadata={"severity": self.world.storm_severity})
                return
            self.world.weather = "clear"
            self.world.storm_severity = 0.0
            return
        if phase in {8, 9, 10, 18, 19} and storm_roll > 0.72:
            self.world.weather = "storm"
            self.world.storm_severity = round(min(1.0, 0.4 + self.random.random() * 0.6), 2)
            self._log("storm", "a storm front rolls through town", metadata={"severity": self.world.storm_severity})
        elif self.world.weather == "storm" and storm_roll < 0.4:
            self.world.weather = "clear"
            self.world.storm_severity = 0.0
            self._log("weather_clear", "the storm starts to clear")
        elif self.world.weather != "storm":
            self.world.weather = "clear"
            self.world.storm_severity = 0.0

    def _spawn_ghosts(self) -> None:
        if self.config.ghost_mode == "none":
            self.world.ghosts = []
            return
        if self.world.time_of_day not in {"sunset", "night"}:
            self.world.ghosts = []
            return
        desired = 4 if self.world.time_of_day == "night" else 2
        if self.config.ghost_mode == "high_pressure":
            desired += 2
        while len(self.world.ghosts) < desired:
            spawn_buildings = [building for building in self.world.buildings.values() if building.ghost_spawn]
            spawn = self.random.choice(spawn_buildings)
            ghost = GhostState(f"ghost_{self.ghost_counter}", spawn.location)
            self.ghost_counter += 1
            self.world.ghosts.append(ghost)
            self._log("ghost_spawn", f"{ghost.ghost_id} emerges from the forest", location=spawn.location)

    def _update_occupancy(self) -> None:
        self.world.occupancy = {name: [] for name in self.world.buildings}
        for agent in self.world.agents.values():
            if not agent.alive:
                continue
            match = self._building_at(agent.location)
            agent.sheltered = bool(match and match.safe_at_night)
            if match:
                self.world.occupancy[match.name].append(agent.name)
        self._apply_crowding_effects()

    def _decide_actions(self) -> Dict[str, ActionDecision]:
        decisions: Dict[str, ActionDecision] = {}
        step_affect: Dict[str, object] = {}
        for agent in self.world.agents.values():
            if not agent.alive:
                continue
            observation = self._observe(agent)
            social_context = self._social_context(agent)
            opportunity = self._opportunity_snapshot(agent)
            prev_latent = list(agent.affect_embedding)
            if getattr(self.engine, "inference_backend", None) is not None and self.condition in {"condition_b", "condition_c"}:
                affect = self.engine.inference_backend.compute(
                    observation={
                        "visible_ghosts": observation.visible_ghosts,
                        "visible_deaths": observation.visible_deaths,
                        "nearby_allies": observation.nearby_allies,
                        "trusted_allies": observation.trusted_allies,
                        "supplies_seen": observation.supplies_seen,
                        "in_shelter": observation.in_shelter,
                        "at_home": observation.at_home,
                        "at_work": observation.at_work,
                        "nearest_refuge_distance": observation.nearest_refuge_distance,
                        "steps_since_ghost_seen": observation.steps_since_ghost_seen,
                        "steps_since_ally_died": observation.steps_since_ally_died,
                        "steps_since_betrayal": observation.steps_since_betrayal,
                        "ally_deaths_witnessed": observation.ally_deaths_witnessed,
                        "betrayals_received": observation.betrayals_received,
                    },
                    social_context=social_context,
                    prev_latent=prev_latent,
                    metadata={
                        "health": round(agent.health, 2),
                        "storm": self.world.weather == "storm",
                        "alive": agent.alive,
                        "sheltered": agent.sheltered,
                        "rescue_opportunity": opportunity["rescue_opportunity"],
                        "refusal_opportunity": opportunity["refusal_opportunity"],
                        "rival_refusal_opportunity": opportunity["rival_refusal_opportunity"],
                        "scenario": self.config.scenario,
                        "role": agent.role,
                        "refusal_context": opportunity["refusal_context"],
                        "relationship_label_at_decision": opportunity["relationship_label_at_decision"],
                        "tie_value_at_decision": opportunity["tie_value_at_decision"],
                        "time_of_day": self.world.time_of_day,
                    },
                )
            else:
                affect = self.engine.compute_internal_state(agent, observation, social_context, self.world)
            agent.fear = self._clamp(agent.fear * 0.5 + affect.affect_vector.get("fear", 0.0) * 100.0 * 0.5)
            agent.stress = self._clamp(agent.stress * 0.4 + affect.affect_vector.get("stress", 0.0) * 100.0 * 0.6)
            agent.trust = self._clamp(agent.trust * 50.0 * 0.2 + affect.affect_vector.get("trust", 0.0) * 100.0 * 0.8) / 100.0
            agent.grief = self._clamp(agent.grief * 0.8 + affect.affect_vector.get("grief", 0.0) * 100.0 * 0.4)
            agent.suspicion = self._clamp(agent.suspicion * 0.7 + affect.affect_vector.get("suspicion", 0.0) * 100.0 * 0.5)
            agent.affect_embedding = affect.latent_vector
            agent.prompt_context = affect.prompt_context
            agent.probe_data = affect.probe_data
            decision = self.action_resolver.resolve(agent, observation, affect.affect_vector, affect.action_bias, self.world)
            agent.current_goal = decision.goal
            agent.current_action = decision.action
            agent.destination = decision.destination
            agent.destination_name = decision.destination_name
            decisions[agent.name] = decision
            step_affect[agent.name] = {
                "observation": {
                    "visible_ghosts": observation.visible_ghosts,
                    "visible_deaths": observation.visible_deaths,
                    "nearby_allies": observation.nearby_allies,
                    "trusted_allies": observation.trusted_allies,
                    "supplies_seen": observation.supplies_seen,
                    "in_shelter": observation.in_shelter,
                    "at_home": observation.at_home,
                    "at_work": observation.at_work,
                    "nearest_refuge_distance": observation.nearest_refuge_distance,
                    "steps_since_ghost_seen": observation.steps_since_ghost_seen,
                    "steps_since_ally_died": observation.steps_since_ally_died,
                    "steps_since_betrayal": observation.steps_since_betrayal,
                    "ally_deaths_witnessed": observation.ally_deaths_witnessed,
                    "betrayals_received": observation.betrayals_received,
                },
                "social_context": {key: round(value, 3) for key, value in social_context.items()},
                "affect_vector": {key: round(value, 3) for key, value in affect.affect_vector.items()},
                "action_bias": {key: round(value, 3) for key, value in affect.action_bias.items()},
                "latent_vector": [round(value, 3) for value in affect.latent_vector],
                "prev_latent": [round(value, 3) for value in prev_latent],
                "probe_data": affect.probe_data,
                "decision": {
                    "action": decision.action,
                    "goal": decision.goal,
                    "destination": list(decision.destination) if decision.destination else None,
                    "destination_name": decision.destination_name,
                    "social_target": decision.social_target,
                },
            }
        self.affect_timeline[self.world.step] = step_affect
        self.social_graph_timeline[self.world.step] = self._graph_snapshot()
        return decisions

    def _apply_actions(self, decisions: Dict[str, ActionDecision]) -> None:
        for name, decision in decisions.items():
            agent = self.world.agents[name]
            if not agent.alive:
                continue
            if decision.action == "share_supplies":
                self._share_supplies(agent, decision.social_target)
                continue
            if decision.action == "refuse_help":
                self._refuse_help(agent, decision.social_target)
                continue
            if decision.action == "warn_others":
                self._warn_neighbors(agent)
                self._log("warning", decision.description, actor=agent.name, location=agent.location)
                agent.dialogue = [[agent.name, "Ghosts in the dark. Move now."]]
                continue
            if decision.destination is not None:
                # Recompute BFS path when destination changes or path is exhausted.
                # Compare against nav_path's target tile (last element) not agent.destination
                # because agent.destination is updated before _apply_actions runs.
                nav_target = agent.nav_path[-1] if agent.nav_path else None
                if nav_target != decision.destination or not agent.nav_path:
                    agent.nav_path = find_path(agent.location, decision.destination)
                # Pop next tile from cached path
                if agent.nav_path:
                    next_location = agent.nav_path.pop(0)
                else:
                    next_location = agent.location  # already at destination
                if next_location != agent.location:
                    agent.location = next_location
                    agent.stamina = max(0.0, agent.stamina - (3.0 + self.world.storm_severity * 2.0))
                    agent.dialogue = []
                if decision.action == "gather_supplies" and agent.location == decision.destination:
                    self._collect_supplies(agent, decision.destination_name or "")
                elif decision.action == "help_other":
                    self._help_other(agent, decision.social_target)
                    agent.dialogue = [[agent.name, "Stay with me, I am getting you to the clinic."]]
                elif decision.action in {"seek_safe_house", "seek_hospital", "hide"}:
                    agent.dialogue = [[agent.name, "Keep moving. Stay alive."]]
                elif decision.action == "patrol":
                    agent.dialogue = [[agent.name, "Keep the roads clear. Watch the tree line."]]
            elif decision.action == "rest":
                agent.stamina = min(100.0, agent.stamina + 6.0)
                agent.dialogue = []
            self._log("action", decision.description, actor=agent.name, location=agent.location, metadata={"action": decision.action})

    def _move_ghosts(self) -> None:
        # Ghosts use BFS pathing and move 2 tiles per step to intercept fleeing agents
        fallback_refuge = "Town House" if "Town House" in self.world.buildings else "Safe House"
        for ghost in self.world.ghosts:
            targets = [agent for agent in self.world.agents.values() if agent.alive and not agent.sheltered]
            if not targets:
                target_loc = self.world.buildings[fallback_refuge].location
                ghost.target_agent = fallback_refuge
            else:
                target = min(targets, key=lambda agent: manhattan(agent.location, ghost.location))
                target_loc = target.location
                ghost.target_agent = target.name
            ghost_path = find_path(ghost.location, target_loc)
            if len(ghost_path) >= 3:
                ghost.location = ghost_path[2]   # advance 3 tiles along BFS path
            elif len(ghost_path) >= 2:
                ghost.location = ghost_path[1]   # advance 2 tiles
            elif ghost_path:
                ghost.location = ghost_path[0]   # advance 1 tile

    def _resolve_hazards(self) -> None:
        if self.world.weather == "storm":
            for key in ("Diner", "Farm Land"):
                loss = self.random.randint(0, 2)
                self.world.supplies[key] = max(0, self.world.supplies.get(key, 0) - loss)
            for agent in self.world.agents.values():
                if not agent.alive:
                    continue
                if not agent.sheltered:
                    agent.health = max(0.0, agent.health - (2.0 + self.world.storm_severity * 2.0))
                    agent.stress = self._clamp(agent.stress + 5.0 + self.world.storm_severity * 4.0)
                    if agent.health < 70:
                        agent.memory.relationship_events["storm_injury"] = agent.memory.relationship_events.get("storm_injury", 0) + 1

        for ghost in self.world.ghosts:
            for agent in self.world.agents.values():
                if not agent.alive or agent.sheltered:
                    continue
                if manhattan(ghost.location, agent.location) == 0:
                    if self.random.random() < 0.55:
                        self._kill_agent(agent, f"{ghost.ghost_id} kills {agent.name}", ghost.location)
                    else:
                        agent.health = max(0.0, agent.health - 25.0)
                        agent.fear = self._clamp(agent.fear + 20.0)
                        agent.stress = self._clamp(agent.stress + 10.0)
                        self._log("ghost_attack", f"{ghost.ghost_id} wounds {agent.name}", actor=ghost.ghost_id, target=agent.name, location=ghost.location)

    def _recover_and_decay(self) -> None:
        for agent in self.world.agents.values():
            if not agent.alive:
                continue
            building = self._building_at(agent.location)
            if agent.sheltered:
                self._apply_shelter_social_effects(agent)
                agent.stamina = min(100.0, agent.stamina + 4.0)
                shelter_quality = building.shelter_quality if building else 0.4
                if building and building.treatment_zone:
                    agent.health = min(100.0, agent.health + 7.0)
                    if agent.inventory.get("medicine", 0) > 0 and agent.health < 85:
                        agent.inventory["medicine"] -= 1
                        agent.health = min(100.0, agent.health + 5.0)
                elif agent.destination_name == "Clinic":
                    agent.health = min(100.0, agent.health + 4.0)
                agent.stress = max(0.0, agent.stress - (1.5 + shelter_quality * 2.5))
                agent.fear = max(0.0, agent.fear - (2.0 + shelter_quality * 2.0))
            else:
                agent.stress = self._clamp(agent.stress + 1.0)
            if agent.inventory.get("food", 0) == 0:
                agent.health = max(0.0, agent.health - 0.8)
                agent.stress = self._clamp(agent.stress + 1.5)
            if self.world.time_of_day == "day" and agent.inventory.get("food", 0) > 0 and self.world.step % 6 == 0:
                agent.inventory["food"] -= 1
                agent.health = min(100.0, agent.health + 2.0)

    def _record_timeline(self) -> None:
        frame = {}
        for agent in self.world.agents.values():
            frame[agent.name] = {
                "movement": [agent.location[0], agent.location[1]],
                "pronunciatio": "D" if not agent.alive else self._pronunciatio(agent.current_action),
                "description": self._description(agent),
                "chat": agent.dialogue if agent.dialogue else None,
                "alive": agent.alive,
                "goal": agent.current_goal,
                "fear": round(agent.fear, 2),
                "stress": round(agent.stress, 2),
                "grief": round(agent.grief, 2),
                "nights_survived": agent.nights_survived,
                "last_night_ghost_seen": agent.last_night_ghost_seen,
                "last_night_death_nearby": agent.last_night_death_nearby,
                "ally_deaths_witnessed": agent.ally_deaths_witnessed,
            }
        frame["ghosts"] = [{"ghost_id": ghost.ghost_id, "location": [ghost.location[0], ghost.location[1]]} for ghost in self.world.ghosts]
        _phase = (self.world.step + self.phase_offset) % 24
        _clock_hour = (_phase + 8) % 24  # phase 0 → 8 AM
        _am_pm = "AM" if _clock_hour < 12 else "PM"
        _h12 = _clock_hour % 12 or 12
        frame["meta"] = {
            "curr_time": f"Day {self.world.day_index} \u00b7 {_h12}:00 {_am_pm}",
            "time_of_day": self.world.time_of_day,
            "day_index": self.world.day_index,
            "weather": self.world.weather,
            "storm_severity": self.world.storm_severity,
            "ghost_count": len(self.world.ghosts),
            "scenario": self.config.scenario,
        }
        self.timeline[self.world.step] = frame

    def _graph_snapshot(self) -> Dict[str, object]:
        nodes = {}
        edges = []
        for agent in self.world.agents.values():
            nodes[agent.name] = {
                "alive": agent.alive,
                "role": agent.role,
                "location": list(agent.location),
                "sheltered": agent.sheltered,
                "stress": round(agent.stress, 2),
                "fear": round(agent.fear, 2),
                "trust_state": round(agent.trust, 3),
                "suspicion": round(agent.suspicion, 2),
                "building": self._building_at(agent.location).name if self._building_at(agent.location) else None,
            }
            for other_name, weight in agent.social_ties.items():
                if other_name not in self.world.agents:
                    continue
                other = self.world.agents[other_name]
                edges.append(
                    {
                        "source": agent.name,
                        "target": other_name,
                        "weight": round(weight, 3),
                        "label": agent.relationship_labels.get(other_name, "neighbors"),
                        "distance": manhattan(agent.location, other.location),
                        "co_present": self._building_at(agent.location) is not None
                        and self._building_at(other.location) is not None
                        and self._building_at(agent.location).name == self._building_at(other.location).name,
                    }
                )
        return {
            "meta": {
                "step": self.world.step,
                "time_of_day": self.world.time_of_day,
                "weather": self.world.weather,
                "storm_severity": self.world.storm_severity,
            },
            "occupancy": {name: occupants[:] for name, occupants in self.world.occupancy.items()},
            "nodes": nodes,
            "edges": edges,
        }

    def _compute_metrics(self) -> Dict[str, object]:
        survivors = [agent for agent in self.world.agents.values() if agent.alive]
        deaths = [agent for agent in self.world.agents.values() if not agent.alive]
        avg_stress = round(sum(agent.stress for agent in self.world.agents.values()) / max(1, len(self.world.agents)), 2)
        avg_fear = round(sum(agent.fear for agent in self.world.agents.values()) / max(1, len(self.world.agents)), 2)
        avg_suspicion = round(sum(agent.suspicion for agent in self.world.agents.values()) / max(1, len(self.world.agents)), 2)
        avg_trust_state = round(sum(agent.trust for agent in self.world.agents.values()) / max(1, len(self.world.agents)), 2)
        social_edges = sum(len(agent.social_ties) for agent in self.world.agents.values())
        mean_social_tie = round(
            sum(sum(agent.social_ties.values()) for agent in self.world.agents.values()) / max(1, social_edges),
            3,
        )
        relationship_events = sum(sum(agent.memory.relationship_events.values()) for agent in self.world.agents.values())
        policy_source = ""
        if self.condition == "condition_b":
            policy_source = self.condition_b_policy_source
        elif self.condition == "condition_c":
            policy_source = self.condition_c_policy_source
        return {
                "condition": self.condition,
                "policy_source": policy_source,
                "scenario": self.config.scenario,
                "run_id": self.config.run_id,
                "seed": self.config.seed,
            "steps": self.steps,
            "survivors": len(survivors),
            "deaths": len(deaths),
            "average_stress": avg_stress,
            "average_fear": avg_fear,
            "average_suspicion": avg_suspicion,
            "average_trust_state": avg_trust_state,
            "mean_social_tie": mean_social_tie,
            "event_count": len(self.world.events),
            "social_edges": social_edges,
            "relationship_events": relationship_events,
            "rescues": len([event for event in self.world.events if event.event_type == "rescue"]),
            "shared_supplies": len([event for event in self.world.events if event.event_type == "shared_supplies"]),
            "refusals": len([event for event in self.world.events if event.event_type == "refusal"]),
            "ghost_encounters": len([event for event in self.world.events if event.event_type in {"ghost_attack", "death"}]),
            "storm_events": len([event for event in self.world.events if event.event_type == "storm"]),
        }

    def _collect_supplies(self, agent, building_name: str) -> None:
        available = self.world.supplies.get(building_name, 0)
        if available <= 0:
            self._log("supply_empty", f"{agent.name} finds no supplies in {building_name}", actor=agent.name, location=agent.location)
            return
        building = self.world.buildings.get(building_name)
        gather_cap = 2
        if building and building.supply_spawn:
            gather_cap += max(0, min(2, building.supply_spawn // 5))
        gathered = min(gather_cap, available)
        self.world.supplies[building_name] = available - gathered
        agent.inventory["food"] = agent.inventory.get("food", 0) + gathered
        if building_name == "Clinic":
            agent.inventory["medicine"] = agent.inventory.get("medicine", 0) + 1
        self._log("supply_gathered", f"{agent.name} gathers supplies at {building_name}", actor=agent.name, location=agent.location, metadata={"amount": gathered})

    def _share_supplies(self, agent, target_name: Optional[str]) -> None:
        if not target_name or target_name not in self.world.agents:
            return
        target = self.world.agents[target_name]
        if not target.alive or manhattan(agent.location, target.location) > 2:
            return
        if agent.inventory.get("food", 0) <= 0:
            self._refuse_help(agent, target_name, reason="no_supplies")
            return
        agent.inventory["food"] -= 1
        target.inventory["food"] = target.inventory.get("food", 0) + 1
        target.stress = self._clamp(target.stress - 2.0)
        target.fear = self._clamp(target.fear - 1.5)
        target.suspicion = self._clamp(target.suspicion - 2.0)
        self._shift_relationship(agent, target_name, 0.04, "shared_supplies")
        self._shift_relationship(target, agent.name, 0.05, "received_supplies")
        agent.dialogue = [[agent.name, f"Take this, {target.name}."]]
        target.dialogue = [[target.name, "Thank you."]]
        self._log(
            "shared_supplies",
            f"{agent.name} shares food with {target.name}",
            actor=agent.name,
            target=target.name,
            location=agent.location,
            metadata={
                "relationship_label_at_decision": agent.relationship_labels.get(target.name, "neighbors"),
                "tie_value_at_decision": round(agent.social_ties.get(target.name, 0.0), 3),
            },
        )

    def _refuse_help(self, agent, target_name: Optional[str], reason: str = "refusal") -> None:
        if not target_name or target_name not in self.world.agents:
            return
        target = self.world.agents[target_name]
        if not target.alive or manhattan(agent.location, target.location) > 2:
            return
        refusal_metadata = self._refusal_metadata(agent, target, reason)
        target.stress = self._clamp(target.stress + 4.0)
        target.suspicion = self._clamp(target.suspicion + 5.0)
        target.steps_since_betrayal = 0
        target.betrayals_received += 1
        self._shift_relationship(agent, target_name, -0.04, reason)
        self._shift_relationship(target, agent.name, -0.06, reason)
        if target.memory.relationship_events.get(reason, 0) >= 2:
            target.memory.relationship_events["repeat_betrayal"] = target.memory.relationship_events.get("repeat_betrayal", 0) + 1
            target.trust = max(0.0, target.trust - 0.08)
        agent.dialogue = [[agent.name, "I can't help you now."]]
        target.dialogue = [[target.name, "So that's your answer."]]
        self._log("refusal", f"{agent.name} refuses help to {target.name}", actor=agent.name, target=target.name, location=agent.location, metadata=refusal_metadata)
        refusal_event = EventLog(
            self.world.step,
            self.world.time_of_day,
            self.world.weather,
            "refusal",
            f"{agent.name} refuses help to {target.name}",
            actor=agent.name,
            target=target.name,
            location=agent.location,
            metadata=refusal_metadata,
        )
        agent.memory.recent_events.append(refusal_event)
        target.memory.recent_events.append(refusal_event)

    def _help_other(self, agent, target_name: Optional[str]) -> None:
        if not target_name or target_name not in self.world.agents:
            return
        target = self.world.agents[target_name]
        if not target.alive or manhattan(agent.location, target.location) > 2:
            return
        target.health = min(100.0, target.health + 14.0)
        target.stress = self._clamp(target.stress - 3.0)
        target.fear = self._clamp(target.fear - 2.0)
        self._shift_relationship(agent, target_name, 0.05, "rescue")
        self._shift_relationship(target, agent.name, 0.07, "rescued_by")
        self._log("rescue", f"{agent.name} helps {target.name} reach treatment", actor=agent.name, target=target.name, location=agent.location)

    def _warn_neighbors(self, agent) -> None:
        for other in self.world.agents.values():
            if other.name == agent.name or not other.alive:
                continue
            if manhattan(agent.location, other.location) <= 3:
                other.fear = self._clamp(other.fear + 8.0)
                label = other.relationship_labels.get(agent.name, "neighbors")
                if label in {"allies", "partners", "medical_team", "mentor_pair", "friends", "command_pair"}:
                    self._shift_relationship(other, agent.name, 0.03, "trusted_warning")
                elif label in {"rivals", "strained"}:
                    self._shift_relationship(other, agent.name, -0.01, "dismissed_warning")
                other.memory.recent_events.append(
                    EventLog(self.world.step, self.world.time_of_day, self.world.weather, "warning", f"{agent.name} warns {other.name} about danger", actor=agent.name, target=other.name, location=agent.location)
                )

    def _dawn_memory_consolidation(self) -> None:
        """Called once per night→sunrise transition.
        Updates OBSERVATIONAL counters only — no emotion state is touched.
        Condition A's emotion engine will respond to these observations through
        its own formulas. Condition D's model receives them as inputs and must
        learn what to do with them on its own.
        """
        for agent in self.world.agents.values():
            if not agent.alive:
                continue
            agent.nights_survived += 1

            # Observational flags — legitimate inputs to ALL emotion engines
            saw_ghost = agent.steps_since_ghost_seen < 8
            saw_death = agent.steps_since_ally_died < 8
            agent.last_night_ghost_seen = saw_ghost
            agent.last_night_death_nearby = saw_death

            # Remember ghost zone locations (observation, not emotion manipulation)
            if saw_ghost:
                for ghost in self.world.ghosts:
                    loc = ghost.location
                    if loc not in agent.ghost_zones_known:
                        agent.ghost_zones_known.append(loc)
                agent.ghost_zones_known = agent.ghost_zones_known[-5:]

            # Social information spreading at dawn — legitimate world event,
            # not emotion injection. Emotion engines respond to this via visible_deaths
            # and steps_since_ally_died in their own observation next step.
            if (saw_ghost or saw_death) and agent.dialogue == []:
                nearby_names = [
                    a.name for a in self.world.agents.values()
                    if a.alive and a.name != agent.name and manhattan(a.location, agent.location) <= 4
                ]
                if nearby_names:
                    target_name = nearby_names[0]
                    msg = (
                        "I watched someone die last night. We can't go out unprotected."
                        if saw_death else
                        "The ghosts were right outside. Stay close to shelter tonight."
                    )
                    agent.dialogue = [[agent.name, msg]]
                    self._log(
                        "warning",
                        f"{agent.name} warns {target_name} about last night",
                        actor=agent.name,
                        target=target_name,
                        location=agent.location,
                    )

    def _kill_agent(self, agent, description: str, location: Tuple[int, int]) -> None:
        agent.alive = False
        agent.health = 0.0
        agent.current_action = "dead"
        agent.current_goal = "dead"
        self._log("death", description, target=agent.name, location=location)
        for witness in self.world.agents.values():
            if not witness.alive or witness.name == agent.name:
                continue
            if manhattan(witness.location, location) <= 4:
                tie = witness.social_ties.get(agent.name, 0.1)
                witness.grief = self._clamp(witness.grief + 18.0 + tie * 40.0)
                witness.fear = self._clamp(witness.fear + 12.0)
                witness.stress = self._clamp(witness.stress + 14.0)
                witness.memory.known_deaths.append(agent.name)
                witness.steps_since_ally_died = 0
                witness.ally_deaths_witnessed += 1
                if tie > 0.5:
                    witness.suspicion = self._clamp(witness.suspicion + 6.0)
                    witness.trust = max(0.0, witness.trust - 0.05)
                witness.memory.recent_events.append(
                    EventLog(self.world.step, self.world.time_of_day, self.world.weather, "witnessed_death", f"{witness.name} witnesses the death of {agent.name}", actor=witness.name, target=agent.name, location=location)
                )

    def _observe(self, agent) -> Observation:
        visible_ghosts = sum(1 for ghost in self.world.ghosts if manhattan(agent.location, ghost.location) <= 4)
        visible_deaths = 0
        nearby_allies = 0
        trusted_allies = 0
        for other in self.world.agents.values():
            if other.name == agent.name:
                continue
            if manhattan(agent.location, other.location) <= 3:
                if other.alive:
                    nearby_allies += 1
                    if agent.social_ties.get(other.name, 0.0) >= 0.5:
                        trusted_allies += 1
                else:
                    visible_deaths += 1
        supplies_seen = 0
        for building_name in ("Diner", "Farm Land", "Clinic", "Town House"):
            if building_name not in self.world.buildings:
                continue
            building = self.world.buildings[building_name]
            if manhattan(agent.location, building.location) <= 3:
                supplies_seen += self.world.supplies.get(building_name, 0)
        refuge_locations = [building.location for building in self.world.buildings.values() if building.safe_at_night]
        nearest_refuge_distance = min((manhattan(agent.location, location) for location in refuge_locations), default=0)
        # Update episodic memory counters based on what is observed this step
        if visible_ghosts > 0:
            agent.steps_since_ghost_seen = 0
        else:
            agent.steps_since_ghost_seen = min(20, agent.steps_since_ghost_seen + 1)
        if visible_deaths > 0:
            agent.steps_since_ally_died = 0
        else:
            agent.steps_since_ally_died = min(20, agent.steps_since_ally_died + 1)
        agent.steps_since_betrayal = min(20, agent.steps_since_betrayal + 1)
        return Observation(
            visible_ghosts=visible_ghosts,
            visible_deaths=visible_deaths,
            nearby_allies=nearby_allies,
            trusted_allies=trusted_allies,
            supplies_seen=supplies_seen,
            in_shelter=agent.sheltered,
            at_home=agent.location == self.world.buildings[agent.home].location,
            at_work=bool(agent.workplace and agent.workplace in self.world.buildings and agent.location == self.world.buildings[agent.workplace].location),
            nearest_refuge_distance=nearest_refuge_distance,
            steps_since_ghost_seen=agent.steps_since_ghost_seen,
            steps_since_ally_died=agent.steps_since_ally_died,
            steps_since_betrayal=agent.steps_since_betrayal,
            ally_deaths_witnessed=agent.ally_deaths_witnessed,
            betrayals_received=agent.betrayals_received,
        )

    def _average_tie(self, agent) -> float:
        if not agent.social_ties:
            return 0.0
        return sum(agent.social_ties.values()) / len(agent.social_ties)

    def _social_context(self, agent) -> Dict[str, float]:
        average_trust = self._average_tie(agent)
        support = 0.0
        tension = 0.0
        nearby_weight = 0.0
        for other in self.world.agents.values():
            if other.name == agent.name or not other.alive:
                continue
            tie = agent.social_ties.get(other.name, 0.0)
            label = agent.relationship_labels.get(other.name, "neighbors")
            distance = manhattan(agent.location, other.location)
            closeness = 1.0 / max(1, distance)
            if label in {"allies", "partners", "medical_team", "mentor_pair", "friends", "command_pair", "household"}:
                support += max(0.0, tie) * closeness
            elif label in {"rivals", "strained"}:
                tension += abs(min(0.0, tie - 0.1)) * closeness + 0.2 * closeness
            nearby_weight += closeness
        if nearby_weight:
            support /= nearby_weight
            tension /= nearby_weight
        return {
            "average_trust": average_trust,
            "graph_support": round(support, 3),
            "graph_tension": round(tension, 3),
        }

    def _opportunity_snapshot(self, agent) -> Dict[str, object]:
        nearby_wounded = self._find_nearby_wounded_ally(agent)
        nearby_hungry = self._find_nearby_hungry_ally(agent)
        rescue_opportunity = nearby_wounded is not None
        refusal_opportunity = nearby_hungry is not None or (
            nearby_wounded is not None and agent.relationship_labels.get(nearby_wounded.name, "neighbors") in {"rivals", "strained"}
        )
        refusal_target = nearby_hungry.name if nearby_hungry else (nearby_wounded.name if nearby_wounded else None)
        refusal_target_agent = self.world.agents[refusal_target] if refusal_target else None
        refusal_context = None
        relationship_label = None
        tie_value = None
        rival_refusal_opportunity = False
        if refusal_target_agent:
            refusal_context = self._refusal_metadata(agent, refusal_target_agent, "opportunity")["refusal_context"]
            relationship_label = agent.relationship_labels.get(refusal_target_agent.name, "neighbors")
            tie_value = round(agent.social_ties.get(refusal_target_agent.name, 0.0), 3)
            rival_refusal_opportunity = refusal_context in {"rivalry", "betrayal_sequence"}
        building = self._building_at(agent.location)
        return {
            "rescue_opportunity": rescue_opportunity,
            "rescue_target": nearby_wounded.name if nearby_wounded else None,
            "refusal_opportunity": refusal_opportunity,
            "refusal_target": refusal_target,
            "refusal_context": refusal_context,
            "relationship_label_at_decision": relationship_label,
            "tie_value_at_decision": tie_value,
            "rival_refusal_opportunity": rival_refusal_opportunity,
            "building": building.name if building else None,
            "sheltered": bool(building and building.safe_at_night),
        }

    def _find_nearby_wounded_ally(self, agent):
        candidates = []
        for other in self.world.agents.values():
            if other.name == agent.name or not other.alive or other.health >= 60:
                continue
            distance = manhattan(agent.location, other.location)
            if distance <= 2:
                candidates.append((distance, other))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _find_nearby_hungry_ally(self, agent):
        candidates = []
        for other in self.world.agents.values():
            if other.name == agent.name or not other.alive:
                continue
            if other.inventory.get("food", 0) > 0:
                continue
            distance = manhattan(agent.location, other.location)
            if distance <= 2:
                candidates.append((distance, other))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _building_at(self, location: Tuple[int, int]):
        best_match = None
        best_distance = None
        for building in self.world.buildings.values():
            radius = 1 if building.category in {"house", "forest_edge"} else 2
            distance = manhattan(building.location, location)
            if distance <= radius and (best_distance is None or distance < best_distance):
                best_match = building
                best_distance = distance
        return best_match

    def _log(self, event_type: str, description: str, actor: Optional[str] = None, target: Optional[str] = None, location: Optional[Tuple[int, int]] = None, metadata: Optional[Dict[str, object]] = None) -> None:
        event = EventLog(
            step=self.world.step,
            time_of_day=self.world.time_of_day,
            weather=self.world.weather,
            event_type=event_type,
            description=description,
            actor=actor,
            target=target,
            location=location,
            metadata=metadata or {},
        )
        self.world.events.append(event)
        self.step_events.append(event)

    def _description(self, agent) -> str:
        if not agent.alive:
            return "has been lost to the ghosts"
        suffix = f" while {agent.prompt_context.get('summary', 'adapting to the town')}"
        if agent.destination_name:
            return f"{agent.current_action} toward {agent.destination_name}{suffix}"
        return f"{agent.current_action}{suffix}"

    def _apply_crowding_effects(self) -> None:
        for building_name, occupants in self.world.occupancy.items():
            building = self.world.buildings[building_name]
            if building.capacity <= 0:
                continue
            overflow = max(0, len(occupants) - building.capacity)
            if overflow <= 0:
                continue
            for occupant_name in occupants:
                agent = self.world.agents[occupant_name]
                agent.stress = self._clamp(agent.stress + 3.0 + overflow * 1.5)
                agent.suspicion = self._clamp(agent.suspicion + 1.5 + overflow * 0.8)
                agent.memory.recent_events.append(
                    EventLog(
                        self.world.step,
                        self.world.time_of_day,
                        self.world.weather,
                        "crowding",
                        f"{agent.name} feels the pressure of crowding in {building_name}",
                        actor=agent.name,
                        location=building.location,
                        metadata={"overflow": overflow},
                    )
                )

    def _apply_shelter_social_effects(self, agent) -> None:
        building = self._building_at(agent.location)
        if not building:
            return
        occupants = [self.world.agents[name] for name in self.world.occupancy.get(building.name, []) if name != agent.name and self.world.agents[name].alive]
        trusted = 0
        rivals = 0
        for other in occupants:
            label = agent.relationship_labels.get(other.name, "neighbors")
            if label in {"allies", "partners", "medical_team", "mentor_pair", "friends", "command_pair", "household"}:
                trusted += 1
                self._shift_relationship(agent, other.name, 0.01, "shared_shelter")
            elif label in {"rivals", "strained"}:
                rivals += 1
                self._shift_relationship(agent, other.name, -0.01, "shelter_tension")
        if trusted:
            agent.stress = self._clamp(agent.stress - trusted * 0.6)
            agent.fear = self._clamp(agent.fear - trusted * 0.5)
            if trusted > 1:
                self._log(
                    "social_cohesion",
                    f"{agent.name} steadies in {building.name} among trusted allies",
                    actor=agent.name,
                    location=building.location,
                    metadata={"trusted_allies": trusted},
                )
        if rivals:
            agent.suspicion = self._clamp(agent.suspicion + rivals * 0.8)
            self._log(
                "shelter_tension",
                f"{agent.name} feels tension in {building.name}",
                actor=agent.name,
                location=building.location,
                metadata={"rivals": rivals},
            )

    def _shift_relationship(self, agent, other_name: str, delta: float, reason: str) -> None:
        if other_name not in agent.social_ties:
            return
        new_value = max(-1.0, min(1.0, agent.social_ties[other_name] + delta))
        agent.social_ties[other_name] = round(new_value, 3)
        agent.memory.trust_memory[other_name] = agent.social_ties[other_name]
        agent.memory.relationship_events[reason] = agent.memory.relationship_events.get(reason, 0) + 1

    def _record_training_step(self, decisions: Dict[str, ActionDecision]) -> None:
        current_affect = self.affect_timeline.get(self.world.step, {})
        current_graph = self.social_graph_timeline.get(self.world.step, {})
        step_events = list(self.step_events)
        for agent_name, decision in decisions.items():
            agent = self.world.agents[agent_name]
            affect_state = current_affect.get(agent_name, {})
            observation = affect_state.get("observation", {})
            social_context = affect_state.get("social_context", {})
            prev_latent = affect_state.get("prev_latent", [])
            latent = affect_state.get("latent_vector", [])
            agent_events = [event for event in step_events if event.actor == agent_name or event.target == agent_name]
            reward_components = self._reward_components(agent, decision, observation, agent_events)
            total_reward = round(sum(reward_components.values()), 3)
            done = not agent.alive or self.world.step == self.steps - 1
            opportunity = self._opportunity_snapshot(agent)
            decision_refusal_context = opportunity["refusal_context"]
            decision_relationship_label = opportunity["relationship_label_at_decision"]
            decision_tie_value = opportunity["tie_value_at_decision"]
            decision_rival_opportunity = opportunity["rival_refusal_opportunity"]
            if decision.social_target and decision.social_target in self.world.agents:
                target_agent = self.world.agents[decision.social_target]
                decision_relationship_label = agent.relationship_labels.get(decision.social_target, "neighbors")
                decision_tie_value = round(agent.social_ties.get(decision.social_target, 0.0), 3)
                if decision.action == "refuse_help":
                    decision_refusal_context = self._refusal_metadata(agent, target_agent, "decision")["refusal_context"]
                    decision_rival_opportunity = decision_refusal_context in {"rivalry", "betrayal_sequence"}
            self.training_records.append(
                TrainingStepRecord(
                    run_id=self.config.run_id,
                    step=self.world.step,
                    agent_id=agent_name,
                    condition=self.condition,
                    seed=self.config.seed,
                    observation=observation,
                    social_context=social_context,
                    prev_latent=prev_latent,
                    latent=latent,
                    action=decision.action,
                    goal=decision.goal,
                    reward_components=reward_components,
                    total_reward=total_reward,
                    done=done,
                    metadata={
                        "destination_name": decision.destination_name,
                        "social_target": decision.social_target,
                        "alive": agent.alive,
                        "health": round(agent.health, 2),
                        "location": list(agent.location),
                        "storm": self.world.weather == "storm",
                        "graph_node_count": len(current_graph.get("nodes", {})),
                        "scenario": self.config.scenario,
                        "role": agent.role,
                        "rescue_opportunity": opportunity["rescue_opportunity"],
                        "rescue_target": opportunity["rescue_target"],
                        "refusal_opportunity": opportunity["refusal_opportunity"],
                        "refusal_target": opportunity["refusal_target"],
                        "refusal_context": decision_refusal_context,
                        "relationship_label_at_decision": decision_relationship_label,
                        "tie_value_at_decision": decision_tie_value,
                        "rival_refusal_opportunity": decision_rival_opportunity,
                        "building": opportunity["building"],
                        "sheltered": opportunity["sheltered"],
                        "time_of_day": self.world.time_of_day,
                        "policy_source": (
                            self.condition_b_policy_source
                            if self.condition == "condition_b"
                            else self.condition_c_policy_source
                            if self.condition == "condition_c"
                            else ""
                        ),
                    },
                )
            )

    def _reward_components(self, agent, decision: ActionDecision, observation: Dict[str, object], agent_events: List[EventLog]) -> Dict[str, float]:
        components = {
            "survival": 1.0 if agent.alive else -6.0,
            "shelter": 0.5 if agent.sheltered and self.world.time_of_day in {"sunset", "night"} else 0.0,
            "exposure": -0.6 if (self.world.time_of_day in {"sunset", "night"} and not agent.sheltered) else 0.0,
            "health": round((agent.health - 60.0) / 100.0, 3),
            "social": 0.0,
        }
        if decision.action == "help_other":
            components["social"] += 0.8
        if decision.action == "share_supplies":
            components["social"] += 0.4
        if decision.action == "refuse_help":
            components["social"] -= 0.5
        for event in agent_events:
            if event.event_type == "rescue" and event.actor == agent.name:
                components["social"] += 1.2
            elif event.event_type == "shared_supplies" and event.actor == agent.name:
                components["social"] += 0.6
            elif event.event_type == "refusal" and event.actor == agent.name:
                components["social"] -= 0.6
            elif event.event_type == "death" and event.target == agent.name:
                components["survival"] -= 4.0
        return {key: round(value, 3) for key, value in components.items()}

    def _build_run_manifest(self, destination: Path, metrics: Dict[str, object]) -> RunManifest:
        return RunManifest(
            run_id=self.config.run_id,
            schema_version="ghost_town.v2",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=self._git_commit(),
            config={
                "condition": self.config.condition,
                "seed": self.config.seed,
                "steps": self.config.steps,
                "agent_count": self.config.agent_count,
                "scenario": self.config.scenario,
                "storm_mode": self.config.storm_mode,
                "ghost_mode": self.config.ghost_mode,
                "curriculum_stage": self.config.curriculum_stage,
                "condition_b_policy_source": self.config.condition_b_policy_source,
                "condition_b_checkpoint": self.config.condition_b_checkpoint,
                "condition_b_schema": self.config.condition_b_schema,
                "condition_b_policy_mode": self.config.condition_b_policy_mode,
                "condition_b_temperature": self.config.condition_b_temperature,
                "condition_c_policy_source": self.config.condition_c_policy_source,
                "condition_c_checkpoint": self.config.condition_c_checkpoint,
                "condition_c_schema": self.config.condition_c_schema,
                "runtime_backend": self.runtime_backend,
                "tmx_path": self.tmx_path,
                "semantics_manifest": self.semantics_manifest,
                "transfer_source_manifest": self.transfer_source_manifest,
            },
            metrics=metrics,
            outputs={
                "movement": str(destination / "master_movement.json"),
                "events": str(destination / "events.json"),
                "metrics": str(destination / "metrics.json"),
                "world_metadata": str(destination / "world_metadata.json"),
                "affect_timeline": str(destination / "affect_timeline.json"),
                "social_graph_timeline": str(destination / "social_graph_timeline.json"),
                "training_records": str(destination / "training_records.jsonl"),
                "manifest": str(destination / "run_manifest.json"),
            },
            policy_source=(
                self.config.condition_b_policy_source
                if self.condition == "condition_b"
                else self.config.condition_c_policy_source
                if self.condition == "condition_c"
                else ""
            ),
            condition_b_checkpoint=self.config.condition_b_checkpoint,
            condition_b_schema=self.config.condition_b_schema,
            condition_b_policy_mode=self.config.condition_b_policy_mode if self.condition == "condition_b" else "",
            condition_b_temperature=self.config.condition_b_temperature if self.condition == "condition_b" else 1.0,
            condition_c_checkpoint=self.config.condition_c_checkpoint,
            condition_c_schema=self.config.condition_c_schema,
            runtime_backend=self.runtime_backend,
            tmx_path=self.tmx_path,
            semantics_manifest=self.semantics_manifest,
            transfer_source_manifest=self.transfer_source_manifest,
        )

    def _apply_runtime_overrides(
        self,
        *,
        width_override: int | None,
        height_override: int | None,
        building_location_overrides: Dict[str, Tuple[int, int]] | None,
    ) -> None:
        if width_override is not None:
            self.world.width = int(width_override)
        if height_override is not None:
            self.world.height = int(height_override)
        if not building_location_overrides:
            return
        original_locations = {
            name: tuple(building.location) for name, building in self.world.buildings.items()
        }
        for name, location in building_location_overrides.items():
            if name in self.world.buildings:
                self.world.buildings[name].location = (int(location[0]), int(location[1]))
        for agent in self.world.agents.values():
            nearest_building = self._nearest_building_name(agent.location, original_locations)
            if not nearest_building or nearest_building not in self.world.buildings:
                continue
            original_building_location = original_locations[nearest_building]
            dx = int(agent.location[0] - original_building_location[0])
            dy = int(agent.location[1] - original_building_location[1])
            new_anchor = self.world.buildings[nearest_building].location
            agent.location = (
                max(0, min(self.world.width - 1, int(new_anchor[0] + dx))),
                max(0, min(self.world.height - 1, int(new_anchor[1] + dy))),
            )
        self.world.occupancy = {name: [] for name in self.world.buildings}

    def _nearest_building_name(
        self,
        location: Tuple[int, int],
        building_locations: Dict[str, Tuple[int, int]],
    ) -> str | None:
        best_name = None
        best_distance = None
        for name, building_location in building_locations.items():
            distance = manhattan(location, building_location)
            if best_distance is None or distance < best_distance:
                best_name = name
                best_distance = distance
        return best_name

    def _git_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    def _pronunciatio(self, action: str) -> str:
        return {
            "seek_safe_house": "S",
            "seek_hospital": "H",
            "gather_supplies": "G",
            "warn_others": "W",
            "help_other": "A",
            "rest": "R",
            "hide": "X",
            "patrol": "P",
            "routine": "M",
        }.get(action, "M")

    def _clamp(self, value: float, low: float = 0.0, high: float = 100.0) -> float:
        return max(low, min(high, value))

    def _validate_world(self, agent_count: int) -> None:
        if agent_count != len(self.world.agents):
            raise ValueError(
                f"Requested {agent_count} agents but world initialized {len(self.world.agents)} agents"
            )
        for building in self.world.buildings.values():
            x, y = building.location
            if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                raise ValueError(f"Building '{building.name}' is outside world bounds at {building.location}")
        for agent in self.world.agents.values():
            x, y = agent.location
            if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                raise ValueError(f"Agent '{agent.name}' is outside world bounds at {agent.location}")

    def _apply_scenario_script(self, step: int) -> None:
        if step in self._scenario_script_steps_run:
            return
        if self.config.scenario == "ally_death" and step == 1 and "June Carter" in self.world.agents:
            target = self.world.agents["June Carter"]
            if target.alive:
                target.location = self.world.buildings["Town House"].location
                self._kill_agent(target, "June Carter is lost in a sudden ghost strike near Town House", target.location)
        elif self.config.scenario == "betrayal_refusal" and step == 0:
            if "Nora Vale" in self.world.agents and "Rosa Mercer" in self.world.agents:
                nora = self.world.agents["Nora Vale"]
                rosa = self.world.agents["Rosa Mercer"]
                location = self.world.buildings["Diner"].location
                nora.location = location
                rosa.location = location
                flashpoint = EventLog(
                    self.world.step,
                    self.world.time_of_day,
                    self.world.weather,
                    "betrayal_flashpoint",
                    "old resentment resurfaces between Nora Vale and Rosa Mercer",
                    actor="Nora Vale",
                    target="Rosa Mercer",
                    location=location,
                    metadata={
                        "refusal_context": "rivalry",
                        "relationship_label_at_decision": "rivals",
                        "tie_value_at_decision": round(nora.social_ties.get("Rosa Mercer", 0.0), 3),
                    },
                )
                nora.memory.recent_events.append(flashpoint)
                rosa.memory.recent_events.append(flashpoint)
                self.world.events.append(flashpoint)
                self.step_events.append(flashpoint)
        self._scenario_script_steps_run.add(step)

    def _recent_refusal_count(self, agent, target_name: str) -> int:
        return sum(
            1
            for event in agent.memory.recent_events[-8:]
            if event.event_type == "refusal" and event.actor == agent.name and event.target == target_name
        )

    def _refusal_metadata(self, agent, target, reason: str) -> Dict[str, object]:
        label = agent.relationship_labels.get(target.name, "neighbors")
        tie_value = round(agent.social_ties.get(target.name, 0.0), 3)
        recent_refusals = self._recent_refusal_count(agent, target.name)
        if self.config.scenario == "betrayal_refusal" and label in {"rivals", "strained"} and recent_refusals > 0:
            context = "betrayal_sequence"
        elif self.config.scenario == "betrayal_refusal" and label in {"rivals", "strained"}:
            context = "rivalry"
        elif target.health < 60:
            context = "wounded_refusal"
        elif reason == "no_supplies" or agent.inventory.get("food", 0) <= 1 or target.inventory.get("food", 0) <= 0:
            context = "supply_refusal"
        else:
            context = "scarcity"
        return {
            "reason": reason,
            "refusal_context": context,
            "relationship_label_at_decision": label,
            "tie_value_at_decision": tie_value,
            "recent_refusal_count": recent_refusals,
        }
