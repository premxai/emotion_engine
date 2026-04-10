from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from pathlib import Path
from typing import Dict, List, Optional

from .types import AffectOutput, AgentState, WorldState


@dataclass
class Observation:
    visible_ghosts: int
    visible_deaths: int
    nearby_allies: int
    trusted_allies: int
    supplies_seen: int
    in_shelter: bool
    at_home: bool
    at_work: bool
    nearest_refuge_distance: int
    # Episodic memory — steps since key events (capped at 20; 20 = never seen)
    steps_since_ghost_seen: int = 20
    steps_since_ally_died: int = 20
    steps_since_betrayal: int = 20
    ally_deaths_witnessed: int = 0
    betrayals_received: int = 0


class EmotionEngine:
    mode = "baseline"

    def compute_internal_state(
        self,
        agent_state: AgentState,
        observation: Observation,
        social_context: Dict[str, float],
        world_state: WorldState,
    ) -> AffectOutput:
        raise NotImplementedError


class NeutralEmotionEngine(EmotionEngine):
    mode = "baseline_0"

    def compute_internal_state(self, agent_state, observation, social_context, world_state):
        return AffectOutput(
            label="neutral",
            affect_vector={"fear": 0.0, "stress": 0.0, "trust": 0.0, "grief": 0.0, "suspicion": 0.0, "relief": 0.0},
            action_bias={},
            prompt_context={"tone": "measured", "summary": "staying focused on survival"},
            latent_vector=[0.0] * 8,
            probe_data={},
        )


class ProgrammedEmotionEngine(EmotionEngine):
    mode = "condition_a"

    def compute_internal_state(self, agent_state, observation, social_context, world_state):
        courage = agent_state.traits.get("courage", 0.5)
        empathy = agent_state.traits.get("empathy", 0.5)
        baseline_suspicion = agent_state.traits.get("suspicion", 0.4)
        resilience = agent_state.traits.get("resilience", 0.5)
        graph_tension = social_context.get("graph_tension", 0.0)
        graph_support = social_context.get("graph_support", 0.0)
        total_supplies = sum(world_state.supplies.values())
        scarcity = max(0.0, min(1.0, (42.0 - total_supplies) / 38.0))
        personal_scarcity = 0.12 if agent_state.inventory.get("food", 0) <= 1 else 0.0
        medicine_scarcity = 0.05 if agent_state.inventory.get("medicine", 0) <= 0 and world_state.supplies.get("Clinic", 0) <= 2 else 0.0
        scenario_hostility = 0.12 if world_state.scenario == "betrayal_refusal" else 0.0
        betrayal_memory = 0.0
        if world_state.scenario == "betrayal_refusal":
            betrayal_memory = min(
                0.22,
                0.05
                * sum(
                    1
                    for event in agent_state.memory.recent_events[-6:]
                    if event.event_type in {"betrayal_flashpoint", "refusal"}
                    and (event.actor == agent_state.name or event.target == agent_state.name)
                ),
            )
        crowding = 0.0
        for building_name, occupants in world_state.occupancy.items():
            building = world_state.buildings[building_name]
            if agent_state.name in occupants and building.capacity > 0:
                crowding = max(0.0, (len(occupants) - building.capacity) / building.capacity)
                break
        hazard = 0.0
        if world_state.time_of_day in {"sunset", "night"}:
            hazard += 0.35
        hazard += min(0.5, observation.visible_ghosts * 0.25)
        hazard += world_state.storm_severity * 0.2
        if not observation.in_shelter:
            hazard += 0.15
        hazard += scarcity * 0.04
        grief = min(1.0, agent_state.grief / 100.0 + observation.visible_deaths * (0.12 + empathy * 0.2))
        fear = min(
            1.0,
            hazard
            + grief * 0.1
            + crowding * 0.08
            + max(0.0, (40.0 - agent_state.health) / 100.0)
            + max(0.0, (3 - observation.nearest_refuge_distance) * 0.03)
            - courage * 0.18
            - resilience * 0.08,
        )
        stress = min(
            1.0,
            fear * 0.65
            + world_state.storm_severity * 0.3
            + scarcity * 0.12
            + crowding * 0.12
            + max(0.0, (50.0 - agent_state.stamina) / 100.0)
            - resilience * 0.12,
        )
        trust = max(
            0.0,
            min(
                1.0,
                social_context.get("average_trust", 0.0)
                + observation.trusted_allies * (0.04 + empathy * 0.05)
                + (0.06 if observation.at_home or observation.at_work else 0.0)
                + graph_support * 0.16
                - grief * 0.08
                - baseline_suspicion * 0.08
                - graph_tension * 0.22
                - scarcity * 0.08
                - personal_scarcity * 0.7
                - medicine_scarcity * 0.6
                - scenario_hostility
                - betrayal_memory,
            ),
        )
        suspicion = min(
            1.0,
            max(
                0.0,
                baseline_suspicion
                + 0.1
                - trust
                + grief * 0.18
                + stress * 0.08
                + graph_tension * 0.22
                + scarcity * 0.08
                + personal_scarcity * 0.4
                + medicine_scarcity * 0.2
                + scenario_hostility
                + betrayal_memory
                - empathy * 0.1,
            ),
        )
        relief = max(0.0, 0.65 if observation.in_shelter and observation.visible_ghosts == 0 else 0.12)
        seek_safe_house = fear * 1.25 + stress + world_state.storm_severity * 0.2 + crowding * 0.1 - courage * 0.12
        help_bias = trust + empathy * 0.2 - fear * 0.25 - suspicion * 0.35 - scarcity * 0.32 - graph_tension * 0.25
        warn_bias = fear + trust * 0.35 + courage * 0.15 - scarcity * 0.04
        gather_bias = max(0.0, 0.35 + scarcity * 0.7 - fear * 0.3 - world_state.storm_severity * 0.22 - crowding * 0.08 + resilience * 0.18)
        return AffectOutput(
            label="programmed",
            affect_vector={
                "fear": round(fear, 3),
                "stress": round(stress, 3),
                "trust": round(trust, 3),
                "grief": round(grief, 3),
                "suspicion": round(suspicion, 3),
                "relief": round(relief, 3),
            },
            action_bias={
                "seek_safe_house": round(max(0.0, seek_safe_house), 3),
                "seek_hospital": max(0.0, (60.0 - agent_state.health) / 50.0),
                "help": round(max(-1.0, help_bias), 3),
                "warn": round(max(0.0, warn_bias), 3),
                "gather": round(gather_bias, 3),
            },
            prompt_context={
                "tone": "anxious" if fear > 0.55 else "steady",
                "summary": f"fear={fear:.2f}, stress={stress:.2f}, trust={trust:.2f}, suspicion={suspicion:.2f}",
            },
            latent_vector=[fear, stress, trust, grief, suspicion, relief, hazard, float(observation.nearby_allies) / 5.0],
            probe_data={
                "appraisal": {
                    "hazard": round(hazard, 3),
                    "courage": round(courage, 3),
                    "empathy": round(empathy, 3),
                    "resilience": round(resilience, 3),
                    "baseline_suspicion": round(baseline_suspicion, 3),
                    "graph_tension": round(graph_tension, 3),
                    "graph_support": round(graph_support, 3),
                    "scarcity": round(scarcity, 3),
                    "crowding": round(crowding, 3),
                    "betrayal_memory": round(betrayal_memory, 3),
                }
            },
        )


class EmergentEmotionEngine(EmotionEngine):
    mode = "condition_b"

    def __init__(self, inference_backend: Optional[object] = None):
        self.inference_backend = inference_backend

    def _latent(self, agent_state: AgentState, observation: Observation, social_context: Dict[str, float], world_state: WorldState) -> List[float]:
        courage = agent_state.traits.get("courage", 0.5)
        empathy = agent_state.traits.get("empathy", 0.5)
        resilience = agent_state.traits.get("resilience", 0.5)
        suspicion_trait = agent_state.traits.get("suspicion", 0.4)
        weather_term = 0.4 if world_state.weather == "storm" else -0.1
        shelter_term = -0.3 if observation.in_shelter else 0.25
        health_term = (50.0 - agent_state.health) / 50.0
        grief_term = agent_state.grief / 80.0
        ties_term = social_context.get("graph_support", 0.0)
        graph_tension = social_context.get("graph_tension", 0.0)
        previous = list(agent_state.affect_embedding) if agent_state.affect_embedding else [0.0] * 8
        proposal = [
            tanh(observation.visible_ghosts * 0.7 + shelter_term - courage * 0.25),
            tanh(weather_term + observation.visible_deaths * 0.5 + max(0, 3 - observation.nearest_refuge_distance) * 0.08),
            tanh(health_term - resilience * 0.2),
            tanh((50.0 - agent_state.stamina) / 40.0 - resilience * 0.15),
            tanh(ties_term + observation.trusted_allies * 0.18 + empathy * 0.2 - graph_tension * 0.2),
            tanh(grief_term),
            tanh((2 - agent_state.inventory.get("food", 0)) * 0.4 + world_state.storm_severity - resilience * 0.1),
            tanh(agent_state.suspicion / 50.0 + suspicion_trait * 0.5 + graph_tension * 0.3 + (0.2 if world_state.time_of_day == "night" else -0.1)),
        ]
        return [tanh(previous[idx] * 0.55 + proposal[idx] * 0.9) for idx in range(8)]

    def compute_internal_state(self, agent_state, observation, social_context, world_state):
        if self.inference_backend is not None:
            return self.inference_backend.compute(
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
                    # Cross-night episodic memory — lets model learn temporal adaptation
                    "steps_since_ghost_seen": agent_state.steps_since_ghost_seen,
                    "steps_since_ally_died": agent_state.steps_since_ally_died,
                    "ally_deaths_witnessed": agent_state.ally_deaths_witnessed,
                    "nights_survived": agent_state.nights_survived,
                    "last_night_ghost_seen": float(agent_state.last_night_ghost_seen),
                    "last_night_death_nearby": float(agent_state.last_night_death_nearby),
                },
                social_context=social_context,
                prev_latent=agent_state.affect_embedding,
                metadata={
                    "health": agent_state.health,
                    "storm": world_state.weather == "storm",
                    "alive": agent_state.alive,
                    "sheltered": agent_state.sheltered,
                    "rescue_opportunity": False,
                    "refusal_opportunity": False,
                    "rival_refusal_opportunity": False,
                    "scenario": world_state.scenario,
                    "role": agent_state.role,
                    "refusal_context": "scarcity",
                    "relationship_label_at_decision": "neighbors",
                    "tie_value_at_decision": 0.0,
                    "time_of_day": world_state.time_of_day,
                },
            )
        latent = self._latent(agent_state, observation, social_context, world_state)
        fear = max(0.0, latent[0] * 0.6 + latent[1] * 0.2 + latent[7] * 0.2)
        stress = max(0.0, latent[1] * 0.4 + latent[2] * 0.2 + latent[3] * 0.2 + latent[6] * 0.2)
        trust = max(0.0, latent[4] * 0.7 - latent[7] * 0.2)
        grief = max(0.0, latent[5])
        suspicion = max(0.0, latent[7] * 0.8 - latent[4] * 0.2)
        relief = max(0.0, -latent[0] * 0.4 - latent[1] * 0.2 + (0.4 if observation.in_shelter else 0.0))
        no_death_probe = self._latent(
            agent_state,
            Observation(
                visible_ghosts=observation.visible_ghosts,
                visible_deaths=0,
                nearby_allies=observation.nearby_allies,
                trusted_allies=observation.trusted_allies,
                supplies_seen=observation.supplies_seen,
                in_shelter=observation.in_shelter,
                at_home=observation.at_home,
                at_work=observation.at_work,
                nearest_refuge_distance=observation.nearest_refuge_distance,
            ),
            social_context,
            world_state,
        )
        shelter_probe = self._latent(
            agent_state,
            Observation(
                visible_ghosts=observation.visible_ghosts,
                visible_deaths=observation.visible_deaths,
                nearby_allies=observation.nearby_allies,
                trusted_allies=observation.trusted_allies,
                supplies_seen=observation.supplies_seen,
                in_shelter=True,
                at_home=observation.at_home,
                at_work=observation.at_work,
                nearest_refuge_distance=0,
            ),
            social_context,
            world_state,
        )
        counterfactuals = {
            "no_death": {
                "latent": [round(value, 3) for value in no_death_probe],
                "grief_delta": round(grief - max(0.0, no_death_probe[5]), 3),
            },
            "with_shelter": {
                "latent": [round(value, 3) for value in shelter_probe],
                "fear_delta": round(fear - max(0.0, shelter_probe[0] * 0.6 + shelter_probe[1] * 0.2 + shelter_probe[7] * 0.2), 3),
                "stress_delta": round(stress - max(0.0, shelter_probe[1] * 0.4 + shelter_probe[2] * 0.2 + shelter_probe[3] * 0.2 + shelter_probe[6] * 0.2), 3),
            },
        }
        return AffectOutput(
            label="emergent",
            affect_vector={
                "fear": round(fear, 3),
                "stress": round(stress, 3),
                "trust": round(trust, 3),
                "grief": round(grief, 3),
                "suspicion": round(suspicion, 3),
                "relief": round(relief, 3),
            },
            action_bias={
                "seek_safe_house": fear + stress * 0.8,
                "seek_hospital": max(0.0, latent[2]),
                "help": trust - stress * 0.2,
                "warn": fear + trust * 0.2,
                "gather": max(0.0, 0.6 - fear + latent[6] * 0.3),
            },
            prompt_context={
                "tone": "urgent" if fear > 0.5 else "watchful",
                "summary": (
                    "graph-latent affect "
                    f"z0={latent[0]:.2f}, z4={latent[4]:.2f}, "
                    f"probe_no_death={no_death_probe[5]:.2f}, probe_shelter={shelter_probe[0]:.2f}"
                ),
            },
            latent_vector=[round(value, 3) for value in latent],
            probe_data={
                "social_context": {key: round(value, 3) for key, value in social_context.items()},
                "counterfactuals": counterfactuals,
            },
        )


class HybridEmotionEngine(EmotionEngine):
    mode = "condition_c"

    def __init__(self):
        self.programmed = ProgrammedEmotionEngine()
        self.emergent = EmergentEmotionEngine()

    def compute_internal_state(self, agent_state, observation, social_context, world_state):
        programmed = self.programmed.compute_internal_state(agent_state, observation, social_context, world_state)
        emergent = self.emergent.compute_internal_state(agent_state, observation, social_context, world_state)
        keys = programmed.affect_vector.keys()
        affect = {}
        for key in keys:
            affect[key] = round(programmed.affect_vector[key] * 0.6 + emergent.affect_vector[key] * 0.4, 3)
        action_bias = {}
        for key in set(programmed.action_bias) | set(emergent.action_bias):
            action_bias[key] = round(programmed.action_bias.get(key, 0.0) * 0.6 + emergent.action_bias.get(key, 0.0) * 0.4, 3)
        latent = [round(programmed.latent_vector[idx] * 0.6 + emergent.latent_vector[idx] * 0.4, 3) for idx in range(8)]
        return AffectOutput(
            label="hybrid",
            affect_vector=affect,
            action_bias=action_bias,
            prompt_context={
                "tone": "strained" if affect["stress"] > 0.45 else "guarded",
                "summary": "hybrid appraisal plus latent adaptation",
            },
            latent_vector=latent,
            probe_data={
                "blend": {"programmed_weight": 0.6, "emergent_weight": 0.4},
                "programmed": programmed.probe_data,
                "emergent": emergent.probe_data,
            },
        )


class SelfSupervisedConditionCEngine(EmotionEngine):
    mode = "condition_c"

    def __init__(self, inference_backend: Optional[object] = None):
        self.inference_backend = inference_backend
        self.fallback = HybridEmotionEngine()

    def compute_internal_state(self, agent_state, observation, social_context, world_state):
        if self.inference_backend is None:
            return self.fallback.compute_internal_state(agent_state, observation, social_context, world_state)
        return self.inference_backend.compute(
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
            },
            social_context=social_context,
            prev_latent=agent_state.affect_embedding,
            metadata={
                "health": agent_state.health,
                "storm": world_state.weather == "storm",
                "alive": agent_state.alive,
                "sheltered": agent_state.sheltered,
                "rescue_opportunity": False,
                "refusal_opportunity": False,
                "rival_refusal_opportunity": False,
                "scenario": world_state.scenario,
                "role": agent_state.role,
                "refusal_context": "scarcity",
                "relationship_label_at_decision": "neighbors",
                "tie_value_at_decision": 0.0,
                "time_of_day": world_state.time_of_day,
            },
        )


def build_condition_b_engine(
    checkpoint_path: str = "",
    schema_path: str = "",
    policy_mode: str = "deterministic",
    temperature: float = 1.0,
    rng_seed: int | None = None,
) -> EmergentEmotionEngine:
    if not checkpoint_path or not schema_path:
        return EmergentEmotionEngine()
    from ghost_town_training.inference import ConditionBInferenceBackend

    backend = ConditionBInferenceBackend(
        Path(checkpoint_path),
        Path(schema_path),
        policy_mode=policy_mode,
        temperature=temperature,
        rng_seed=rng_seed,
    )
    return EmergentEmotionEngine(inference_backend=backend)


def build_condition_c_engine(
    checkpoint_path: str = "",
    schema_path: str = "",
) -> EmotionEngine:
    if not checkpoint_path or not schema_path:
        return SelfSupervisedConditionCEngine()
    from ghost_town_training.inference_c import ConditionCInferenceBackend

    backend = ConditionCInferenceBackend(Path(checkpoint_path), Path(schema_path))
    return SelfSupervisedConditionCEngine(inference_backend=backend)


class PredictiveEmotionEngine(EmotionEngine):
    """Condition D — emotions emerge from predicting future world events.

    The affect_vector is derived from 12 prediction head outputs rather than
    hand-coded formulas or imitation. A dimension earns its emotion label only
    by successfully predicting the corresponding future event.

    Falls back to EmergentEmotionEngine scaffold when no checkpoint is loaded.
    """

    mode = "condition_d"

    def __init__(self, inference_backend=None):
        self.inference_backend = inference_backend
        self._fallback = EmergentEmotionEngine()

    def compute_internal_state(self, agent_state, observation, social_context, world_state):
        if self.inference_backend is None:
            return self._fallback.compute_internal_state(agent_state, observation, social_context, world_state)

        metadata = {
            "health": float(getattr(agent_state, "health", 100.0)),
            "storm": bool(getattr(world_state, "storm_active", False)),
            "alive": bool(getattr(agent_state, "alive", True)),
            "sheltered": bool(getattr(agent_state, "sheltered", False)),
            "rescue_opportunity": bool(getattr(agent_state, "rescue_opportunity", False)),
            "refusal_opportunity": bool(getattr(agent_state, "refusal_opportunity", False)),
            "rival_refusal_opportunity": bool(getattr(agent_state, "rival_refusal_opportunity", False)),
            "tie_value_at_decision": float(getattr(agent_state, "tie_value_at_decision", 0.0)),
            "scenario": str(getattr(world_state, "scenario", "standard_night")),
            "role": str(getattr(agent_state, "role", "")),
            "relationship_label_at_decision": str(getattr(agent_state, "relationship_label_at_decision", "")),
            "refusal_context": str(getattr(agent_state, "refusal_context", "")),
            "time_of_day": str(getattr(world_state, "time_of_day", "day")),
        }
        # Use zeros for prev_latent — model was trained with prev_latent=0 (scaffold fallback
        # during data collection), so feeding back actual latent causes exponential explosion.
        prev_latent = []
        # Convert Observation dataclass to dict if needed
        if isinstance(observation, dict):
            obs_dict = observation
        else:
            import dataclasses
            obs_dict = dataclasses.asdict(observation) if dataclasses.is_dataclass(observation) else {}
        social_dict = social_context if isinstance(social_context, dict) else {}

        return self.inference_backend.compute(obs_dict, social_dict, prev_latent, metadata)


def build_condition_d_engine(
    checkpoint_path: str = "",
    schema_path: str = "",
) -> PredictiveEmotionEngine:
    if not checkpoint_path or not schema_path:
        return PredictiveEmotionEngine()
    from ghost_town_training.inference_d import ConditionDInferenceBackend

    backend = ConditionDInferenceBackend(Path(checkpoint_path), Path(schema_path))
    return PredictiveEmotionEngine(inference_backend=backend)
