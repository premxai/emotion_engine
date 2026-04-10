from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .emotions import Observation
from .pathfinding import find_path
from .types import ActionDecision, AgentState, Building, Coordinate, WorldState


def manhattan(a: Coordinate, b: Coordinate) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(start: Coordinate, end: Coordinate, world: WorldState) -> Coordinate:
    """Advance one tile toward end using the agent's cached BFS path (world.nav_paths).

    Falls back to straight-line if the cache is empty.
    """
    if start == end:
        return start
    # Straight-line fallback (used when nav_path cache is bypassed)
    x, y = start
    dx = 0 if start[0] == end[0] else (1 if end[0] > start[0] else -1)
    dy = 0 if start[1] == end[1] else (1 if end[1] > start[1] else -1)
    candidates = []
    if dx:
        candidates.append((x + dx, y))
    if dy:
        candidates.append((x, y + dy))
    for candidate in candidates:
        if 0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height:
            return candidate
    return start


class ActionResolver:
    def resolve(
        self,
        agent: AgentState,
        observation: Observation,
        affect: dict,
        action_bias: dict,
        world: WorldState,
    ) -> ActionDecision:
        if not agent.alive:
            return ActionDecision("dead", "dead", agent.location, None, "is gone", "D")

        clinic_name = "Clinic" if "Clinic" in world.buildings else "Hospital"
        town_refuge_name = "Town House" if "Town House" in world.buildings else "Safe House"
        courage = agent.traits.get("courage", 0.5)
        empathy = agent.traits.get("empathy", 0.5)
        resilience = agent.traits.get("resilience", 0.5)
        suspicion = agent.traits.get("suspicion", 0.4)
        fear = float(affect.get("fear", 0.0))
        stress = float(affect.get("stress", 0.0))
        trust = float(affect.get("trust", 0.0))
        grief = float(affect.get("grief", 0.0))
        suspicion_state = float(affect.get("suspicion", 0.0))
        relief = float(affect.get("relief", 0.0))
        safe_bias = float(action_bias.get("seek_safe_house", 0.0))
        hospital_bias = float(action_bias.get("seek_hospital", 0.0))
        help_bias = float(action_bias.get("help", 0.0))
        warn_bias = float(action_bias.get("warn", 0.0))
        gather_bias = float(action_bias.get("gather", 0.0))
        has_medical = any(skill in agent.skills for skill in ("medicine", "triage", "comfort"))
        has_defense = any(skill in agent.skills for skill in ("combat", "patrol", "tracking", "escort"))
        has_supply = any(skill in agent.skills for skill in ("farming", "foraging", "cooking", "scavenging"))
        # Only farm/foraging workers make routine Farm Land runs; cooks/scavengers work at their own buildings
        is_farm_worker = any(skill in agent.skills for skill in ("farming", "foraging"))
        current_suspicion = max(suspicion, suspicion_state)
        social_drive = max(trust, help_bias + empathy * 0.2)
        danger_drive = max(fear, safe_bias * 0.6)
        scarcity_pressure = self._scarcity_pressure(world, agent)
        low_food = agent.inventory.get("food", 0) <= 1
        betrayal_mode = world.scenario == "betrayal_refusal"
        guarded_mode = scarcity_pressure > 0.58 or (low_food and betrayal_mode) or current_suspicion > 0.62 or stress > 0.48

        if agent.health < 45 or hospital_bias > 0.75:
            return self._travel_to(world.buildings[clinic_name], "seek_hospital", "heading to the clinic")

        wounded_anywhere = self._find_wounded_ally(agent, world.agents.values())
        nearby_wounded = wounded_anywhere if wounded_anywhere and manhattan(agent.location, wounded_anywhere.location) <= 2 else None
        crisis_mode = world.time_of_day in {"sunset", "night"} or world.weather == "storm"
        if wounded_anywhere and (world.time_of_day == "day" or nearby_wounded):
            label = agent.relationship_labels.get(wounded_anywhere.name, "neighbors")
            tie_value = agent.social_ties.get(wounded_anywhere.name, 0.0)
            positive_tie = tie_value > 0.25
            strong_tie = tie_value > 0.55 or label in {"allies", "partners", "medical_team", "mentor_pair", "friends", "command_pair", "household"}
            recent_refusal_count = self._recent_refusal_count(agent, wounded_anywhere.name)
            betrayal_pressure = betrayal_mode and (recent_refusal_count > 0 or tie_value < -0.2)
            if has_medical and (positive_tie or empathy > 0.68 or wounded_anywhere.health < 35 or help_bias > 0.45 or grief > 0.35):
                if guarded_mode and not strong_tie and wounded_anywhere.health >= 25 and (betrayal_pressure or label in {"rivals", "strained"}):
                    return ActionDecision(
                        "refuse_help",
                        "refuse",
                        agent.location,
                        None,
                        f"holding back from helping {wounded_anywhere.name} under pressure",
                        "N",
                        social_target=wounded_anywhere.name,
                    )
                if nearby_wounded:
                    return ActionDecision(
                        "help_other",
                        "rescue",
                        world.buildings[clinic_name].location,
                        clinic_name,
                        f"escorting {wounded_anywhere.name} toward treatment",
                        "A",
                        social_target=wounded_anywhere.name,
                    )
                return ActionDecision(
                    "help_other",
                    "rescue",
                    wounded_anywhere.location,
                    wounded_anywhere.name,
                    f"moving to reach injured {wounded_anywhere.name}",
                    "A",
                    social_target=wounded_anywhere.name,
                )
            if has_defense and nearby_wounded and (positive_tie or courage > 0.78 or wounded_anywhere.health < 30 or help_bias > 0.4):
                if guarded_mode and not strong_tie and wounded_anywhere.health >= 22 and (betrayal_pressure or label in {"rivals", "strained"}):
                    return ActionDecision(
                        "refuse_help",
                        "refuse",
                        agent.location,
                        None,
                        f"refusing to risk an escort for {wounded_anywhere.name}",
                        "N",
                        social_target=wounded_anywhere.name,
                    )
                return ActionDecision(
                    "help_other",
                    "escort",
                    world.buildings[town_refuge_name].location,
                    town_refuge_name,
                    f"escorting {wounded_anywhere.name} toward shelter",
                    "A",
                    social_target=wounded_anywhere.name,
                )
            if nearby_wounded and (
                label in {"rivals", "strained"}
                or (betrayal_pressure and not strong_tie and not positive_tie)
                or (current_suspicion > 0.45 and betrayal_mode)
                or scarcity_pressure > 0.58
                or (crisis_mode and help_bias < 0.22 and (betrayal_pressure or label in {"rivals", "strained"}))
            ):
                return ActionDecision(
                    "refuse_help",
                    "refuse",
                    agent.location,
                    None,
                    f"refusing to risk resources for {wounded_anywhere.name}",
                    "N",
                    social_target=wounded_anywhere.name,
                )

        nearby_hungry = self._find_nearby_hungry_ally(agent, world.agents.values())
        if nearby_hungry and world.time_of_day == "day":
            label = agent.relationship_labels.get(nearby_hungry.name, "neighbors")
            tie_value = agent.social_ties.get(nearby_hungry.name, 0.0)
            positive_tie = tie_value > 0.35
            strong_tie = tie_value > 0.58 or label in {"allies", "partners", "medical_team", "mentor_pair", "friends", "command_pair", "household"}
            recent_refusal_count = self._recent_refusal_count(agent, nearby_hungry.name)
            betrayal_pressure = betrayal_mode and (recent_refusal_count > 0 or tie_value < -0.2)
            required_food = 3 if scarcity_pressure > 0.45 or world.weather == "storm" else 2
            willing_to_share = (
                agent.inventory.get("food", 0) >= required_food
                and (
                    strong_tie
                    or (positive_tie and empathy > 0.74 and social_drive > 0.55 and scarcity_pressure < 0.55)
                )
            )
            if willing_to_share:
                return ActionDecision(
                    "share_supplies",
                    "share",
                    agent.location,
                    None,
                    f"sharing food with {nearby_hungry.name}",
                    "G",
                    social_target=nearby_hungry.name,
                )
            if (
                label in {"rivals", "strained"}
                or (betrayal_pressure and not strong_tie)
                or (current_suspicion > 0.52 and not positive_tie)
                or low_food
                or scarcity_pressure > 0.5
                or (betrayal_pressure and current_suspicion > 0.35)
                or (not positive_tie and empathy < 0.72)
            ):
                return ActionDecision(
                    "refuse_help",
                    "refuse",
                    agent.location,
                    None,
                    f"turning away {nearby_hungry.name}'s request for supplies",
                    "N",
                    social_target=nearby_hungry.name,
                )

        if world.weather == "storm" and world.time_of_day == "day":
            if has_supply and world.supplies.get("Farm Land", 0) > 5 and courage > 0.8 and resilience > 0.82 and gather_bias > 0.45 and scarcity_pressure < 0.4:
                return self._travel_to(world.buildings["Farm Land"], "gather_supplies", "making a quick storm supply run")
            refuge_name = self._pick_night_refuge(agent, world, town_refuge_name, fear=fear, stress=stress)
            return self._travel_to(world.buildings[refuge_name], "hide", f"moving early toward {refuge_name.lower()} because of the storm")

        if observation.visible_ghosts and has_defense and courage > 0.7 and observation.nearby_allies > 0 and warn_bias > 0.25:
            return ActionDecision("warn_others", "warn", agent.location, None, "holding position long enough to warn nearby survivors", "W")

        # Shelter trigger: sunset/night is the hard rule for all conditions.
        # During daytime, early shelter is ONLY triggered if the emotion engine
        # itself outputs a high seek_safe_house bias — letting each condition's
        # model decide whether past danger warrants early retreat.
        _daytime_early_shelter = (
            world.time_of_day == "day"
            and safe_bias > 1.2          # model must output strong shelter signal
            and agent.nights_survived > 0  # only after at least one night
        )

        if world.time_of_day in {"sunset", "night"} or _daytime_early_shelter:
            if not agent.sheltered or danger_drive > 0.12 or world.weather == "storm" or stress > 0.18 or scarcity_pressure > 0.45:
                refuge_name = self._pick_night_refuge(agent, world, town_refuge_name, fear=fear, stress=stress)
                action_desc = (
                    f"heading inside early — something feels wrong"
                    if _daytime_early_shelter
                    else f"rushing toward {refuge_name.lower()}"
                )
                return self._travel_to(world.buildings[refuge_name], "seek_safe_house", action_desc)

        # Mourning: only if the emotion engine outputs high grief (not hardcoded to day/time)
        if world.time_of_day == "day" and grief > 30.0 and safe_bias < 0.5 and world.step % 24 < 6:
            return ActionDecision(
                "mourn", "mourn", agent.location, None,
                "sitting in silence, unable to move past the grief",
                "😢",
            )

        if world.weather == "storm" and not agent.sheltered:
            refuge_name = self._pick_night_refuge(agent, world, "Police Station", fear=fear, stress=stress)
            return self._travel_to(world.buildings[refuge_name], "hide", f"looking for storm cover in {refuge_name.lower()}")

        if (agent.inventory.get("food", 0) == 0 and gather_bias > 0.0) or (
            is_farm_worker and world.supplies.get("Farm Land", 0) > 0 and world.time_of_day == "day" and (world.step % 4 == 0 or scarcity_pressure > 0.55) and gather_bias > 0.1
        ):
            target_name = self._best_supply_target(world)
            return self._travel_to(world.buildings[target_name], "gather_supplies", f"trying to gather supplies at {target_name.lower()}")

        if observation.visible_ghosts and observation.nearby_allies and warn_bias > 0.15:
            return ActionDecision("warn_others", "warn", agent.location, None, "warning nearby survivors about ghosts", "W")

        if world.time_of_day == "day":
            if betrayal_mode and current_suspicion > 0.55:
                nearby_rival = self._find_nearby_rival(agent, world.agents.values())
                current_building = next(
                    (building for building in world.buildings.values() if manhattan(agent.location, building.location) <= (1 if building.category in {"house", "forest_edge"} else 2)),
                    None,
                )
                if nearby_rival and current_building:
                    return ActionDecision(
                        "patrol",
                        "patrol",
                        current_building.location,
                        current_building.name,
                        f"holding ground near {current_building.name.lower()} and watching {nearby_rival.name}",
                        "P",
                        social_target=nearby_rival.name,
                    )
            if has_defense and (world.step % 5 == 0 or guarded_mode):
                patrol_target = "Police Station" if current_suspicion > 0.55 or stress > 0.35 else town_refuge_name
                return self._travel_to(world.buildings[patrol_target], "patrol", f"patrolling around {patrol_target.lower()}")
            if has_medical and (world.step % 4 == 0 or hospital_bias > 0.3 or help_bias > 0.35):
                return self._travel_to(world.buildings[clinic_name], "routine", "checking the clinic for injuries")
            if is_farm_worker and (world.step % 4 == 0 or scarcity_pressure > 0.3) and gather_bias > 0.15:
                supply_target = "Farm Land" if "Farm Land" in world.buildings else self._best_supply_target(world)
                return self._travel_to(world.buildings[supply_target], "gather_supplies", f"working through {supply_target.lower()}")
            target_name = self._pick_day_target(agent, world)
            return self._travel_to(world.buildings[target_name], "routine", f"moving through town toward {target_name.lower()}")

        retreat_name = agent.home if (resilience < 0.7 or grief > 0.35 or relief < 0.1) else town_refuge_name
        return self._travel_to(world.buildings[retreat_name], "rest", f"staying close to {retreat_name.lower()}")

    def _pick_day_target(self, agent: AgentState, world: WorldState) -> str:
        for goal_name in self._ordered_goals(agent.day_goals, world.step, len(agent.name)):
            if goal_name in world.buildings:
                return goal_name
        if agent.workplace and agent.workplace in world.buildings:
            return agent.workplace
        return agent.home

    def _pick_goal_destination(self, ordered_goals: List[str], world: WorldState, fallback: str) -> str:
        for goal_name in ordered_goals:
            if goal_name in world.buildings:
                return goal_name
        return fallback

    def _pick_night_refuge(self, agent: AgentState, world: WorldState, fallback: str, fear: float = 0.0, stress: float = 0.0) -> str:
        candidates = [self._pick_goal_destination(agent.night_goals, world, fallback)]
        if any(skill in agent.skills for skill in ("combat", "patrol", "escort")) and "Police Station" in world.buildings:
            candidates.append("Police Station")
        if any(skill in agent.skills for skill in ("medicine", "triage")) and "Clinic" in world.buildings:
            candidates.append("Clinic")
        if "Town House" in world.buildings:
            candidates.append("Town House")
        if fear > 0.25 or stress > 0.25 or world.weather == "storm":
            unique_candidates = list(dict.fromkeys(candidates))
            best_quality = max(world.buildings[name].shelter_quality for name in unique_candidates)
            viable = [
                name
                for name in unique_candidates
                if world.buildings[name].shelter_quality >= best_quality - 0.1
            ]
            return sorted(
                viable,
                key=lambda name: (
                    manhattan(agent.location, world.buildings[name].location),
                    -world.buildings[name].shelter_quality,
                ),
            )[0]
        return candidates[0]

    def _ordered_goals(self, goals: List[str], step: int, offset: int) -> List[str]:
        if not goals:
            return []
        pivot = (step // 3 + offset) % len(goals)
        return goals[pivot:] + goals[:pivot]

    def _best_supply_target(self, world: WorldState) -> str:
        available = [(name, count) for name, count in world.supplies.items() if count > 0 and name in world.buildings]
        if not available:
            return "Diner" if "Diner" in world.buildings else next(iter(world.buildings))
        return sorted(available, key=lambda item: item[1], reverse=True)[0][0]

    def _scarcity_pressure(self, world: WorldState, agent: AgentState) -> float:
        total_supplies = sum(world.supplies.values())
        town_scarcity = max(0.0, min(1.0, (42.0 - total_supplies) / 24.0))
        inventory_pressure = 0.45 if agent.inventory.get("food", 0) <= 0 else (0.22 if agent.inventory.get("food", 0) == 1 else 0.0)
        storm_penalty = 0.15 if world.weather == "storm" else 0.0
        return min(1.0, town_scarcity + inventory_pressure + storm_penalty)

    def _travel_to(self, building: Building, action: str, description: str) -> ActionDecision:
        pronunciatio = {
            "seek_safe_house": "S",
            "seek_hospital": "H",
            "gather_supplies": "G",
            "help_other": "A",
            "hide": "X",
            "patrol": "P",
            "routine": "M",
            "rest": "R",
        }.get(action, "M")
        return ActionDecision(action, action, building.location, building.name, description, pronunciatio)

    def _find_nearby_wounded_ally(self, agent: AgentState, others: Iterable[AgentState]) -> Optional[AgentState]:
        candidates: List[Tuple[int, AgentState]] = []
        for other in others:
            if other.name == agent.name or not other.alive or other.health >= 60:
                continue
            distance = manhattan(agent.location, other.location)
            if distance <= 2:
                candidates.append((distance, other))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _find_wounded_ally(self, agent: AgentState, others: Iterable[AgentState]) -> Optional[AgentState]:
        candidates: List[Tuple[int, AgentState]] = []
        for other in others:
            if other.name == agent.name or not other.alive or other.health >= 60:
                continue
            candidates.append((manhattan(agent.location, other.location), other))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _find_nearby_hungry_ally(self, agent: AgentState, others: Iterable[AgentState]) -> Optional[AgentState]:
        candidates: List[Tuple[int, AgentState]] = []
        for other in others:
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

    def _recent_refusal_count(self, agent: AgentState, target_name: str) -> int:
        return sum(
            1
            for event in agent.memory.recent_events[-8:]
            if event.event_type == "refusal" and event.actor == agent.name and event.target == target_name
        )

    def _find_nearby_rival(self, agent: AgentState, others: Iterable[AgentState]) -> Optional[AgentState]:
        candidates: List[Tuple[int, AgentState]] = []
        for other in others:
            if other.name == agent.name or not other.alive:
                continue
            if agent.relationship_labels.get(other.name, "neighbors") not in {"rivals", "strained"}:
                continue
            distance = manhattan(agent.location, other.location)
            if distance <= 3:
                candidates.append((distance, other))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]
