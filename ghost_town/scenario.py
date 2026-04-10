from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from .content import HEIGHT, WIDTH, create_agent_roster, create_buildings
from .types import AgentState, Building, WorldState


DEFAULT_SCENARIO = "standard_night"

SCENARIO_PRESETS: Dict[str, Dict[str, object]] = {
    # ── Demo scenarios (3-day story arc, start 8 AM, clean 24-step days) ──────
    "three_day_story": {
        "phase_offset": 0,          # step 0 = 8 AM; every 24 steps = 1 new day
        "default_storm_mode": "none",
        "default_ghost_mode": "standard",
    },
    "three_day_high_ghost": {
        "phase_offset": 0,
        "default_storm_mode": "none",
        "default_ghost_mode": "high_pressure",
        "supply_overrides": {"Diner": 12, "Clinic": 8, "Town House": 12, "Farm Land": 10, "Police Station": 7},
    },
    "three_day_storm": {
        "phase_offset": 0,
        "default_storm_mode": "scheduled",
        "default_ghost_mode": "standard",
        "supply_overrides": {"Diner": 6, "Clinic": 5, "Town House": 5, "Farm Land": 4, "Police Station": 3},
    },
    "three_day_betrayal": {
        "phase_offset": 0,
        "default_storm_mode": "none",
        "default_ghost_mode": "standard",
        "supply_overrides": {"Diner": 4, "Clinic": 7, "Town House": 5, "Farm Land": 6, "Police Station": 5},
        "agent_locations": {"Nora Vale": "Diner", "Rosa Mercer": "Diner", "Ivy Hart": "Diner"},
        "inventory_overrides": {"Rosa Mercer": {"food": 0}, "Nora Vale": {"food": 1}, "Ivy Hart": {"food": 0}},
        "social_tie_overrides": {
            ("Nora Vale", "Rosa Mercer"): -0.68,
            ("Rosa Mercer", "Nora Vale"): -0.55,
            ("Nora Vale", "Ivy Hart"): -0.42,
            ("Ivy Hart", "Nora Vale"): -0.35,
        },
        "label_overrides": {
            ("Nora Vale", "Rosa Mercer"): "rivals",
            ("Rosa Mercia", "Nora Vale"): "rivals",
            ("Nora Vale", "Ivy Hart"): "strained",
            ("Ivy Hart", "Nora Vale"): "strained",
        },
    },
    # ── Training / proof scenarios (original offsets kept for reproducibility) ─
    "standard_night": {
        "phase_offset": 8,
        "default_storm_mode": "stochastic",
        "default_ghost_mode": "standard",
    },
    "high_ghost_pressure": {
        "phase_offset": 10,
        "default_storm_mode": "none",
        "default_ghost_mode": "high_pressure",
        "supply_overrides": {"Diner": 12, "Clinic": 8, "Town House": 12, "Farm Land": 10, "Police Station": 7},
    },
    "storm_scarcity": {
        "phase_offset": 8,
        "default_storm_mode": "scheduled",
        "default_ghost_mode": "standard",
        "supply_overrides": {"Diner": 6, "Clinic": 5, "Town House": 5, "Farm Land": 4, "Police Station": 3},
    },
    "ally_death": {
        "phase_offset": 10,
        "default_storm_mode": "none",
        "default_ghost_mode": "standard",
        "agent_locations": {
            "June Carter": "Town House",
            "Owen Pike": "Town House",
            "Dr. Mira Chen": "Town House",
        },
    },
    "crowded_shelter": {
        "phase_offset": 10,
        "default_storm_mode": "none",
        "default_ghost_mode": "standard",
        "capacity_overrides": {"Town House": 4, "Police Station": 4, "Clinic": 5, "Diner": 3},
        "agent_locations": {
            "Alma Ward": "Town House",
            "June Carter": "Town House",
            "Rosa Mercer": "Town House",
            "Caleb Dunn": "Town House",
            "Gideon Marsh": "Town House",
            "Wren Holloway": "Town House",
            "Ivy Hart": "Town House",
            "Owen Pike": "Town House",
        },
    },
    "betrayal_refusal": {
        "phase_offset": 5,
        "default_storm_mode": "none",
        "default_ghost_mode": "standard",
        "supply_overrides": {"Diner": 4, "Clinic": 7, "Town House": 5, "Farm Land": 6, "Police Station": 5},
        "agent_locations": {"Nora Vale": "Diner", "Rosa Mercer": "Diner", "Ivy Hart": "Diner"},
        "inventory_overrides": {"Rosa Mercer": {"food": 0}, "Nora Vale": {"food": 1}, "Ivy Hart": {"food": 0}},
        "social_tie_overrides": {
            ("Nora Vale", "Rosa Mercer"): -0.68,
            ("Rosa Mercer", "Nora Vale"): -0.55,
            ("Nora Vale", "Ivy Hart"): -0.42,
            ("Ivy Hart", "Nora Vale"): -0.35,
        },
        "label_overrides": {
            ("Nora Vale", "Rosa Mercer"): "rivals",
            ("Rosa Mercer", "Nora Vale"): "rivals",
            ("Nora Vale", "Ivy Hart"): "strained",
            ("Ivy Hart", "Nora Vale"): "strained",
        },
    },
}


def get_scenario_preset(scenario: str) -> Dict[str, object]:
    if scenario not in SCENARIO_PRESETS:
        valid = ", ".join(sorted(SCENARIO_PRESETS))
        raise ValueError(f"Unsupported scenario '{scenario}'. Expected one of: {valid}")
    return deepcopy(SCENARIO_PRESETS[scenario])


def resolve_mode_for_scenario(scenario: str, storm_mode: str | None, ghost_mode: str | None) -> Dict[str, str]:
    preset = get_scenario_preset(scenario)
    return {
        "storm_mode": storm_mode or str(preset["default_storm_mode"]),
        "ghost_mode": ghost_mode or str(preset["default_ghost_mode"]),
    }


def create_world(agent_count: int = 12, scenario: str = DEFAULT_SCENARIO) -> WorldState:
    buildings: Dict[str, Building] = create_buildings()
    agents: Dict[str, AgentState] = {agent.name: agent for agent in create_agent_roster(agent_count)}
    occupancy = {name: [] for name in buildings}
    supplies = {
        "Diner": 18,
        "Clinic": 10,
        "Town House": 14,
        "Farm Land": 16,
        "Police Station": 8,
    }
    world = WorldState(
        step=0,
        width=WIDTH,
        height=HEIGHT,
        scenario=scenario,
        time_of_day="day",
        weather="clear",
        storm_severity=0.0,
        day_index=1,
        agents=agents,
        buildings=buildings,
        ghosts=[],
        supplies=supplies,
        occupancy=occupancy,
    )
    apply_scenario(world, scenario)
    return world


def apply_scenario(world: WorldState, scenario: str) -> None:
    preset = get_scenario_preset(scenario)
    world.scenario = scenario

    for building_name, capacity in preset.get("capacity_overrides", {}).items():
        if building_name in world.buildings:
            world.buildings[building_name].capacity = int(capacity)

    for building_name, supply_value in preset.get("supply_overrides", {}).items():
        if building_name in world.supplies:
            world.supplies[building_name] = int(supply_value)

    for agent_name, building_name in preset.get("agent_locations", {}).items():
        if agent_name in world.agents and building_name in world.buildings:
            world.agents[agent_name].location = world.buildings[building_name].location

    for agent_name, inventory_updates in preset.get("inventory_overrides", {}).items():
        if agent_name not in world.agents:
            continue
        agent = world.agents[agent_name]
        for key, value in dict(inventory_updates).items():
            agent.inventory[key] = int(value)

    for pair, tie_value in preset.get("social_tie_overrides", {}).items():
        source, target = pair
        if source in world.agents and target in world.agents[source].social_ties:
            world.agents[source].social_ties[target] = round(float(tie_value), 3)
            world.agents[source].memory.trust_memory[target] = world.agents[source].social_ties[target]

    for pair, label in preset.get("label_overrides", {}).items():
        source, target = pair
        if source in world.agents and target in world.agents[source].relationship_labels:
            world.agents[source].relationship_labels[target] = str(label)

    world.occupancy = {name: [] for name in world.buildings}
    for agent in world.agents.values():
        building = building_at(world.buildings, agent.location)
        if building:
            world.occupancy[building.name].append(agent.name)
            agent.sheltered = building.safe_at_night
        else:
            agent.sheltered = False


def building_at(buildings: Dict[str, Building], location: tuple[int, int]) -> Building | None:
    best_match: Building | None = None
    best_distance: int | None = None
    for building in buildings.values():
        radius = 1 if building.category in {"house", "forest_edge"} else 2
        distance = abs(building.location[0] - location[0]) + abs(building.location[1] - location[1])
        if distance <= radius and (best_distance is None or distance < best_distance):
            best_match = building
            best_distance = distance
    return best_match


def list_scenarios() -> List[str]:
    return list(SCENARIO_PRESETS.keys())
