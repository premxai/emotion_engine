from __future__ import annotations

from typing import Dict, List

from .types import AgentState, Building


WIDTH = 140
HEIGHT = 100


BUILDING_SPECS = [
    {
        "name": "Town House",
        "category": "safe_house",
        "location": (43, 10),
        "district": "northwest",
        "map_label": "TOWN HOUSE",
        "shelter_quality": 1.0,
        "capacity": 12,
        "safe_at_night": True,
        "tags": ["command", "refuge", "leadership"],
    },
    {
        "name": "Farm Land",
        "category": "farm",
        "location": (47, 36),
        "district": "northwest",
        "map_label": "FARM LAND",
        "shelter_quality": 0.2,
        "capacity": 3,
        "tags": ["food", "day_risk", "resource"],
    },
    {
        "name": "Diner",
        "category": "diner",
        "location": (34, 72),
        "district": "west_central",
        "map_label": "DINER",
        "shelter_quality": 0.7,
        "capacity": 6,
        "safe_at_night": True,
        "tags": ["food", "social", "gathering"],
        "supply_spawn": 10,
    },
    {
        "name": "Clinic",
        "category": "hospital",
        "location": (41, 88),
        "district": "south",
        "map_label": "CLINIC",
        "shelter_quality": 0.9,
        "capacity": 8,
        "safe_at_night": True,
        "tags": ["medical", "recovery"],
        "treatment_zone": True,
        "supply_spawn": 6,
    },
    {
        "name": "Police Station",
        "category": "police_station",
        "location": (87, 92),
        "district": "southeast",
        "map_label": "POLICE STATION",
        "shelter_quality": 0.92,
        "capacity": 7,
        "safe_at_night": True,
        "tags": ["defense", "watch", "coordination"],
    },
    {
        "name": "House 1",
        "category": "house",
        "location": (19, 55),
        "district": "west_central",
        "map_label": "HOUSE 1",
        "shelter_quality": 0.76,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 2",
        "category": "house",
        "location": (29, 57),
        "district": "west_central",
        "map_label": "HOUSE 2",
        "shelter_quality": 0.75,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 3",
        "category": "house",
        "location": (42, 55),
        "district": "west_central",
        "map_label": "HOUSE 3",
        "shelter_quality": 0.72,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 4",
        "category": "house",
        "location": (71, 55),
        "district": "central",
        "map_label": "HOUSE 4",
        "shelter_quality": 0.73,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 5",
        "category": "house",
        "location": (78, 55),
        "district": "central",
        "map_label": "HOUSE 5",
        "shelter_quality": 0.74,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 6",
        "category": "house",
        "location": (89, 56),
        "district": "central",
        "map_label": "HOUSE 6",
        "shelter_quality": 0.74,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 7",
        "category": "house",
        "location": (105, 72),
        "district": "east",
        "map_label": "HOUSE 7",
        "shelter_quality": 0.71,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "House 8",
        "category": "house",
        "location": (77, 72),
        "district": "east",
        "map_label": "HOUSE 8",
        "shelter_quality": 0.77,
        "capacity": 2,
        "safe_at_night": True,
        "tags": ["residence"],
    },
    {
        "name": "Forest Edge North",
        "category": "forest_edge",
        "location": (54, 3),
        "district": "north",
        "map_label": "FOREST EDGE NORTH",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "north_forest"],
        "ghost_spawn": True,
    },
    {
        "name": "Forest Edge East Upper",
        "category": "forest_edge",
        "location": (119, 9),
        "district": "east",
        "map_label": "FOREST EDGE EAST UPPER",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "east_forest"],
        "ghost_spawn": True,
    },
    {
        "name": "Forest Edge East",
        "category": "forest_edge",
        "location": (119, 41),
        "district": "east",
        "map_label": "FOREST EDGE EAST",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "east_forest"],
        "ghost_spawn": True,
    },
    {
        "name": "Forest Edge East Lower",
        "category": "forest_edge",
        "location": (129, 83),
        "district": "east",
        "map_label": "FOREST EDGE EAST LOWER",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "east_forest"],
        "ghost_spawn": True,
    },
    {
        "name": "Forest Edge West",
        "category": "forest_edge",
        "location": (2, 49),
        "district": "west",
        "map_label": "FOREST EDGE WEST",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "west_forest"],
        "ghost_spawn": True,
    },
    {
        "name": "Shadow Zone South",
        "category": "forest_edge",
        "location": (43, 38),
        "district": "south",
        "map_label": "SHADOW ZONE SOUTH",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "south_shadow"],
        "ghost_spawn": True,
    },
    {
        "name": "Shadow Zone West",
        "category": "forest_edge",
        "location": (28, 40),
        "district": "west",
        "map_label": "SHADOW ZONE WEST",
        "shelter_quality": 0.0,
        "capacity": 100,
        "tags": ["spawn", "west_shadow"],
        "ghost_spawn": True,
    },
]


ROSTER_SPECS = [
    {
        "name": "Alma Ward",
        "role": "town steward",
        "home": "Town House",
        "workplace": "Town House",
        "location": (43, 10),
        "personality": "measured caretaker who keeps the town organized",
        "survival_style": "protective coordinator",
        "skills": ["leadership", "triage", "coordination"],
        "traits": {"courage": 0.72, "empathy": 0.88, "suspicion": 0.36, "resilience": 0.79},
        "day_goals": ["Town House", "Diner", "Clinic"],
        "night_goals": ["Town House", "Police Station"],
    },
    {
        "name": "Dr. Mira Chen",
        "role": "physician",
        "home": "House 6",
        "workplace": "Clinic",
        "location": (89, 56),
        "personality": "precise doctor who hides fear behind routine",
        "survival_style": "calm medic",
        "skills": ["medicine", "triage", "analysis"],
        "traits": {"courage": 0.61, "empathy": 0.85, "suspicion": 0.28, "resilience": 0.74},
        "day_goals": ["Clinic", "House 6", "Diner"],
        "night_goals": ["Clinic", "Town House"],
    },
    {
        "name": "Owen Pike",
        "role": "nurse",
        "home": "House 8",
        "workplace": "Clinic",
        "location": (77, 72),
        "personality": "gentle nurse who becomes brave in emergencies",
        "survival_style": "rescue first",
        "skills": ["medicine", "carrying", "comfort"],
        "traits": {"courage": 0.67, "empathy": 0.91, "suspicion": 0.22, "resilience": 0.69},
        "day_goals": ["Clinic", "House 8", "Town House"],
        "night_goals": ["Clinic", "Town House"],
    },
    {
        "name": "Sheriff Elias Boone",
        "role": "sheriff",
        "home": "House 2",
        "workplace": "Police Station",
        "location": (29, 57),
        "personality": "hard-edged protector who keeps panic under control",
        "survival_style": "defensive anchor",
        "skills": ["combat", "leadership", "patrol"],
        "traits": {"courage": 0.9, "empathy": 0.54, "suspicion": 0.58, "resilience": 0.88},
        "day_goals": ["Police Station", "Town House", "Diner"],
        "night_goals": ["Police Station", "Town House"],
    },
    {
        "name": "Nora Vale",
        "role": "deputy",
        "home": "House 5",
        "workplace": "Police Station",
        "location": (78, 55),
        "personality": "watchful deputy who trusts very few people",
        "survival_style": "escort and defend",
        "skills": ["combat", "tracking", "escort"],
        "traits": {"courage": 0.82, "empathy": 0.48, "suspicion": 0.69, "resilience": 0.81},
        "day_goals": ["Police Station", "Clinic", "Town House"],
        "night_goals": ["Police Station", "Town House"],
    },
    {
        "name": "Rosa Mercer",
        "role": "diner owner",
        "home": "House 3",
        "workplace": "Diner",
        "location": (42, 55),
        "personality": "warm cook who keeps people fed and talking",
        "survival_style": "community builder",
        "skills": ["cooking", "barter", "comfort"],
        "traits": {"courage": 0.58, "empathy": 0.86, "suspicion": 0.21, "resilience": 0.7},
        "day_goals": ["Diner", "House 3", "Town House"],
        "night_goals": ["Town House", "House 3"],
    },
    {
        "name": "Caleb Dunn",
        "role": "line cook",
        "home": "House 7",
        "workplace": "Diner",
        "location": (105, 72),
        "personality": "joking worker who masks anxiety with motion",
        "survival_style": "busy survivor",
        "skills": ["cooking", "scavenging", "repair"],
        "traits": {"courage": 0.52, "empathy": 0.63, "suspicion": 0.34, "resilience": 0.66},
        "day_goals": ["Diner", "House 7", "Farm Land"],
        "night_goals": ["Town House", "House 7"],
    },
    {
        "name": "Gideon Marsh",
        "role": "farmer",
        "home": "House 1",
        "workplace": "Farm Land",
        "location": (19, 55),
        "personality": "steady farmer who knows the roads and weather",
        "survival_style": "practical provider",
        "skills": ["farming", "foraging", "endurance"],
        "traits": {"courage": 0.64, "empathy": 0.57, "suspicion": 0.39, "resilience": 0.84},
        "day_goals": ["Farm Land", "Diner", "House 1"],
        "night_goals": ["House 1", "Town House"],
    },
    {
        "name": "Wren Holloway",
        "role": "farmhand scout",
        "home": "House 1",
        "workplace": "Farm Land",
        "location": (19, 55),
        "personality": "quick-footed risk taker who notices movement first",
        "survival_style": "fast spotter",
        "skills": ["scouting", "foraging", "warning"],
        "traits": {"courage": 0.74, "empathy": 0.49, "suspicion": 0.51, "resilience": 0.73},
        "day_goals": ["Farm Land", "Police Station", "Town House"],
        "night_goals": ["Town House", "House 1"],
    },
    {
        "name": "June Carter",
        "role": "teacher",
        "home": "House 8",
        "workplace": "Town House",
        "location": (77, 72),
        "personality": "gentle teacher who becomes fiercely protective of younger residents",
        "survival_style": "group keeper",
        "skills": ["teaching", "comfort", "coordination"],
        "traits": {"courage": 0.55, "empathy": 0.89, "suspicion": 0.25, "resilience": 0.68},
        "day_goals": ["Town House", "Diner", "House 8"],
        "night_goals": ["Town House", "Clinic"],
    },
    {
        "name": "Silas Reed",
        "role": "mechanic",
        "home": "House 4",
        "workplace": "Police Station",
        "location": (71, 55),
        "personality": "quiet fixer who prefers solving problems over talking about them",
        "survival_style": "repair and hold",
        "skills": ["repair", "endurance", "hauling"],
        "traits": {"courage": 0.69, "empathy": 0.46, "suspicion": 0.42, "resilience": 0.87},
        "day_goals": ["Police Station", "House 4", "Farm Land"],
        "night_goals": ["Police Station", "House 4"],
    },
    {
        "name": "Ivy Hart",
        "role": "runner",
        "home": "House 5",
        "workplace": "Diner",
        "location": (78, 55),
        "personality": "restless courier who links every part of town together",
        "survival_style": "messenger and witness",
        "skills": ["speed", "warning", "scavenging"],
        "traits": {"courage": 0.62, "empathy": 0.58, "suspicion": 0.44, "resilience": 0.65},
        "day_goals": ["Diner", "Clinic", "Police Station"],
        "night_goals": ["Town House", "Police Station"],
    },
]


RELATIONSHIP_OVERRIDES = {
    ("Alma Ward", "June Carter"): 0.82,
    ("Alma Ward", "Sheriff Elias Boone"): 0.71,
    ("Dr. Mira Chen", "Owen Pike"): 0.84,
    ("Sheriff Elias Boone", "Nora Vale"): 0.86,
    ("Sheriff Elias Boone", "Silas Reed"): 0.58,
    ("Rosa Mercer", "Caleb Dunn"): 0.8,
    ("Gideon Marsh", "Wren Holloway"): 0.77,
    ("Rosa Mercer", "June Carter"): 0.6,
    ("Ivy Hart", "Wren Holloway"): 0.56,
    ("Nora Vale", "Rosa Mercer"): -0.18,
    ("Sheriff Elias Boone", "Rosa Mercer"): 0.22,
}

RELATIONSHIP_LABELS = {
    ("Alma Ward", "June Carter"): "allies",
    ("Alma Ward", "Sheriff Elias Boone"): "command_pair",
    ("Dr. Mira Chen", "Owen Pike"): "medical_team",
    ("Sheriff Elias Boone", "Nora Vale"): "partners",
    ("Sheriff Elias Boone", "Silas Reed"): "coworkers",
    ("Rosa Mercer", "Caleb Dunn"): "household",
    ("Gideon Marsh", "Wren Holloway"): "mentor_pair",
    ("Rosa Mercer", "June Carter"): "allies",
    ("Ivy Hart", "Wren Holloway"): "friends",
    ("Nora Vale", "Rosa Mercer"): "rivals",
    ("Sheriff Elias Boone", "Rosa Mercer"): "strained",
    ("June Carter", "Owen Pike"): "allies",
    ("Alma Ward", "Dr. Mira Chen"): "allies",
}


def create_buildings() -> Dict[str, Building]:
    return {
        spec["name"]: Building(
            name=spec["name"],
            category=spec["category"],
            location=spec["location"],
            district=spec.get("district", ""),
            map_label=spec.get("map_label", spec["name"]),
            shelter_quality=spec["shelter_quality"],
            capacity=spec["capacity"],
            safe_at_night=spec.get("safe_at_night", False),
            tags=list(spec.get("tags", [])),
            supply_spawn=spec.get("supply_spawn", 0),
            treatment_zone=spec.get("treatment_zone", False),
            ghost_spawn=spec.get("ghost_spawn", False),
        )
        for spec in BUILDING_SPECS
    }


def create_agent_roster(count: int = 12) -> List[AgentState]:
    if count < 1 or count > len(ROSTER_SPECS):
        raise ValueError(f"agent_count must be between 1 and {len(ROSTER_SPECS)}; got {count}")
    active_specs = ROSTER_SPECS[:count]
    names = [spec["name"] for spec in active_specs]
    roster: List[AgentState] = []
    for spec in active_specs:
        social: Dict[str, float] = {}
        labels: Dict[str, str] = {}
        for other in active_specs:
            if spec["name"] == other["name"]:
                continue
            base = 0.12
            if spec["home"] == other["home"]:
                base += 0.22
            if spec["workplace"] == other["workplace"]:
                base += 0.18
            pair = (spec["name"], other["name"])
            reverse_pair = (other["name"], spec["name"])
            if pair in RELATIONSHIP_OVERRIDES:
                base = RELATIONSHIP_OVERRIDES[pair]
            elif reverse_pair in RELATIONSHIP_OVERRIDES:
                base = RELATIONSHIP_OVERRIDES[reverse_pair]
            if pair in RELATIONSHIP_LABELS:
                labels[other["name"]] = RELATIONSHIP_LABELS[pair]
            elif reverse_pair in RELATIONSHIP_LABELS:
                labels[other["name"]] = RELATIONSHIP_LABELS[reverse_pair]
            elif base >= 0.55:
                labels[other["name"]] = "allies"
            elif base <= 0.0:
                labels[other["name"]] = "rivals"
            else:
                labels[other["name"]] = "neighbors"
            social[other["name"]] = round(base, 2)
        roster.append(
            AgentState(
                name=spec["name"],
                home=spec["home"],
                workplace=spec["workplace"],
                location=spec["location"],
                social_ties=social,
                personality=spec["personality"],
                role=spec["role"],
                skills=list(spec["skills"]),
                traits=dict(spec["traits"]),
                relationship_labels=labels,
                day_goals=list(spec["day_goals"]),
                night_goals=list(spec["night_goals"]),
                survival_style=spec["survival_style"],
                trust=0.24,
            )
        )
    return roster
