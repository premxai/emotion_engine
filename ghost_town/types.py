from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Coordinate = Tuple[int, int]


@dataclass
class Building:
    name: str
    category: str
    location: Coordinate
    shelter_quality: float
    capacity: int
    district: str = ""
    map_label: str = ""
    safe_at_night: bool = False
    tags: List[str] = field(default_factory=list)
    supply_spawn: int = 0
    treatment_zone: bool = False
    ghost_spawn: bool = False


@dataclass
class GhostState:
    ghost_id: str
    location: Coordinate
    target_agent: Optional[str] = None


@dataclass
class EventLog:
    step: int
    time_of_day: str
    weather: str
    event_type: str
    description: str
    actor: Optional[str] = None
    target: Optional[str] = None
    location: Optional[Coordinate] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class AgentMemory:
    recent_events: List[EventLog] = field(default_factory=list)
    known_deaths: List[str] = field(default_factory=list)
    trust_memory: Dict[str, float] = field(default_factory=dict)
    relationship_events: Dict[str, int] = field(default_factory=dict)


@dataclass
class AgentState:
    name: str
    home: str
    location: Coordinate
    social_ties: Dict[str, float]
    personality: str
    role: str = "resident"
    workplace: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    traits: Dict[str, float] = field(default_factory=dict)
    relationship_labels: Dict[str, str] = field(default_factory=dict)
    day_goals: List[str] = field(default_factory=list)
    night_goals: List[str] = field(default_factory=list)
    survival_style: str = "steady"
    alive: bool = True
    health: float = 100.0
    stamina: float = 100.0
    stress: float = 5.0
    fear: float = 0.0
    trust: float = 0.0
    grief: float = 0.0
    suspicion: float = 0.0
    inventory: Dict[str, int] = field(default_factory=lambda: {"food": 3, "medicine": 0})
    sheltered: bool = False
    current_goal: str = "routine"
    current_action: str = "rest"
    destination: Optional[Coordinate] = None
    destination_name: Optional[str] = None
    nav_path: List[Coordinate] = field(default_factory=list)  # BFS path cache; popped each step
    memory: AgentMemory = field(default_factory=AgentMemory)
    # Episodic memory counters updated each step by the simulator
    steps_since_ghost_seen: int = 20   # 0 = ghost visible now; 20 = never (cap)
    steps_since_ally_died: int = 20    # 0 = death visible now; 20 = never (cap)
    steps_since_betrayal: int = 20     # 0 = betrayed this step; 20 = never (cap)
    ally_deaths_witnessed: int = 0     # cumulative deaths seen across all nights
    betrayals_received: int = 0        # cumulative betrayals received this episode
    # Cross-night persistent memory
    nights_survived: int = 0           # how many nights this agent has lived through
    last_night_ghost_seen: bool = False  # saw a ghost last night → shelters earlier today
    last_night_death_nearby: bool = False  # witnessed a death last night → grieving today
    ghost_zones_known: List[Coordinate] = field(default_factory=list)  # danger locations remembered
    affect_embedding: List[float] = field(default_factory=list)
    prompt_context: Dict[str, object] = field(default_factory=dict)
    probe_data: Dict[str, object] = field(default_factory=dict)
    dialogue: List[List[str]] = field(default_factory=list)


@dataclass
class AffectOutput:
    label: str
    affect_vector: Dict[str, float]
    action_bias: Dict[str, float]
    prompt_context: Dict[str, object]
    latent_vector: List[float] = field(default_factory=list)
    probe_data: Dict[str, object] = field(default_factory=dict)


@dataclass
class WorldState:
    step: int
    width: int
    height: int
    scenario: str
    time_of_day: str
    weather: str
    storm_severity: float
    day_index: int
    agents: Dict[str, AgentState]
    buildings: Dict[str, Building]
    ghosts: List[GhostState]
    supplies: Dict[str, int]
    occupancy: Dict[str, List[str]]
    events: List[EventLog] = field(default_factory=list)


@dataclass
class ActionDecision:
    action: str
    goal: str
    destination: Optional[Coordinate]
    destination_name: Optional[str]
    description: str
    pronunciatio: str
    social_target: Optional[str] = None
    chat: Optional[List[List[str]]] = None


@dataclass
class ExperimentConfig:
    condition: str
    seed: int
    steps: int
    agent_count: int
    scenario: str = "standard_night"
    storm_mode: str = "stochastic"
    ghost_mode: str = "standard"
    run_id: str = ""
    curriculum_stage: str = "12-agent"
    condition_b_policy_source: str = "scaffold_fallback"
    condition_b_checkpoint: str = ""
    condition_b_schema: str = ""
    condition_b_policy_mode: str = "deterministic"
    condition_b_temperature: float = 1.0
    condition_c_policy_source: str = "hybrid_fallback"
    condition_c_checkpoint: str = ""
    condition_c_schema: str = ""
    condition_d_policy_source: str = "scaffold_fallback"
    condition_d_checkpoint: str = ""
    condition_d_schema: str = ""
    runtime_backend: str = "native"
    tmx_path: str = ""
    semantics_manifest: str = ""
    transfer_source_manifest: str = ""


@dataclass
class TrainingStepRecord:
    run_id: str
    step: int
    agent_id: str
    condition: str
    seed: int
    observation: Dict[str, object]
    social_context: Dict[str, float]
    prev_latent: List[float]
    latent: List[float]
    action: str
    goal: str
    reward_components: Dict[str, float]
    total_reward: float
    done: bool
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RunManifest:
    run_id: str
    schema_version: str
    timestamp_utc: str
    git_commit: str
    config: Dict[str, object]
    metrics: Dict[str, object]
    outputs: Dict[str, str]
    policy_source: str = ""
    condition_b_checkpoint: str = ""
    condition_b_schema: str = ""
    condition_b_policy_mode: str = ""
    condition_b_temperature: float = 1.0
    condition_c_checkpoint: str = ""
    condition_c_schema: str = ""
    condition_d_checkpoint: str = ""
    condition_d_schema: str = ""
    runtime_backend: str = "native"
    tmx_path: str = ""
    semantics_manifest: str = ""
    transfer_source_manifest: str = ""


@dataclass
class BatchExperimentResult:
    batch_id: str
    runs: Dict[str, List[str]]
    aggregate_summary_path: str
    aggregate_metrics_path: str
