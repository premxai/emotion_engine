# Emotion Engine: Emergent Emotional Appraisal in Generative Agents

A five-condition empirical comparison of emotion architectures for generative agents,
built on a custom ghost-survival simulation (Ghost Town).

The core finding: **a model trained only to predict future world events independently
rediscovers fear, grief, and suspicion as functional internal states** — without any
emotion rules, labels, or reward shaping.

> Paper: *Emergent Emotional Appraisal in Generative Agents via Predictive World Modeling*
> Author: Prem Babu Kanaparthi

---

## Overview

Five emotion-engine conditions are compared in a controlled survival scenario:

| Condition | Architecture |
|-----------|-------------|
| Baseline (0) | No emotion — priority heuristics only |
| Condition A | Hand-coded OCC appraisal rules |
| Condition B | Behavioral cloning on Condition A |
| Condition C | Emotion dynamics model (latent transition) |
| **Condition D** | **Predictive world modeling (proposed)** |

All conditions share the same world, agents, and random seeds.
Only the emotion architecture differs.

---

## Key Results

| Signature | Condition A | **Condition D** |
|-----------|-------------|-----------------|
| Fear → Shelter | 97.9% | **99.0%** |
| Fisher p-value | 1.4×10⁻¹⁰¹ | **3.3×10⁻¹¹³** |
| Suspicion decay | programmed | **emergent** |
| Latent dims interpretable | N/A | **7 / 8** |
| Mean survivors / 12 | 10.1 | **11.8** |

N = 205,940 agent-step records, 8 seeds × 6 scenarios × 12 agents × 72 steps.

---

## Repository Structure

```
emotion-engine/
├── ghost_town/              # Core simulation engine
│   ├── simulator.py         # Main simulation loop
│   ├── emotions.py          # All 5 emotion-engine implementations
│   ├── planner.py           # Action resolution & pathfinding
│   ├── scenario.py          # 6 scenario presets
│   ├── types.py             # AgentState, WorldState, AffectOutput
│   ├── content.py           # Agent roster & building definitions
│   ├── export.py            # Result serialisation
│   └── analysis.py          # Batch statistics
├── ghost_town_training/     # PyTorch training pipeline
│   ├── model.py             # ConditionB/C/D model definitions
│   ├── dataset.py           # Imitation learning dataset
│   ├── dataset_d.py         # Predictive modeling dataset
│   ├── training.py          # Condition B training loop
│   ├── training_c.py        # Condition C training loop
│   ├── training_d.py        # Condition D training loop
│   ├── inference.py         # Condition B inference backend
│   ├── inference_c.py       # Condition C inference backend
│   └── inference_d.py       # Condition D inference backend
├── environment/             # Django visualization server (browser demo)
│   └── frontend_server/
├── tests/                   # Test suite (unittest)
├── run_ghost_town.py        # Single-run entry point
├── run_ghost_town_batch.py  # Batch experiment runner
├── train_condition_b.py     # Train Condition B
├── train_condition_c.py     # Train Condition C
├── train_condition_d.py     # Train Condition D
├── behavioral_proof.py      # Statistical proof (Fisher's exact)
└── requirements.txt
```

---

## Results

### Publication Figures

All figures from the paper are in [`figures/`](figures/):

| File | What it shows |
|------|--------------|
| [fig1_behavioral_proof.png](figures/fig1_behavioral_proof.png) | Fear→Shelter rates and Suspicion Decay curves across all 5 conditions |
| [fig2_interpretability.png](figures/fig2_interpretability.png) | 8×12 Pearson r latent interpretability heatmap + χ² per dimension |
| [fig3_gallup.png](figures/fig3_gallup.png) | Resting daytime emotion rates vs. Gallup 2024 (N=145,000) |
| [fig4_social.png](figures/fig4_social.png) | Survival, trust, rescues, and refusals across all conditions and scenarios |
| [emergent_emotion_proof.png](figures/emergent_emotion_proof.png) | Full emergent emotion proof composite |
| [gallup_calibration.png](figures/gallup_calibration.png) | Extended Gallup calibration chart |
| [cross_night_shelter_timing.png](figures/cross_night_shelter_timing.png) | Cross-night shelter timing (night 1 vs. 2 vs. 3) |

### Pre-Trained Checkpoint

A trained Condition D model is included at [`checkpoints/condition_d/`](checkpoints/condition_d/):

| File | Purpose |
|------|---------|
| `best.pt` | Best checkpoint by validation loss (164 KB) |
| `vocab_and_schema.json` | Feature schema required for inference |
| `eval_metrics.json` | Validation metrics at best checkpoint |
| `train_metrics.json` | Full training loss curve |

Use it directly without training:

```bash
python run_ghost_town_batch.py \
  --conditions condition_d \
  --scenario standard_night \
  --seeds 0 1 2 3 \
  --condition-d-checkpoint checkpoints/condition_d/best.pt \
  --condition-d-schema checkpoints/condition_d/vocab_and_schema.json \
  --output outputs/my_run
```

### Full Dataset

The complete simulation dataset (205,940 agent-step records, ~2.1 GB) is too large
for this repository. It will be hosted on Zenodo/HuggingFace Datasets —
link to be added here upon publication.

To regenerate it locally:

```bash
python run_ghost_town_batch.py \
  --conditions baseline_0 condition_a condition_b condition_c condition_d \
  --scenario standard_night high_ghost_pressure storm_scarcity \
             ally_death betrayal_refusal crowded_shelter \
  --seeds 0 1 2 3 4 5 6 7 \
  --condition-d-checkpoint checkpoints/condition_d/best.pt \
  --condition-d-schema checkpoints/condition_d/vocab_and_schema.json \
  --output outputs/five_condition_72steps
```

---

## Setup

```bash
# Python 3.9+ required
pip install -r requirements.txt
```

---

## Running Simulations

### Single run

```bash
# Condition D, standard night scenario, 12 agents, 72 steps
python run_ghost_town.py \
  --condition condition_d \
  --scenario standard_night \
  --agents 12 \
  --steps 72 \
  --output outputs/my_run
```

### Batch experiment (all 5 conditions, 8 seeds)

```bash
python run_ghost_town_batch.py \
  --conditions baseline_0 condition_a condition_b condition_c condition_d \
  --scenario standard_night \
  --seeds 0 1 2 3 4 5 6 7 \
  --condition-d-checkpoint outputs/condition_d/best.pt \
  --condition-d-schema outputs/condition_d/vocab_and_schema.json \
  --output outputs/five_condition_72steps
```

Available scenarios: `standard_night`, `high_ghost_pressure`, `storm_scarcity`,
`ally_death`, `betrayal_refusal`, `crowded_shelter`.

---

## Training

### Train Condition B (behavioral cloning on Condition A)

```bash
python train_condition_b.py \
  --data outputs/five_condition_72steps \
  --output outputs/condition_b
```

### Train Condition C (emotion dynamics model)

```bash
python train_condition_c.py \
  --data outputs/five_condition_72steps \
  --output outputs/condition_c
```

### Train Condition D (predictive world modeling)

```bash
python train_condition_d.py \
  --data outputs/five_condition_72steps \
  --output outputs/condition_d
```

---

## Statistical Proof

```bash
python behavioral_proof.py \
  --batch-dir outputs/five_condition_72steps \
  --output outputs/behavioral_proof
```

Runs Fisher's exact tests for all three OCC behavioral signatures across all conditions.

---

## Visualization Server

```bash
cd environment/frontend_server

# Create local settings (first time only)
cat > frontend_server/settings/local.py << 'EOF'
from .base import *
SECRET_KEY = 'your-secret-key-here'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']
EOF

python manage.py migrate
python manage.py runserver
```

Then open `http://localhost:8000` to view the simulation demo.

---

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## The Ghost Town Environment

**12 agents** with fixed names, roles, and social ties navigate a shared
grid world (40×25 tiles) over 72 discrete steps (3 simulated days).

- **Day**: agents gather supplies, interact socially, plan shelter routes.
- **Night**: ghosts enter and hunt. Agents not in a safe building die.
- **Social graph**: trust, betrayal, and refusal cascade through the network.

The environment provides natural opportunity for all three OCC appraisal circuits:
- **Fear**: ghost threat → shelter-seeking
- **Grief**: ally deaths → behavioral reorganization
- **Suspicion**: betrayals → refusal, with decay over time

---

## Condition D Architecture

Condition D's model receives a **21-dimensional observation vector** (ghost visibility,
death witness count, ally proximity, supply state, shelter status, social graph tension)
and outputs two heads simultaneously:

**Head 1 — Action logits** (7 actions):
`hide`, `gather_supplies`, `seek_safe_house`, `seek_hospital`,
`refuse_help`, `warn`, `patrol`

**Head 2 — 12 future-event predictions** (binary, sigmoid):
`ghost_nearby_t3`, `my_death_t5`, `health_drop_t3`, `nearby_death_t5`,
`help_success_t5`, `refusal_received_t5`, `tie_increase_t5`,
`shelter_achieved_t2`, `storm_onset_t3`, `scarcity_t5`,
`graph_tension_t3`, `valence_t5`

Emotion labels are derived **post-hoc** from prediction confidence:

```
fear  = P(ghost_nearby_t3) + 0.3 × P(health_drop_t3)
grief = 0.8 × P(nearby_death_t5) + 0.2 × P(my_death_t5)
```

The model never trains on emotion labels. They emerge as interpretable
correlates of the prediction objective.

---

## Citation

```bibtex
@article{kanaparthi2026emotion,
  author  = {Kanaparthi, Prem Babu},
  title   = {Emergent Emotional Appraisal in Generative Agents
             via Predictive World Modeling},
  year    = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
