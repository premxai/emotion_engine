from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch

from ghost_town.types import AffectOutput
from .dataset import FeatureSchema
from .model import ConditionDPredictiveModel


class ConditionDInferenceBackend:
    """Runtime inference for condition_d: predictive emotion engine.

    The affect_vector is populated directly from prediction head outputs:
      - fear      = P(ghost nearby in 3 steps) + 0.3 * P(health drop in 3 steps)
      - grief     = P(nearby death in 5 steps) * 0.8 + P(my death in 5 steps) * 0.2
      - trust     = P(help success) * 0.6 + (1 - P(refusal received)) * 0.4
      - stress    = P(storm onset) * 0.5 + P(scarcity) * 0.3 + P(graph tension up) * 0.2
      - relief    = P(shelter achieved in 2 steps) * 0.7 + (1 - P(ghost nearby)) * 0.3
      - suspicion = P(refusal received) * 0.5 + (1 - P(tie increase)) * 0.3 + P(tension up) * 0.2

    Emotions are earned labels: "fear" exists because ghost_nearby_t3 predicts danger.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        schema_path: str | Path,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.schema_path = Path(schema_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.schema = FeatureSchema.load(self.schema_path)

        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        hidden_dims = payload.get("hidden_dims", [192, 128])
        self.model = ConditionDPredictiveModel(self.schema, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self._checkpoint_meta = {
            "training_scenarios": payload.get("training_scenarios", []),
            "holdout_scenarios": payload.get("holdout_scenarios", []),
        }

    def compute(
        self,
        observation: Dict[str, object],
        social_context: Dict[str, object],
        prev_latent: Sequence[float],
        metadata: Dict[str, object],
    ) -> AffectOutput:
        numeric, categorical = self.schema.encode_runtime(observation, social_context, prev_latent, metadata)
        numeric = numeric.to(self.device)
        categorical = categorical.to(self.device)

        with torch.no_grad():
            outputs = self.model(numeric, categorical)

        # Extract scalar predictions from each head
        def _scalar(key: str) -> float:
            return float(outputs[key][0].item())

        ghost_p = _scalar("ghost_nearby_t3")
        my_death_p = _scalar("my_death_t5")
        health_p = _scalar("health_drop_t3")
        nearby_death_p = _scalar("nearby_death_t5")
        help_p = _scalar("help_success_t5")
        refusal_p = _scalar("refusal_received_t5")
        tie_p = _scalar("tie_increase_t5")
        shelter_p = _scalar("shelter_achieved_t2")
        storm_p = _scalar("storm_onset_t3")
        scarcity_p = _scalar("scarcity_t5")
        tension_p = _scalar("graph_tension_increase_t3")
        valence = _scalar("valence_t5")  # [-1, 1]

        latent_vector = outputs["latent"][0].detach().cpu().tolist()
        action_logits = outputs["action_logits"][0].detach().cpu().tolist()
        action_probs = torch.softmax(outputs["action_logits"][0], dim=-1).tolist()
        top_action_idx = int(max(range(len(action_probs)), key=lambda i: action_probs[i]))
        top_action = self.schema.action_vocab[top_action_idx]

        # Compose affect vector from prediction outputs
        def _clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

        affect_vector = {
            "fear":      round(_clamp(ghost_p + 0.3 * health_p), 3),
            "grief":     round(_clamp(nearby_death_p * 0.8 + my_death_p * 0.2), 3),
            "trust":     round(_clamp(help_p * 0.6 + (1.0 - refusal_p) * 0.4), 3),
            "stress":    round(_clamp(storm_p * 0.5 + scarcity_p * 0.3 + tension_p * 0.2), 3),
            "relief":    round(_clamp(shelter_p * 0.7 + (1.0 - ghost_p) * 0.3), 3),
            "suspicion": round(_clamp(refusal_p * 0.5 + (1.0 - tie_p) * 0.3 + tension_p * 0.2), 3),
        }

        # Action bias derived from action probabilities
        action_bias = self._action_bias_from_probs(action_probs)

        top_pairs = sorted(
            zip(self.schema.action_vocab, action_probs),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return AffectOutput(
            label="condition_d_predictive",
            affect_vector=affect_vector,
            action_bias=action_bias,
            prompt_context={
                "tone": "predictive",
                "summary": (
                    f"top_action={top_action} p={action_probs[top_action_idx]:.2f} "
                    f"ghost_p={ghost_p:.2f} trust_p={help_p:.2f} valence={valence:.2f}"
                ),
            },
            latent_vector=[round(float(v), 3) for v in latent_vector],
            probe_data={
                "policy_source": "trained_checkpoint",
                "checkpoint": str(self.checkpoint_path),
                "top_actions": [[name, round(float(p), 4)] for name, p in top_pairs],
                "predictions": {
                    "ghost_nearby_t3": round(ghost_p, 4),
                    "my_death_t5": round(my_death_p, 4),
                    "health_drop_t3": round(health_p, 4),
                    "nearby_death_t5": round(nearby_death_p, 4),
                    "help_success_t5": round(help_p, 4),
                    "refusal_received_t5": round(refusal_p, 4),
                    "tie_increase_t5": round(tie_p, 4),
                    "shelter_achieved_t2": round(shelter_p, 4),
                    "storm_onset_t3": round(storm_p, 4),
                    "scarcity_t5": round(scarcity_p, 4),
                    "graph_tension_increase_t3": round(tension_p, 4),
                    "valence_t5": round(valence, 4),
                },
            },
        )

    def _action_bias_from_probs(self, action_probs: Sequence[float]) -> Dict[str, float]:
        lookup = {action: float(action_probs[idx]) for idx, action in enumerate(self.schema.action_vocab)}

        def _get(key: str) -> float:
            return lookup.get(key, 0.0)

        safe = _get("seek_safe_house") + 0.75 * _get("hide")
        hospital = _get("seek_hospital")
        help_score = (
            _get("help_other") * 1.05
            + _get("share_supplies") * 0.85
            - _get("refuse_help") * 0.8
        )
        warn = _get("warn_others") + 0.2 * _get("patrol")
        gather = max(0.0, _get("gather_supplies") - 0.15 * (_get("seek_safe_house") + _get("hide")))

        def _clamp01(v: float) -> float:
            return max(0.0, min(1.0, v))

        return {
            "seek_safe_house": round(_clamp01(safe), 3),
            "seek_hospital": round(_clamp01(hospital), 3),
            "help": round(max(-1.0, min(1.0, help_score)), 3),
            "warn": round(_clamp01(warn), 3),
            "gather": round(_clamp01(gather), 3),
        }
