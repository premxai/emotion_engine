from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import torch

from ghost_town.types import AffectOutput

from .dataset import FeatureSchema
from .model import ConditionCDynamicsModel
from .inference import NEGATIVE_RELATIONSHIP_LABELS, POSITIVE_RELATIONSHIP_LABELS


class ConditionCInferenceBackend:
    def __init__(self, checkpoint_path: str | Path, schema_path: str | Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.schema_path = Path(schema_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.schema = FeatureSchema.load(self.schema_path)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        hidden_dims = payload.get("hidden_dims", [128, 64])
        self.model = ConditionCDynamicsModel(self.schema, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

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
            action_logits, next_latent_prediction = self.model(numeric, categorical)
            action_probs = torch.softmax(action_logits, dim=-1)[0].detach().cpu().tolist()
            next_latent = next_latent_prediction[0].detach().cpu().tolist()
        latent_vector = [round(max(0.0, min(1.0, float(value))), 3) for value in next_latent[: self.schema.latent_dim]]
        affect_vector = self._affect_from_latent(latent_vector, observation, social_context, metadata)
        action_bias = self._action_bias_from_probs(action_probs, latent_vector, observation, social_context, metadata)
        top_index = int(max(range(len(action_probs)), key=lambda idx: action_probs[idx]))
        top_action = self.schema.action_vocab[top_index]
        top_pairs = sorted(zip(self.schema.action_vocab, action_probs), key=lambda item: item[1], reverse=True)[:5]
        return AffectOutput(
            label="condition_c_self_supervised",
            affect_vector=affect_vector,
            action_bias=action_bias,
            prompt_context={
                "tone": "adaptive",
                "summary": f"latent-dynamics policy top_action={top_action} p={action_probs[top_index]:.2f}",
            },
            latent_vector=latent_vector,
            probe_data={
                "policy_source": "trained_condition_c_checkpoint",
                "checkpoint": str(self.checkpoint_path),
                "schema": str(self.schema_path),
                "top_actions": [[name, round(float(prob), 4)] for name, prob in top_pairs],
            },
        )

    def _affect_from_latent(
        self,
        latent: Sequence[float],
        observation: Dict[str, object],
        social_context: Dict[str, object],
        metadata: Dict[str, object],
    ) -> Dict[str, float]:
        trusted_context = str(metadata.get("relationship_label_at_decision", "neighbors") or "neighbors") in POSITIVE_RELATIONSHIP_LABELS
        rival_context = str(metadata.get("relationship_label_at_decision", "neighbors") or "neighbors") in NEGATIVE_RELATIONSHIP_LABELS or bool(
            metadata.get("rival_refusal_opportunity", False)
        )
        graph_support = float(social_context.get("graph_support", 0.0) or 0.0)
        graph_tension = float(social_context.get("graph_tension", 0.0) or 0.0)
        visible_ghosts = float(observation.get("visible_ghosts", 0) or 0.0)
        visible_deaths = float(observation.get("visible_deaths", 0) or 0.0)
        shelter_bonus = 0.1 if bool(observation.get("in_shelter", False)) else 0.0
        fear = max(0.0, min(1.0, latent[0] * 0.55 + latent[1] * 0.15 + visible_ghosts * 0.08 + visible_deaths * 0.06 - shelter_bonus))
        stress = max(0.0, min(1.0, latent[1] * 0.45 + latent[3] * 0.15 + latent[6] * 0.2 + graph_tension * 0.15))
        trust = max(0.0, min(1.0, latent[2] * 0.55 + graph_support * 0.2 + (0.08 if trusted_context else 0.0) - graph_tension * 0.05))
        grief = max(0.0, min(1.0, latent[3] * 0.7 + visible_deaths * 0.08))
        suspicion = max(0.0, min(1.0, latent[4] * (1.1 if rival_context else 0.85) + graph_tension * 0.1 - graph_support * 0.05))
        relief = max(0.0, min(1.0, latent[5] * 0.7 + shelter_bonus + graph_support * 0.05 - visible_ghosts * 0.05))
        return {
            "fear": round(fear, 3),
            "stress": round(stress, 3),
            "trust": round(trust, 3),
            "grief": round(grief, 3),
            "suspicion": round(suspicion, 3),
            "relief": round(relief, 3),
        }

    def _action_bias_from_probs(
        self,
        probabilities: Sequence[float],
        latent: Sequence[float],
        observation: Dict[str, object],
        social_context: Dict[str, object],
        metadata: Dict[str, object],
    ) -> Dict[str, float]:
        lookup = {action: float(probabilities[idx]) for idx, action in enumerate(self.schema.action_vocab)}
        rival_context = bool(metadata.get("rival_refusal_opportunity", False)) or str(metadata.get("relationship_label_at_decision", "neighbors")) in NEGATIVE_RELATIONSHIP_LABELS
        trusted_context = str(metadata.get("relationship_label_at_decision", "neighbors")) in POSITIVE_RELATIONSHIP_LABELS or float(
            metadata.get("tie_value_at_decision", 0.0) or 0.0
        ) >= 0.35
        graph_support = float(social_context.get("graph_support", 0.0) or 0.0)
        graph_tension = float(social_context.get("graph_tension", 0.0) or 0.0)
        danger = float(observation.get("visible_ghosts", 0) or 0.0) * 0.35 + float(observation.get("visible_deaths", 0) or 0.0) * 0.2
        safe = lookup.get("seek_safe_house", 0.0) + 0.65 * lookup.get("hide", 0.0) + danger * 0.15 + latent[0] * 0.1
        help_score = (
            lookup.get("help_other", 0.0)
            + 0.7 * lookup.get("share_supplies", 0.0)
            + (0.12 if trusted_context else 0.0)
            - (0.18 if rival_context else 0.0)
            + graph_support * 0.08
            - graph_tension * 0.08
        )
        warn = lookup.get("warn_others", 0.0) + danger * 0.12 + latent[0] * 0.08
        gather = max(0.0, lookup.get("gather_supplies", 0.0) + latent[6] * 0.1 - safe * 0.18)
        return {
            "seek_safe_house": round(min(1.0, max(0.0, safe)), 3),
            "seek_hospital": round(min(1.0, max(0.0, lookup.get("seek_hospital", 0.0) + latent[6] * 0.08)), 3),
            "help": round(min(1.0, max(-1.0, help_score)), 3),
            "warn": round(min(1.0, max(0.0, warn)), 3),
            "gather": round(min(1.0, max(0.0, gather)), 3),
        }
