from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Dict, Sequence

import torch

from ghost_town.types import AffectOutput

from .dataset import FeatureSchema
from .model import ConditionBPolicyModel

POSITIVE_RELATIONSHIP_LABELS = {
    "allies",
    "partners",
    "medical_team",
    "mentor_pair",
    "friends",
    "command_pair",
    "household",
}
NEGATIVE_RELATIONSHIP_LABELS = {"rivals", "strained"}


class ConditionBInferenceBackend:
    def __init__(
        self,
        checkpoint_path: str | Path,
        schema_path: str | Path,
        device: str | None = None,
        policy_mode: str = "deterministic",
        temperature: float = 1.0,
        rng_seed: int | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.schema_path = Path(schema_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy_mode = policy_mode
        self.temperature = max(0.05, float(temperature))
        self.rng = Random(rng_seed)
        self.schema = FeatureSchema.load(self.schema_path)
        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        hidden_dims = payload.get("hidden_dims", [128, 64])
        self.model = ConditionBPolicyModel(self.schema, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def compute(
        self,
        observation: Dict[str, object],
        social_context: Dict[str, object],
        prev_latent: Sequence[float],
        metadata: Dict[str, object],
    ) -> AffectOutput:
        calibration = self._build_calibration_context(observation, social_context, metadata)
        numeric, categorical = self.schema.encode_runtime(observation, social_context, prev_latent, metadata)
        numeric = numeric.to(self.device)
        categorical = categorical.to(self.device)
        with torch.no_grad():
            action_logits, latent_prediction = self.model(numeric, categorical)
            action_probs = torch.softmax(action_logits / self.temperature, dim=-1)[0].detach().cpu().tolist()
            latent_vector = latent_prediction[0].detach().cpu().tolist()
        selected_action = self._select_action(action_probs) if self.policy_mode == "sample" else None
        action_bias = self._action_bias_from_probs(action_probs, calibration, selected_action=selected_action)
        affect_vector = self._affect_from_latent(latent_vector, calibration)
        top_index = int(max(range(len(action_probs)), key=lambda idx: action_probs[idx]))
        top_action = self.schema.action_vocab[top_index]
        top_pairs = sorted(
            zip(self.schema.action_vocab, action_probs),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        return AffectOutput(
            label="condition_b_trained",
            affect_vector=affect_vector,
            action_bias=action_bias,
            prompt_context={
                "tone": "learned",
                "summary": f"trained policy top_action={top_action} p={action_probs[top_index]:.2f}",
            },
            latent_vector=[round(float(value), 3) for value in latent_vector],
            probe_data={
                "policy_source": "trained_checkpoint",
                "policy_mode": self.policy_mode,
                "selected_action": selected_action or "",
                "top_actions": [[name, round(float(prob), 4)] for name, prob in top_pairs],
                "checkpoint": str(self.checkpoint_path),
                "schema": str(self.schema_path),
                "calibration": {key: round(float(value), 4) if isinstance(value, float) else value for key, value in calibration.items()},
            },
        )

    def _select_action(self, probabilities: Sequence[float]) -> str:
        if self.policy_mode != "sample":
            top_index = int(max(range(len(probabilities)), key=lambda idx: probabilities[idx]))
            return self.schema.action_vocab[top_index]
        threshold = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if threshold <= cumulative:
                return self.schema.action_vocab[index]
        return self.schema.action_vocab[-1]

    def _build_calibration_context(
        self,
        observation: Dict[str, object],
        social_context: Dict[str, object],
        metadata: Dict[str, object],
    ) -> Dict[str, float | bool | str]:
        time_of_day = str(metadata.get("time_of_day", "day") or "day")
        visible_ghosts = float(observation.get("visible_ghosts", 0) or 0.0)
        visible_deaths = float(observation.get("visible_deaths", 0) or 0.0)
        nearest_refuge_distance = float(observation.get("nearest_refuge_distance", 4) or 4.0)
        in_shelter = bool(observation.get("in_shelter", False))
        storm = bool(metadata.get("storm", False))
        rescue_opportunity = bool(metadata.get("rescue_opportunity", False))
        refusal_opportunity = bool(metadata.get("refusal_opportunity", False))
        rival_refusal_opportunity = bool(metadata.get("rival_refusal_opportunity", False))
        graph_tension = float(social_context.get("graph_tension", 0.0) or 0.0)
        graph_support = float(social_context.get("graph_support", 0.0) or 0.0)
        relationship_label = str(metadata.get("relationship_label_at_decision", "neighbors") or "neighbors")
        tie_value = float(metadata.get("tie_value_at_decision", 0.0) or 0.0)
        scarcity_signal = 1.0 if float(metadata.get("health", 100.0) or 100.0) < 55.0 else 0.0

        visible_hazard = visible_ghosts > 0 or visible_deaths > 0
        night_pressure = 1.0 if time_of_day in {"sunset", "night"} else 0.0
        rival_context = relationship_label in NEGATIVE_RELATIONSHIP_LABELS or rival_refusal_opportunity
        trusted_context = relationship_label in POSITIVE_RELATIONSHIP_LABELS or tie_value > 0.55 or graph_support > 0.35
        hazard_score = (
            visible_ghosts * 0.45
            + visible_deaths * 0.35
            + (0.18 if storm else 0.0)
            + night_pressure * 0.2
            + (0.08 if not in_shelter else 0.0)
            + max(0.0, (3.0 - nearest_refuge_distance) * 0.03)
            + graph_tension * 0.2
            + scarcity_signal * 0.05
        )
        non_hostile = (
            hazard_score < 0.24
            and not visible_hazard
            and not storm
            and not rescue_opportunity
            and not refusal_opportunity
            and not rival_context
            and time_of_day == "day"
        )
        return {
            "time_of_day": time_of_day,
            "hazard_score": hazard_score,
            "non_hostile": non_hostile,
            "visible_hazard": visible_hazard,
            "storm": storm,
            "rescue_opportunity": rescue_opportunity,
            "refusal_opportunity": refusal_opportunity,
            "rival_context": rival_context,
            "trusted_context": trusted_context,
            "graph_tension": graph_tension,
            "graph_support": graph_support,
        }

    def _action_bias_from_probs(
        self,
        probabilities: Sequence[float],
        calibration: Dict[str, float | bool | str],
        selected_action: str | None = None,
    ) -> Dict[str, float]:
        lookup = {action: float(probabilities[idx]) for idx, action in enumerate(self.schema.action_vocab)}
        hazard_score = float(calibration["hazard_score"])
        non_hostile = bool(calibration["non_hostile"])
        refusal_opportunity = bool(calibration["refusal_opportunity"])
        rescue_opportunity = bool(calibration["rescue_opportunity"])
        rival_context = bool(calibration["rival_context"])
        trusted_context = bool(calibration["trusted_context"])
        graph_tension = float(calibration["graph_tension"])

        safe_scale = 0.62 if non_hostile else min(1.15, 0.82 + hazard_score * 0.45)
        safe = (lookup.get("seek_safe_house", 0.0) + 0.75 * lookup.get("hide", 0.0)) * safe_scale
        hospital = lookup.get("seek_hospital", 0.0)
        refusal_weight = 1.0 if refusal_opportunity else 0.0
        if rival_context:
            refusal_weight = max(refusal_weight, 0.6)
        refusal_penalty = lookup.get("refuse_help", 0.0) * refusal_weight
        sharing_weight = 0.85 if non_hostile and trusted_context else 0.65
        help_bonus = 0.1 if trusted_context else 0.0
        if rescue_opportunity:
            help_bonus += 0.1
        help_score = (
            lookup.get("help_other", 0.0) * (1.05 if trusted_context else 1.0)
            + sharing_weight * lookup.get("share_supplies", 0.0)
            + help_bonus
            - refusal_penalty
        )
        if non_hostile:
            help_score = max(help_score, -0.08)
        warn_scale = 0.6 if non_hostile else min(1.15, 0.8 + hazard_score * 0.35)
        warn = (lookup.get("warn_others", 0.0) + 0.2 * lookup.get("patrol", 0.0)) * warn_scale
        gather = max(
            0.0,
            lookup.get("gather_supplies", 0.0)
            - 0.18 * (lookup.get("seek_safe_house", 0.0) + lookup.get("hide", 0.0))
            - 0.1 * refusal_penalty
        )
        if non_hostile and graph_tension < 0.2:
            gather = min(1.0, gather + 0.08)
        bias = {
            "seek_safe_house": round(min(1.0, max(0.0, safe)), 3),
            "seek_hospital": round(min(1.0, max(0.0, hospital)), 3),
            "help": round(min(1.0, max(-1.0, help_score)), 3),
            "warn": round(min(1.0, max(0.0, warn)), 3),
            "gather": round(min(1.0, max(0.0, gather)), 3),
        }
        if selected_action:
            bias = self._blend_selected_action_bias(bias, selected_action)
        return bias

    def _blend_selected_action_bias(self, bias: Dict[str, float], selected_action: str) -> Dict[str, float]:
        anchored = dict(bias)
        if selected_action in {"seek_safe_house", "hide"}:
            anchored["seek_safe_house"] = max(anchored["seek_safe_house"], 0.95)
        elif selected_action == "seek_hospital":
            anchored["seek_hospital"] = max(anchored["seek_hospital"], 0.95)
        elif selected_action in {"help_other", "share_supplies"}:
            anchored["help"] = max(anchored["help"], 0.9)
        elif selected_action == "refuse_help":
            anchored["help"] = min(anchored["help"], -0.9)
        elif selected_action in {"warn_others", "patrol"}:
            anchored["warn"] = max(anchored["warn"], 0.9)
        elif selected_action == "gather_supplies":
            anchored["gather"] = max(anchored["gather"], 0.9)
        elif selected_action in {"rest", "routine"}:
            anchored["seek_safe_house"] *= 0.65
            anchored["warn"] *= 0.65
        return {
            "seek_safe_house": round(min(1.0, max(0.0, anchored["seek_safe_house"])), 3),
            "seek_hospital": round(min(1.0, max(0.0, anchored["seek_hospital"])), 3),
            "help": round(min(1.0, max(-1.0, anchored["help"])), 3),
            "warn": round(min(1.0, max(0.0, anchored["warn"])), 3),
            "gather": round(min(1.0, max(0.0, anchored["gather"])), 3),
        }

    def _affect_from_latent(
        self,
        latent_vector: Sequence[float],
        calibration: Dict[str, float | bool | str],
    ) -> Dict[str, float]:
        latent = list(latent_vector[:8]) + [0.0] * max(0, 8 - len(latent_vector))
        non_hostile = bool(calibration["non_hostile"])
        rival_context = bool(calibration["rival_context"])
        trusted_context = bool(calibration["trusted_context"])
        hazard_score = float(calibration["hazard_score"])
        graph_support = float(calibration.get("graph_support", 0.0))
        if non_hostile:
            support_relief = 0.08 if trusted_context or graph_support > 0.35 else 0.0
            fear_scale = max(0.34, 0.46 - support_relief)
            stress_scale = max(0.38, 0.52 - support_relief)
        else:
            fear_scale = min(1.05, 0.92 + hazard_score * 0.18)
            stress_scale = min(1.05, 0.94 + hazard_score * 0.12)
        suspicion_scale = 0.48 if non_hostile and not rival_context else (1.0 if rival_context else 0.85)
        return {
            "fear": round(min(1.0, max(0.0, float(latent[0]) * fear_scale)), 3),
            "stress": round(min(1.0, max(0.0, float(latent[1]) * stress_scale)), 3),
            "trust": round(min(1.0, max(0.0, float(latent[2]))), 3),
            "grief": round(min(1.0, max(0.0, float(latent[3]))), 3),
            "suspicion": round(min(1.0, max(0.0, float(latent[4]) * suspicion_scale)), 3),
            "relief": round(min(1.0, max(0.0, float(latent[5]))), 3),
        }
