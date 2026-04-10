from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .dataset import FeatureSchema


def _embedding_dim(cardinality: int) -> int:
    if cardinality <= 4:
        return 2
    if cardinality <= 8:
        return 4
    return min(8, max(4, cardinality // 2))


class ConditionBPolicyModel(nn.Module):
    def __init__(self, schema: FeatureSchema, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        self.schema = schema
        self.hidden_dims = list(hidden_dims)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(schema.categorical_vocabularies[key]), _embedding_dim(len(schema.categorical_vocabularies[key])))
                for key in schema.categorical_keys
            ]
        )
        embedding_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_dim = schema.numeric_dim + embedding_dim
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            last_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(last_dim, len(schema.action_vocab))
        self.latent_head = nn.Linear(last_dim, schema.latent_dim)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = []
        for idx, embedding in enumerate(self.embeddings):
            embedded.append(embedding(categorical[:, idx]))
        if embedded:
            features = torch.cat([numeric, *embedded], dim=-1)
        else:
            features = numeric
        hidden = self.backbone(features)
        return self.action_head(hidden), self.latent_head(hidden)


PREDICTION_HEAD_NAMES = [
    "ghost_nearby_t3",
    "my_death_t5",
    "health_drop_t3",
    "nearby_death_t5",
    "help_success_t5",
    "refusal_received_t5",
    "tie_increase_t5",
    "shelter_achieved_t2",
    "storm_onset_t3",
    "scarcity_t5",
    "graph_tension_increase_t3",
    "valence_t5",
]
LATENT_DIM = 8


class ConditionDPredictiveModel(nn.Module):
    """Predictive emotion model for condition_d.

    Architecture:
    - Shared encoder backbone (scenario-excluded from latent path)
    - 8-dim latent bottleneck — all 12 prediction heads read from this only
    - Action head reads from the full 128-dim backbone hidden state AND
      scenario embedding (so policy can use scenario info without contaminating
      the emergent latent representation)
    - 11 binary sigmoid heads + 1 regression tanh head (valence)
    """

    LATENT_DIM = 8

    def __init__(self, schema: FeatureSchema, hidden_dims: Sequence[int] = (192, 128)):
        super().__init__()
        self.schema = schema
        self.hidden_dims = list(hidden_dims)

        # Identify scenario index within categorical_keys
        self._scenario_cat_idx = list(schema.categorical_keys).index("scenario") if "scenario" in schema.categorical_keys else -1

        # Separate embeddings: non-scenario (go into latent path) and scenario (action head only)
        self.embeddings_no_scenario = nn.ModuleList()
        self.scenario_embedding: nn.Embedding | None = None
        self._non_scenario_cat_indices: list[int] = []
        scenario_embed_dim = 0

        for cat_idx, key in enumerate(schema.categorical_keys):
            vocab_size = len(schema.categorical_vocabularies[key])
            edim = _embedding_dim(vocab_size)
            if key == "scenario":
                self.scenario_embedding = nn.Embedding(vocab_size, edim)
                scenario_embed_dim = edim
            else:
                self.embeddings_no_scenario.append(nn.Embedding(vocab_size, edim))
                self._non_scenario_cat_indices.append(cat_idx)

        # Dimension of features going into the backbone (no scenario)
        no_scenario_embed_dim = sum(e.embedding_dim for e in self.embeddings_no_scenario)
        backbone_input_dim = schema.numeric_dim + no_scenario_embed_dim

        # Shared encoder backbone
        layers: list[nn.Module] = []
        last_dim = backbone_input_dim
        for h in self.hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            last_dim = h
        self.encoder = nn.Sequential(*layers)
        self._encoder_out_dim = last_dim  # 128 by default

        # 8-dim emergent latent bottleneck
        self.latent_bottleneck = nn.Linear(self._encoder_out_dim, self.LATENT_DIM)

        # 12 prediction heads — all from 8-dim latent
        binary_head_names = [n for n in PREDICTION_HEAD_NAMES if n != "valence_t5"]
        self.binary_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(self.LATENT_DIM, 1), nn.Sigmoid())
            for name in binary_head_names
        })
        self.valence_head = nn.Sequential(nn.Linear(self.LATENT_DIM, 1), nn.Tanh())

        # Action head — reads from full encoder hidden + scenario embedding
        action_input_dim = self._encoder_out_dim + scenario_embed_dim
        self.action_head = nn.Linear(action_input_dim, len(schema.action_vocab))

    def _encode(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (backbone_hidden, scenario_embed). Scenario is excluded from backbone."""
        embedded_no_scenario = []
        for list_idx, cat_idx in enumerate(self._non_scenario_cat_indices):
            embedded_no_scenario.append(self.embeddings_no_scenario[list_idx](categorical[:, cat_idx]))

        scenario_embed = torch.zeros(numeric.shape[0], 0, device=numeric.device)
        if self.scenario_embedding is not None and self._scenario_cat_idx >= 0:
            scenario_embed = self.scenario_embedding(categorical[:, self._scenario_cat_idx])

        if embedded_no_scenario:
            features = torch.cat([numeric, *embedded_no_scenario], dim=-1)
        else:
            features = numeric

        hidden = self.encoder(features)
        return hidden, scenario_embed

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with keys: action_logits, latent, + all 12 prediction head names."""
        hidden, scenario_embed = self._encode(numeric, categorical)

        # 8-dim latent bottleneck
        latent = self.latent_bottleneck(hidden)

        # Prediction heads (all from latent only)
        outputs: dict[str, torch.Tensor] = {"latent": latent}
        for name, head in self.binary_heads.items():
            outputs[name] = head(latent).squeeze(-1)
        outputs["valence_t5"] = self.valence_head(latent).squeeze(-1)

        # Action head (from hidden + scenario)
        if scenario_embed.shape[-1] > 0:
            action_input = torch.cat([hidden, scenario_embed], dim=-1)
        else:
            action_input = hidden
        outputs["action_logits"] = self.action_head(action_input)

        return outputs


class ConditionCDynamicsModel(nn.Module):
    def __init__(self, schema: FeatureSchema, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        self.schema = schema
        self.hidden_dims = list(hidden_dims)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(len(schema.categorical_vocabularies[key]), _embedding_dim(len(schema.categorical_vocabularies[key])))
                for key in schema.categorical_keys
            ]
        )
        embedding_dim = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_dim = schema.numeric_dim + embedding_dim
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            last_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.next_latent_head = nn.Linear(last_dim, schema.latent_dim)
        self.action_head = nn.Linear(last_dim, len(schema.action_vocab))

    def encode(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = []
        for idx, embedding in enumerate(self.embeddings):
            embedded.append(embedding(categorical[:, idx]))
        if embedded:
            features = torch.cat([numeric, *embedded], dim=-1)
        else:
            features = numeric
        return self.encoder(features)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(numeric, categorical)
        return self.action_head(hidden), self.next_latent_head(hidden)
