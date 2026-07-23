"""DynaSeqRel-DynEdge: candidate-dependent staging relation scorer.

Version 1 changes one factor relative to SeqRel: goal-space spatial relations
are replaced by staging-space kNN topology and explicit 11-D dynamic geometry.
It intentionally does not call or concatenate SeqRel's legacy ``_pair_context``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import (
    BaseLayoutModel,
    GLOBAL_DIM,
    PART_DIM,
    masked_max,
    masked_mean,
    mlp,
)
from .seqrel_layout_net import _AttnPool
from .. import features as F
from ..dynamic_relations import DYN_EDGE_FEATURE_DIM, DYN_RELATION_MODES

EDGE_ENCODINGS = ("raw_v1", "edge_mlp_v2")
HEAD_MODES = (
    "shared_v1",
    "task_specific_score_v3",
    "frozen_residual_score_adapter_v4",
)


class _DynamicRelationMP(nn.Module):
    """Sparse directed message passing over explicit dynamic edge attributes."""

    def __init__(
        self,
        dim: int,
        edge_dim: int = DYN_EDGE_FEATURE_DIM,
        dropout: float = 0.2,
        edge_encoding: str = "raw_v1",
    ):
        super().__init__()
        if edge_encoding == "raw_v1":
            self.edge_encoder = nn.Identity()
            self.node_msg = None
            self.msg = mlp(
                [2 * dim + edge_dim, dim, dim], last_act=True)
        elif edge_encoding == "edge_mlp_v2":
            # Separate additive node/edge branches prevent small
            # table-normalized geometry values from being dominated by node
            # states in one shared first linear layer.
            self.node_msg = mlp(
                [2 * dim, dim, dim], last_act=True)
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Linear(32, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
            )
            self.msg = None
        else:
            raise ValueError(
                f"unknown edge_encoding={edge_encoding!r}; "
                f"expected one of {EDGE_ENCODINGS}")
        self.upd = mlp([2 * dim, dim, dim], last_act=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_valid: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index[0].long(), edge_index[1].long()
        encoded_edge = self.edge_encoder(edge_feat)
        node_pair = torch.cat(
            [h_flat[source], h_flat[target]], dim=-1)
        if self.node_msg is None:
            messages = self.msg(torch.cat(
                [node_pair, encoded_edge], dim=-1))
        else:
            messages = self.node_msg(node_pair) + encoded_edge
        messages = messages * node_valid[source].unsqueeze(-1)

        aggregate = torch.zeros_like(h_flat)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros(
            h_flat.shape[0], device=h_flat.device, dtype=h_flat.dtype)
        degree.index_add_(0, target, node_valid[source].to(h_flat.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)

        update = self.upd(torch.cat([h_flat, aggregate], dim=-1))
        result = self.norm(h_flat + self.drop(update))
        return result * node_valid.unsqueeze(-1)


class DynaSeqRelDynEdgeLayoutNet(BaseLayoutModel):
    """SeqRel-compatible scorer using staging-space dynamic relations."""

    is_generator = False

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        num_fail_classes: int = F.NUM_FAIL_CLASSES,
        relation_mode: str = "staging_dynedge",
        dynamic_k_spatial: int = 2,
        edge_encoding: str = "raw_v1",
        head_mode: str = "shared_v1",
        adapter_hidden_dim: int = 64,
        adapter_dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        if relation_mode not in DYN_RELATION_MODES:
            raise ValueError(
                f"unknown relation_mode={relation_mode!r}; "
                f"expected one of {DYN_RELATION_MODES}")
        if int(layers) != 2:
            raise ValueError(
                "DynaSeqRel-DynEdge v1 fixes message-passing layers=2")
        if int(dynamic_k_spatial) != 2:
            raise ValueError(
                "DynaSeqRel-DynEdge v1 fixes dynamic_k_spatial=2")
        if edge_encoding not in EDGE_ENCODINGS:
            raise ValueError(
                f"unknown edge_encoding={edge_encoding!r}; "
                f"expected one of {EDGE_ENCODINGS}")
        if head_mode not in HEAD_MODES:
            raise ValueError(
                f"unknown head_mode={head_mode!r}; "
                f"expected one of {HEAD_MODES}")

        self.hidden = int(hidden)
        self.relation_mode = relation_mode
        self.dynamic_k_spatial = int(dynamic_k_spatial)
        self.edge_encoding = edge_encoding
        self.head_mode = head_mode
        self.dynamic_edge_dim = DYN_EDGE_FEATURE_DIM
        self.adapter_hidden_dim = int(adapter_hidden_dim)
        self.adapter_dropout = float(adapter_dropout)
        self.encoder = mlp(
            [PART_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.mp = nn.ModuleList([
            _DynamicRelationMP(
                hidden, DYN_EDGE_FEATURE_DIM, dropout=dropout,
                edge_encoding=edge_encoding)
            for _ in range(2)
        ])
        self.attn_pool = _AttnPool(hidden)
        pool_dim = 3 * hidden + GLOBAL_DIM
        self.trunk = mlp(
            [pool_dim, hidden, hidden], last_act=True, dropout=dropout)
        if head_mode == "task_specific_score_v3":
            # Score regression no longer has to share its only trunk with the
            # much larger classification loss.  Encoder/relations/pooling and
            # the final score head remain unchanged.
            self.score_trunk = mlp(
                [pool_dim, hidden, hidden],
                last_act=True, dropout=dropout)
        self.feas_head = nn.Linear(hidden, 1)
        self.score_head = nn.Linear(hidden, 1)
        self.fail_head = nn.Linear(hidden, int(num_fail_classes))
        if head_mode == "frozen_residual_score_adapter_v4":
            self.score_adapter = nn.Sequential(
                nn.Linear(hidden, self.adapter_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.adapter_dropout),
                nn.Linear(self.adapter_hidden_dim, 1),
            )
            nn.init.zeros_(self.score_adapter[-1].weight)
            nn.init.zeros_(self.score_adapter[-1].bias)
            self.freeze_base_for_score_adapter()

    def frozen_base_modules(self) -> Dict[str, nn.Module]:
        """Modules inherited from the v2 scorer and frozen by adapter-v4."""
        return {
            "encoder": self.encoder,
            "mp": self.mp,
            "attn_pool": self.attn_pool,
            "trunk": self.trunk,
            "feas_head": self.feas_head,
            "score_head": self.score_head,
            "fail_head": self.fail_head,
        }

    def freeze_base_for_score_adapter(self) -> None:
        if self.head_mode != "frozen_residual_score_adapter_v4":
            raise RuntimeError("base freezing is only valid for adapter-v4")
        for module in self.frozen_base_modules().values():
            module.requires_grad_(False)
            module.eval()
        self.score_adapter.requires_grad_(True)

    def train(self, mode: bool = True):
        """Keep every frozen v2 module deterministic during adapter tuning."""
        super().train(mode)
        if self.head_mode == "frozen_residual_score_adapter_v4":
            for module in self.frozen_base_modules().values():
                module.eval()
            self.score_adapter.train(mode)
        return self

    def _effective_edge_feat(
        self, edge_feat: torch.Tensor
    ) -> torch.Tensor:
        if self.relation_mode == "staging_dynedge":
            return edge_feat
        # Topology-only ablation keeps relation flags and geometry validity, but
        # removes all seven continuous geometric attributes.
        effective = edge_feat.clone()
        effective[..., 0:7] = 0.0
        return effective

    def _pooled_context(
        self,
        batch: Dict[str, torch.Tensor],
        edge_feat_override: Optional[torch.Tensor] = None,
        edge_index_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node = batch["node_feat"]
        mask = batch["node_mask"]
        batch_size, max_nodes, _ = node.shape
        edge_index = (
            batch["dynamic_edge_index"]
            if edge_index_override is None else edge_index_override
        )
        edge_feat = (
            batch["dynamic_edge_feat"]
            if edge_feat_override is None else edge_feat_override
        )
        edge_feat = self._effective_edge_feat(edge_feat)

        h = self.encoder(node) * mask.unsqueeze(-1)
        h_flat = h.reshape(batch_size * max_nodes, self.hidden)
        node_valid = mask.reshape(-1)
        for layer in self.mp:
            h_flat = layer(
                h_flat, edge_index, edge_feat, node_valid)
        h = h_flat.reshape(batch_size, max_nodes, self.hidden)

        return torch.cat([
            masked_mean(h, mask),
            masked_max(h, mask),
            self.attn_pool(h, mask),
            batch["global_feat"],
        ], dim=-1)

    def encode(
        self,
        batch: Dict[str, torch.Tensor],
        edge_feat_override: Optional[torch.Tensor] = None,
        edge_index_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the shared feasibility/fail representation."""
        pooled = self._pooled_context(
            batch, edge_feat_override, edge_index_override)
        return self.trunk(pooled)

    def _heads(
        self,
        encoded: torch.Tensor,
        score_encoded: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        score_encoded = encoded if score_encoded is None else score_encoded
        score_logit = self.score_head(score_encoded).squeeze(-1)
        return {
            "feas_logit": self.feas_head(encoded).squeeze(-1),
            "score_pred": torch.sigmoid(score_logit),
            "fail_logits": self.fail_head(encoded),
        }

    def _adapter_heads(
        self,
        encoded: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_score_logit = self.score_head(encoded).squeeze(-1)
        delta_score_logit = self.score_adapter(encoded).squeeze(-1)
        score_logit = base_score_logit + delta_score_logit
        return {
            "feas_logit": self.feas_head(encoded).squeeze(-1),
            "score_pred": torch.sigmoid(score_logit),
            "fail_logits": self.fail_head(encoded),
        }

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pooled = self._pooled_context(batch)
        encoded = self.trunk(pooled)
        if self.head_mode == "frozen_residual_score_adapter_v4":
            return self._adapter_heads(encoded)
        score_encoded = (
            self.score_trunk(pooled)
            if self.head_mode == "task_specific_score_v3" else encoded)
        return self._heads(encoded, score_encoded)

    def forward_debug(
        self,
        batch: Dict[str, torch.Tensor],
        edge_feat_override: Optional[torch.Tensor] = None,
        edge_index_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward with explicit relation overrides for sensitivity checks."""
        pooled = self._pooled_context(
            batch,
            edge_feat_override=edge_feat_override,
            edge_index_override=edge_index_override,
        )
        encoded = self.trunk(pooled)
        score_encoded = (
            self.score_trunk(pooled)
            if self.head_mode == "task_specific_score_v3" else encoded)
        if self.head_mode == "frozen_residual_score_adapter_v4":
            base_score_logit = self.score_head(encoded).squeeze(-1)
            delta_score_logit = self.score_adapter(encoded).squeeze(-1)
            return {
                **self._adapter_heads(encoded),
                "encoded": encoded,
                "score_encoded": encoded,
                "base_score_logit": base_score_logit,
                "delta_score_logit": delta_score_logit,
                "score_logit": base_score_logit + delta_score_logit,
                "dynamic_edge_feat_used": self._effective_edge_feat(
                    batch["dynamic_edge_feat"]
                    if edge_feat_override is None else edge_feat_override),
            }
        score_logit = self.score_head(score_encoded).squeeze(-1)
        return {
            **self._heads(encoded, score_encoded),
            "encoded": encoded,
            "score_encoded": score_encoded,
            "score_logit": score_logit,
            "dynamic_edge_feat_used": self._effective_edge_feat(
                batch["dynamic_edge_feat"]
                if edge_feat_override is None else edge_feat_override),
        }
