"""RelSeqGen: Relational Sequential Layout Generator.

Independent generator model.  Does not import DynEdge private classes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .base import (
    BaseLayoutModel,
    GLOBAL_DIM,
    PART_DIM,
    ScorerHead,
    mlp,
    masked_global,
    masked_mean,
    static_only,
)
from .. import features as F
from ..geometry_features import GEO_FEATURE_DIM
from ..generator_relations import (
    MAX_POSE_CANDIDATES,
    MAX_ROTATIONS,
    PARTIAL_DYN_EDGE_DIM,
    POSE_CAND_DIM,
    STATIC_EDGE_DIM,
    build_partial_dynamic_edge_feature_torch,
)


class _StaticMP(nn.Module):
    def __init__(self, dim: int, edge_dim: int = STATIC_EDGE_DIM, dropout: float = 0.15):
        super().__init__()
        self.msg = mlp([dim + edge_dim, dim, dim], last_act=True, dropout=dropout)
        self.upd = mlp([2 * dim, dim, dim], last_act=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_feat: torch.Tensor, adj: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        m = self.msg(torch.cat([h_j, edge_feat], dim=-1))
        gate = (adj * mask.unsqueeze(1)).unsqueeze(-1)
        m = m * gate
        deg = gate.sum(dim=2).clamp_min(1.0)
        agg = m.sum(dim=2) / deg
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class RelSeqGenLayoutNet(BaseLayoutModel):
    """Relational Sequential Layout Generator."""

    is_generator = True

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 2,
        heads: int = 2,
        mixtures: int = 5,
        dropout: float = 0.15,
        pose_mode: str = "predict",
        geo_dim: int = GEO_FEATURE_DIM,
        pose_cand_dim: int = POSE_CAND_DIM,
        max_pose: int = MAX_POSE_CANDIDATES,
        max_rot: int = MAX_ROTATIONS,
        **_,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.heads = int(heads)
        self.mixtures = int(mixtures)
        self.pose_mode = str(pose_mode)
        self.max_pose = int(max_pose)
        self.max_rot = int(max_rot)

        self.geo_enc = mlp([geo_dim, hidden, hidden], last_act=True, dropout=dropout)
        self.pose_enc = mlp([pose_cand_dim, hidden, hidden], last_act=True, dropout=dropout)
        self.part_enc = mlp(
            [PART_DIM + hidden + hidden, hidden, hidden], last_act=True, dropout=dropout)
        self.static_mp = nn.ModuleList([
            _StaticMP(hidden, STATIC_EDGE_DIM, dropout=dropout) for _ in range(layers)
        ])
        self.ctx_proj = mlp([2 * hidden + GLOBAL_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.station_mean = mlp([hidden, hidden, 2], last_act=False)
        self.station_logstd = mlp([hidden, hidden, 2], last_act=False)
        self.ar_cell = mlp(
            [hidden + hidden + PARTIAL_DYN_EDGE_DIM, hidden, hidden],
            last_act=True, dropout=dropout)
        self.pose_head = nn.Linear(hidden, max_pose)
        self.rot_head = nn.Linear(hidden, max_rot)
        self.mix_head = nn.Linear(hidden, mixtures)
        self.xy_mean_head = nn.Linear(hidden, mixtures * 2)
        self.xy_logstd_head = nn.Linear(hidden, mixtures * 2)
        pool_dim = hidden + GLOBAL_DIM
        self.head = ScorerHead(pool_dim, hidden)

    def _encode_parts(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        node = static_only(batch["node_feat"], batch["static_mask"])
        geo = batch.get("geometry_feat")
        if geo is None:
            geo = torch.zeros(
                node.shape[0], node.shape[1], GEO_FEATURE_DIM,
                device=node.device, dtype=node.dtype)
        pose = batch.get("pose_candidate_feat")
        pose_mask = batch.get("pose_candidate_mask")
        if pose is None:
            pose = torch.zeros(
                node.shape[0], node.shape[1], self.max_pose, POSE_CAND_DIM,
                device=node.device, dtype=node.dtype)
            pose_mask = torch.zeros(
                node.shape[0], node.shape[1], self.max_pose,
                device=node.device, dtype=node.dtype)
        geo_h = self.geo_enc(geo)
        pose_h = self.pose_enc(pose)
        pose_pool = (pose_h * pose_mask.unsqueeze(-1)).sum(dim=2) / pose_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        part_h = self.part_enc(torch.cat([node, geo_h, pose_pool], dim=-1))
        mask = batch["node_mask"]
        for layer in self.static_mp:
            part_h = layer(part_h, batch["static_edge_attr"], batch["static_adj"], mask)
        return part_h, mask

    def _global_context(self, part_h: torch.Tensor, mask: torch.Tensor,
                        batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        g = masked_global(batch)
        pooled = masked_mean(part_h, mask)
        ctx = self.ctx_proj(torch.cat([pooled, pooled, g], dim=-1))
        head_in = torch.cat([pooled, g], dim=-1)
        return ctx, head_in

    def _decode_outputs(self, step_h: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, H = step_h.shape
        M = self.mixtures
        mix_logits = self.mix_head(step_h)
        xy_mean = torch.tanh(self.xy_mean_head(step_h).view(B, N, M, 2))
        xy_logstd = self.xy_logstd_head(step_h).view(B, N, M, 2).clamp(-4, 2)
        return {
            "pose_logits": self.pose_head(step_h),
            "rotation_logits": self.rot_head(step_h),
            "mix_logits": mix_logits,
            "xy_mean": xy_mean,
            "xy_logstd": xy_logstd,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        part_h, mask = self._encode_parts(batch)
        ctx, head_in = self._global_context(part_h, mask, batch)
        station_mean = torch.tanh(self.station_mean(ctx))
        station_logstd = self.station_logstd(ctx).clamp(-4, 2)
        B, N, H = part_h.shape
        step_states = []
        # AR layout state for edge features: detached (no in-place grad through history).
        gen_xy_state = torch.zeros(B, N, 2, device=part_h.device, dtype=part_h.dtype)
        generated_pose = torch.zeros(B, N, device=part_h.device, dtype=part_h.dtype)
        is_first = batch.get("is_first_mask", torch.zeros(B, N, device=part_h.device))
        order_idx = batch.get("order_index", torch.arange(N, device=part_h.device).float().view(1, N).expand(B, N))
        footprints = batch.get("selected_footprint", torch.zeros(B, N, 2, device=part_h.device))
        table_bounds = batch["table_bounds"].to(device=part_h.device, dtype=part_h.dtype)
        scale = torch.clamp(
            torch.maximum(
                table_bounds[:, 1] - table_bounds[:, 0],
                table_bounds[:, 3] - table_bounds[:, 2],
            ),
            min=1e-6,
        )

        order_sorted = torch.argsort(order_idx, dim=1)
        batch_ar = torch.arange(B, device=part_h.device)
        for step in range(N):
            rel_feat = torch.zeros(
                B, PARTIAL_DYN_EDGE_DIM, device=part_h.device, dtype=part_h.dtype)
            idx = order_sorted[:, step]
            if step > 0:
                prev_idx = order_sorted[:, step - 1]
                src_xy = gen_xy_state[batch_ar, prev_idx]
                tgt_raw = gen_xy_state[batch_ar, idx]
                tgt_xy = torch.where(
                    (tgt_raw.abs().sum(dim=-1, keepdim=True) > 0).expand_as(tgt_raw),
                    tgt_raw, src_xy)
                rel_feat = build_partial_dynamic_edge_feature_torch(
                    src_xy, tgt_xy,
                    footprints[batch_ar, prev_idx],
                    footprints[batch_ar, idx],
                    torch.ones(B, device=part_h.device, dtype=part_h.dtype),
                    (is_first[batch_ar, idx] < 0.5).to(part_h.dtype),
                    table_bounds,
                    (order_idx[batch_ar, idx] - order_idx[batch_ar, prev_idx]) / max(N, 1),
                    torch.zeros(B, device=part_h.device, dtype=part_h.dtype),
                    scale,
                )
            gather = idx.view(B, 1, 1).expand(B, 1, H)
            part_step = part_h.gather(1, gather).squeeze(1)
            cell_in = torch.cat([part_step, ctx, rel_feat], dim=-1)
            step_h = self.ar_cell(cell_in).unsqueeze(1)
            step_states.append(step_h)
            dec = self._decode_outputs(step_h)
            if self.pose_mode == "predict":
                pose_pick = dec["pose_logits"].argmax(dim=-1).squeeze(1).float()
            elif "target_pose_index" in batch:
                pose_pick = batch["target_pose_index"].gather(1, idx.view(B, 1)).squeeze(1).float()
            else:
                pose_pick = torch.zeros(B, device=part_h.device, dtype=part_h.dtype)
            best_mix = dec["mix_logits"].argmax(dim=-1).squeeze(1)
            xy_pick = dec["xy_mean"].squeeze(1).gather(
                1, best_mix.view(B, 1, 1).expand(B, 1, 2)).squeeze(1)
            gen_xy_state[batch_ar, idx] = xy_pick.detach()
            generated_pose[batch_ar, idx] = pose_pick

        step_h_all = torch.cat(step_states, dim=1)
        out = self._decode_outputs(step_h_all)
        out.update({
            "station_mean": station_mean,
            "station_logstd": station_logstd,
            "feas_logit": self.head(head_in)["feas_logit"],
            "score_pred": self.head(head_in)["score_pred"],
            "xy_pred": out["xy_mean"][:, :, 0, :],
        })
        return out

    @torch.no_grad()
    def propose(self, batch: Dict[str, torch.Tensor], k: int) -> torch.Tensor:
        structured = self.propose_structured(batch, k=k)
        B = batch["node_feat"].shape[0]
        N = batch["node_feat"].shape[1]
        outs = torch.zeros(k, B, N, 2, device=batch["node_feat"].device)
        for ki, prop in enumerate(structured):
            for i, part in enumerate(prop["parts"]):
                outs[ki, 0, i] = torch.as_tensor(
                    part["offset_xy_norm"], device=outs.device, dtype=outs.dtype)
        return outs.squeeze(1)

    @torch.no_grad()
    def propose_structured(
        self,
        batch: Dict[str, torch.Tensor],
        k: int,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[Dict]:
        if seed is not None:
            torch.manual_seed(int(seed))
        out = self.forward(batch)
        B = batch["node_feat"].shape[0]
        N = batch["node_feat"].shape[1]
        proposals: List[Dict] = []
        station_mean = out["station_mean"]
        station_logstd = out["station_logstd"]
        is_first = batch.get("is_first_mask", torch.zeros(B, N, device=station_mean.device))
        order_idx = batch.get(
            "order_index",
            torch.arange(N, device=station_mean.device).float().view(1, N).expand(B, N),
        )
        order_sorted = torch.argsort(order_idx, dim=1)

        for kk in range(k):
            if kk == 0:
                st = station_mean
            else:
                noise = torch.randn_like(station_mean) * temperature
                st = torch.tanh(station_mean + torch.exp(station_logstd) * noise)
            prop = {
                "station_xy_norm": st[0].cpu().numpy(),
                "parts": [],
                "proposal_logprob": 0.0,
            }
            for step in range(N):
                i = int(order_sorted[0, step].item())
                pose_logits = out["pose_logits"][0, step] / max(temperature, 1e-6)
                rot_logits = out["rotation_logits"][0, step] / max(temperature, 1e-6)
                mix_logits = out["mix_logits"][0, step] / max(temperature, 1e-6)
                if bool(is_first[0, i] > 0.5) or (
                        batch.get("xy_valid") is not None
                        and float(batch["xy_valid"][0, i]) < 0.5):
                    prop["parts"].append({
                        "part_index": i,
                        "pose_index": 0,
                        "rotation_index": 0,
                        "offset_xy_norm": [0.0, 0.0],
                        "confidence": 1.0,
                    })
                    continue
                if kk == 0:
                    pose_idx = int(pose_logits.argmax().item())
                    rot_idx = int(rot_logits.argmax().item())
                    mix_idx = int(mix_logits.argmax().item())
                else:
                    pose_idx = int(torch.multinomial(Fn.softmax(pose_logits, dim=-1), 1).item())
                    rot_idx = int(torch.multinomial(Fn.softmax(rot_logits, dim=-1), 1).item())
                    mix_idx = int(torch.multinomial(Fn.softmax(mix_logits, dim=-1), 1).item())
                xy = out["xy_mean"][0, step, mix_idx]
                if kk > 0:
                    std = torch.exp(out["xy_logstd"][0, step, mix_idx])
                    xy = torch.clamp(xy + std * torch.randn_like(xy) * temperature, -1.0, 1.0)
                conf = float(Fn.softmax(mix_logits, dim=-1)[mix_idx].item())
                prop["parts"].append({
                    "part_index": i,
                    "pose_index": pose_idx,
                    "rotation_index": rot_idx,
                    "offset_xy_norm": xy.cpu().numpy().tolist(),
                    "confidence": conf,
                })
            proposals.append(prop)
        return proposals

    @torch.no_grad()
    def predict_station(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        part_h, mask = self._encode_parts(batch)
        ctx, _ = self._global_context(part_h, mask, batch)
        return torch.tanh(self.station_mean(ctx))
