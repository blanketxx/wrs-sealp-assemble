"""RelSeqGen composite training losses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as Fn

from .losses import LossWeights, _focal_bce


@dataclass
class RelSeqGenLossWeights:
    lambda_station: float = 1.0
    lambda_pose: float = 1.0
    lambda_rot: float = 0.5
    lambda_xy: float = 1.0
    lambda_boundary: float = 0.2
    lambda_overlap: float = 0.2
    lambda_keepout: float = 0.1
    lambda_diversity: float = 0.05
    lambda_cls: float = 1.0
    lambda_score: float = 0.5
    elite_quantile: float = 0.70
    score_threshold: float = 0.0


def _elite_mask(batch: Dict[str, torch.Tensor], weights: RelSeqGenLossWeights) -> torch.Tensor:
    feas = batch["feas"] > 0.5
    score = batch["score"]
    if weights.score_threshold > 0:
        return feas & (score >= weights.score_threshold)
    if feas.any():
        thr = torch.quantile(score[feas], weights.elite_quantile)
        return feas & (score >= thr)
    return feas


def _gaussian_nll(mean: torch.Tensor, logstd: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * logstd).clamp_min(1e-6)
    return 0.5 * (((target - mean) ** 2) / var + 2.0 * logstd + math.log(2 * math.pi))


def _gmm_nll(
    mix_logits: torch.Tensor,
    xy_mean: torch.Tensor,
    xy_logstd: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """mix_logits [B,N,M], mean/logstd [B,N,M,2], target [B,N,2]."""
    B, N, M, _ = xy_mean.shape
    log_pi = Fn.log_softmax(mix_logits, dim=-1)
    tgt = target.unsqueeze(2).expand(B, N, M, 2)
    nll = _gaussian_nll(xy_mean, xy_logstd, tgt).sum(-1)
    log_prob = torch.logsumexp(log_pi + (-0.5 * nll), dim=-1)
    denom = valid.sum().clamp_min(1.0)
    return -(log_prob * valid).sum() / denom


def _soft_boundary_penalty(
    xy_mean: torch.Tensor,
    footprint: torch.Tensor,
    table_bounds: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Penalize predicted centers outside table using footprint."""
    xlo = table_bounds[:, 0:1]
    xhi = table_bounds[:, 1:2]
    ylo = table_bounds[:, 2:3]
    yhi = table_bounds[:, 3:4]
    hx = footprint[:, :, 0] / 2.0
    hy = footprint[:, :, 1] / 2.0
    left = Fn.relu(xlo + hx - xy_mean[:, :, 0])
    right = Fn.relu(xy_mean[:, :, 0] + hx - xhi)
    bottom = Fn.relu(ylo + hy - xy_mean[:, :, 1])
    top = Fn.relu(xy_mean[:, :, 1] + hy - yhi)
    pen = (left + right + bottom + top) * valid
    return pen.sum() / valid.sum().clamp_min(1.0)


def _soft_overlap_penalty(
    xy_mean: torch.Tensor,
    footprint: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    B, N, _ = xy_mean.shape
    total = torch.zeros((), device=xy_mean.device)
    count = 0
    for b in range(B):
        idx = torch.nonzero(valid[b] > 0.5, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        for i in range(idx.numel()):
            for j in range(i + 1, idx.numel()):
                a, c = int(idx[i]), int(idx[j])
                d = xy_mean[b, c] - xy_mean[b, a]
                need_x = (footprint[b, a, 0] + footprint[b, c, 0]) / 2.0
                need_y = (footprint[b, a, 1] + footprint[b, c, 1]) / 2.0
                ox = Fn.relu(need_x - d[0].abs())
                oy = Fn.relu(need_y - d[1].abs())
                total = total + ox + oy
                count += 1
    return total / max(count, 1)


def compute_relseqgen_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    scorer_weights: LossWeights,
    gen_weights: RelSeqGenLossWeights,
) -> Dict[str, torch.Tensor]:
    device = out["feas_logit"].device
    feas = batch["feas"]
    score = batch["score"]
    elite = _elite_mask(batch, gen_weights)
    xy_valid = batch["xy_valid"]
    teach_mask = xy_valid * elite.float().unsqueeze(1)

    pw = torch.as_tensor(float(scorer_weights.pos_weight), device=device)
    if scorer_weights.use_focal:
        l_cls = _focal_bce(out["feas_logit"], feas, pw, scorer_weights.focal_gamma)
    else:
        l_cls = Fn.binary_cross_entropy_with_logits(out["feas_logit"], feas, pos_weight=pw)

    feas_mask = feas > 0.5
    if feas_mask.any():
        l_score = Fn.smooth_l1_loss(out["score_pred"][feas_mask], score[feas_mask])
    else:
        l_score = torch.zeros((), device=device)

    # station
    if elite.any():
        st_mask = elite.float() * batch.get("station_valid", torch.ones_like(elite))
        denom_s = st_mask.sum().clamp_min(1.0)
        l_station = (
            _gaussian_nll(
                out["station_mean"], out["station_logstd"], batch["station_target"]
            ).sum(-1) * st_mask
        ).sum() / denom_s
    else:
        l_station = torch.zeros((), device=device)

    # pose / rotation CE
    if "pose_logits" in out and "target_pose_index" in batch:
        pose_logits = out["pose_logits"]
        pose_tgt = batch["target_pose_index"].long()
        pose_mask = teach_mask
        denom_p = pose_mask.sum().clamp_min(1.0)
        ce_pose = Fn.cross_entropy(
            pose_logits.reshape(-1, pose_logits.shape[-1]),
            pose_tgt.reshape(-1),
            reduction="none",
        ).reshape(pose_logits.shape[0], pose_logits.shape[1])
        l_pose = (ce_pose * pose_mask).sum() / denom_p
    else:
        l_pose = torch.zeros((), device=device)

    if "rotation_logits" in out and "target_rotation_index" in batch:
        rot_logits = out["rotation_logits"]
        rot_tgt = batch["target_rotation_index"].long()
        denom_r = teach_mask.sum().clamp_min(1.0)
        ce_rot = Fn.cross_entropy(
            rot_logits.reshape(-1, rot_logits.shape[-1]),
            rot_tgt.reshape(-1),
            reduction="none",
        ).reshape(rot_logits.shape[0], rot_logits.shape[1])
        l_rot = (ce_rot * teach_mask).sum() / denom_r
    else:
        l_rot = torch.zeros((), device=device)

    # xy GMM NLL on best mixture component target
    if teach_mask.any():
        xy_tgt = batch["xy_target"]
        l_xy = _gmm_nll(
            out["mix_logits"], out["xy_mean"], out["xy_logstd"], xy_tgt, teach_mask)
        best_mix = out["mix_logits"].argmax(dim=-1)
        xy_best = out["xy_mean"].gather(
            2, best_mix.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
        fp = batch.get("selected_footprint", torch.zeros_like(xy_best))
        l_boundary = _soft_boundary_penalty(xy_best, fp, batch["table_bounds"], teach_mask)
        l_overlap = _soft_overlap_penalty(xy_best, fp, teach_mask)
        l_keepout = torch.zeros((), device=device)
    else:
        l_xy = l_boundary = l_overlap = l_keepout = torch.zeros((), device=device)

    l_diversity = torch.zeros((), device=device)
    if "proposal_xy_samples" in out and out["proposal_xy_samples"].shape[0] > 1:
        samples = out["proposal_xy_samples"]
        flat = samples.reshape(samples.shape[0], -1)
        sim = Fn.cosine_similarity(flat.unsqueeze(0), flat.unsqueeze(1), dim=-1)
        tri = torch.triu(sim, diagonal=1)
        l_diversity = -tri[tri != 0].mean() if tri.numel() else l_diversity

    total = (
        gen_weights.lambda_cls * l_cls
        + gen_weights.lambda_score * l_score
        + gen_weights.lambda_station * l_station
        + gen_weights.lambda_pose * l_pose
        + gen_weights.lambda_rot * l_rot
        + gen_weights.lambda_xy * l_xy
        + gen_weights.lambda_boundary * l_boundary
        + gen_weights.lambda_overlap * l_overlap
        + gen_weights.lambda_keepout * l_keepout
        + gen_weights.lambda_diversity * l_diversity
    )

    logs = {
        "l_cls": l_cls.detach(),
        "l_score": l_score.detach(),
        "l_station": l_station.detach(),
        "l_pose": l_pose.detach(),
        "l_rot": l_rot.detach(),
        "l_xy": l_xy.detach(),
        "l_boundary": l_boundary.detach(),
        "l_overlap": l_overlap.detach(),
        "l_keepout": l_keepout.detach(),
        "l_diversity": l_diversity.detach(),
        "total": total.detach(),
    }
    return {"loss": total, "logs": logs}
