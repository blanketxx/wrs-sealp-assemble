"""Hybrid surrogate for surrogate-assisted GA (SAGA) layout optimization.

Motivation (ROBIO submission)
-----------------------------
The GA in ``infer_assembly_ga.py`` searches over per-part staging placements.
The *ground-truth* fitness -- ``WeightedInitialLayoutSearcher.evaluate_layout``
-- walks the asmdef assembly order and, at every step, places already-assembled
parts at their GOAL pose and not-yet-assembled parts at their STAGING pose
(dynamic obstacles), then checks common grasps / collisions / the hard order-x
constraint.  That is expensive (per-part multi-arm ``reason_common_gids`` over
hundreds of grasps, along the whole sequence), so it cannot be called for every
individual of every generation.

This module provides a **learned surrogate of that expensive evaluator**.  It
predicts, for a *full* candidate layout:

  * ``feas`` -- probability the layout PASSES the real sequence + dynamic-
    obstacle + order-x L2 check, and
  * ``score`` -- the layout_score it would receive if feasible,

directly from the GA genome.  Because a part's pickability depends on where all
the OTHER parts sit (dynamic obstacles), and because order-x is a *pairwise*
ordering relation, the surrogate is a small permutation-equivariant Set-
Transformer: each part is a token, self-attention models the pairwise
interactions, and a global (CLS) token reads out the two heads.

Hybrid / acceleration loop (implemented in ``infer_assembly_ga.py``):

  1. WARM-UP: a handful of random individuals are scored by the *real*
     evaluate_layout to bootstrap a labelled buffer; the surrogate is fit.
  2. SEARCH: the GA fitness for the whole population is the cheap analytic proxy
     PLUS the surrogate feasibility probability.  No WRS calls per individual.
  3. SELECTIVE VERIFICATION: only the top-M surrogate-ranked individuals per
     generation get the real evaluate_layout (guaranteeing the reported best is
     truly L2-feasible).
  4. ACTIVE LEARNING: every real (layout -> pass/fail, score) label is appended
     to the buffer and the surrogate is periodically re-fit, so it gets more
     accurate exactly in the region the GA is exploring -> fewer wasted WRS
     verifications as the search progresses.

The module is intentionally self-contained (torch only) and consumes the GA's
own genome + precomputed candidate pools, so it does not depend on the JSONL
dataset feature pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore
    _HAS_TORCH = False


# Per-part token feature layout (see _encode_part).
SURR_PART_DIM = 16
# Global (CLS) feature layout (see CenterContext.global_vector).
SURR_GLOBAL_DIM = 5


# ------------------------------------------------------------------
# Feature encoding: GA genome -> part tokens
# ------------------------------------------------------------------
@dataclass
class CenterContext:
    """Everything needed to featurize an individual at one assembly center.

    One context is built per kept assembly center (goal poses / pools depend on
    the center).  Encoding a genome is then a cheap array lookup + arithmetic.
    """

    center_xy: np.ndarray                    # [2] assembly center on the table
    order_pids: List[str]                    # GA parts in assembly order
    first_goal_xy: Optional[np.ndarray]      # [2] preassembled first part goal xy
    table_x_range: Tuple[float, float]
    table_y_range: Tuple[float, float]
    goal_xy: Dict[str, np.ndarray] = field(default_factory=dict)   # per-part goal xy
    # per-part, per-candidate cached scalars (parallel to the GA pool order)
    grasp_norm: float = 40.0
    manip_norm: float = 0.05
    dist_decay: float = 0.40

    def __post_init__(self):
        xr, yr = self.table_x_range, self.table_y_range
        self._x0, self._xw = float(xr[0]), max(float(xr[1] - xr[0]), 1e-6)
        self._y0, self._yw = float(yr[0]), max(float(yr[1] - yr[0]), 1e-6)
        self._diag = float(np.hypot(self._xw, self._yw))
        self._order_index = {pid: i for i, pid in enumerate(self.order_pids)}
        self._n = max(len(self.order_pids), 1)

    # -- normalizers -------------------------------------------------
    def _nx(self, x: float) -> float:
        return float(np.clip(2.0 * (x - self._x0) / self._xw - 1.0, -1.5, 1.5))

    def _ny(self, y: float) -> float:
        return float(np.clip(2.0 * (y - self._y0) / self._yw - 1.0, -1.5, 1.5))

    def global_vector(self) -> np.ndarray:
        g = np.zeros(SURR_GLOBAL_DIM, dtype=np.float32)
        g[0] = self._nx(float(self.center_xy[0]))
        g[1] = self._ny(float(self.center_xy[1]))
        g[2] = float(self._n) / 8.0
        if self.first_goal_xy is not None:
            g[3] = self._nx(float(self.first_goal_xy[0]))
            g[4] = self._ny(float(self.first_goal_xy[1]))
        return g

    def _encode_part(self, pid: str, rec: Dict, prev_x: Optional[float]) -> np.ndarray:
        f = np.zeros(SURR_PART_DIM, dtype=np.float32)
        oi = self._order_index.get(pid, 0)
        order_frac = oi / max(self._n - 1, 1)
        xy = np.asarray(rec["xy"], dtype=float)[:2]
        gxy = self.goal_xy.get(pid, xy)
        dx = float(gxy[0] - xy[0])
        dy = float(gxy[1] - xy[1])
        dist = float(rec.get("dist_to_goal", np.hypot(dx, dy)))
        fp = np.asarray(rec.get("footprint", [0.05, 0.05]), dtype=float)[:2]
        g = float(max(rec.get("common_grasp_count", 0.0), 0.0))
        m = float(max(rec.get("manipulability", 0.0), 0.0))
        f[0] = order_frac
        f[1] = float(np.sin(2.0 * np.pi * order_frac))
        f[2] = float(np.cos(2.0 * np.pi * order_frac))
        f[3] = self._nx(float(xy[0]))
        f[4] = self._ny(float(xy[1]))
        f[5] = self._nx(float(gxy[0]))
        f[6] = self._ny(float(gxy[1]))
        f[7] = dx / self._diag
        f[8] = dy / self._diag
        f[9] = dist / self._diag
        f[10] = float(fp[0]) / self._xw
        f[11] = float(fp[1]) / self._yw
        f[12] = g / (g + self.grasp_norm)                       # common-grasp saturation
        f[13] = 1.0 - float(np.exp(-m / max(self.manip_norm, 1e-9)))
        # signed order-x overshoot vs the previous part in the assembly sequence
        # (>0 means this later part stages FURTHER +x than its predecessor -> the
        # dynamic-obstacle / hard order-x constraint the surrogate must learn).
        if prev_x is not None:
            f[14] = float((float(xy[0]) - prev_x) / self._diag)
        # signed order-x overshoot vs the preassembled first part (seat) goal x:
        # a strong dynamic-obstacle / order-x cue the analytic proxy handles only
        # coarsely.
        if self.first_goal_xy is not None:
            f[15] = float((float(xy[0]) - float(self.first_goal_xy[0])) / self._diag)
        return f

    def encode(self, genes: Dict[str, int],
               pools: Dict[str, List[Dict]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (tokens [N, SURR_PART_DIM], mask [N], global [SURR_GLOBAL_DIM])."""
        toks: List[np.ndarray] = []
        prev_x: Optional[float] = None
        for pid in self.order_pids:
            if pid not in genes:
                continue
            rec = pools[pid][genes[pid]]
            toks.append(self._encode_part(pid, rec, prev_x))
            prev_x = float(np.asarray(rec["xy"], dtype=float)[0])
        if not toks:
            toks = [np.zeros(SURR_PART_DIM, dtype=np.float32)]
        tokens = np.stack(toks, axis=0).astype(np.float32)
        mask = np.ones(tokens.shape[0], dtype=np.float32)
        return tokens, mask, self.global_vector()


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------
if _HAS_TORCH:

    def _mlp(dims: Sequence[int], last_act: bool = False, dropout: float = 0.0) -> "nn.Sequential":
        layers: List["nn.Module"] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last or last_act:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    class _SAB(nn.Module):
        """Set-attention block (self-attention + FFN, pre-norm)."""

        def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)
            self.ff = _mlp([dim, 2 * dim, dim], last_act=False, dropout=dropout)

        def forward(self, x: "torch.Tensor", kpm: "torch.Tensor") -> "torch.Tensor":
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
            x = x + a
            x = x + self.ff(self.ln2(x))
            return x

    class AssemblyGASurrogate(nn.Module):
        """Set-Transformer surrogate of ``evaluate_layout``.

        Inputs (batched, padded):
            tokens : [B, N, SURR_PART_DIM]
            glob   : [B, SURR_GLOBAL_DIM]
            mask   : [B, N]   (1 = valid part, 0 = padding)
        Outputs:
            feas_logit : [B]   BCE target = real L2 pass/fail
            score_pred : [B]   in (0,1), regression to layout_score of feasible ones
        """

        def __init__(self, part_dim: int = SURR_PART_DIM, global_dim: int = SURR_GLOBAL_DIM,
                     d_model: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.d_model = int(d_model)
            self.part_proj = _mlp([part_dim, d_model, d_model], last_act=True, dropout=dropout)
            self.global_proj = _mlp([global_dim, d_model, d_model], last_act=True, dropout=dropout)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.blocks = nn.ModuleList([_SAB(d_model, heads, dropout) for _ in range(int(layers))])
            self.norm = nn.LayerNorm(d_model)
            self.feas_head = _mlp([d_model, d_model, 1], last_act=False, dropout=dropout)
            self.score_head = _mlp([d_model, d_model, 1], last_act=False, dropout=dropout)

        def forward(self, tokens: "torch.Tensor", glob: "torch.Tensor",
                    mask: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            b, n, _ = tokens.shape
            h = self.part_proj(tokens)                                   # [B,N,d]
            cls = self.cls.expand(b, 1, self.d_model) + self.global_proj(glob).unsqueeze(1)
            x = torch.cat([cls, h], dim=1)                               # [B,1+N,d]
            valid = torch.cat([torch.ones(b, 1, device=mask.device), mask], dim=1)
            kpm = (valid <= 0)                                           # True = ignore
            for blk in self.blocks:
                x = blk(x, kpm)
            cls_out = self.norm(x[:, 0])
            return {
                "feas_logit": self.feas_head(cls_out).squeeze(-1),
                "score_pred": torch.sigmoid(self.score_head(cls_out)).squeeze(-1),
            }


# ------------------------------------------------------------------
# Online (active-learning) manager used by the GA driver
# ------------------------------------------------------------------
class HybridSurrogateManager:
    """Owns the surrogate net + labelled replay buffer + online fit loop.

    The GA driver only needs: ``add_label`` (after each real evaluate_layout),
    ``fit`` (periodically), ``predict`` (batched surrogate fitness), and the
    ``ready`` flag (True once enough labels + one fit have happened).
    """

    def __init__(self, contexts: List[CenterContext], pools_by_c: List[Dict[str, List[Dict]]],
                 *, d_model: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.1,
                 device: str = "cpu", lr: float = 1e-3, min_labels: int = 16,
                 max_buffer: int = 4000, seed: int = 0):
        if not _HAS_TORCH:
            raise RuntimeError("HybridSurrogateManager requires torch")
        self.contexts = contexts
        self.pools_by_c = pools_by_c
        self.device = device
        self.min_labels = int(min_labels)
        self.max_buffer = int(max_buffer)
        self._rng = np.random.default_rng(int(seed))
        self.net = AssemblyGASurrogate(d_model=d_model, heads=heads,
                                       layers=layers, dropout=dropout).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(lr), weight_decay=1e-5)
        # replay buffer of encoded samples
        self._tokens: List[np.ndarray] = []
        self._masks: List[np.ndarray] = []
        self._globs: List[np.ndarray] = []
        self._feas: List[float] = []
        self._score: List[float] = []
        self._keys = set()
        self.ready = False
        self.n_fit = 0

    # -- labelling ---------------------------------------------------
    def _encode(self, ind: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = int(ind["c"])
        return self.contexts[c].encode(ind["genes"], self.pools_by_c[c])

    @staticmethod
    def _key(ind: Dict):
        return (int(ind["c"]), tuple(sorted(ind["genes"].items())))

    def add_label(self, ind: Dict, passed: bool, score: float) -> bool:
        """Append one real evaluate_layout outcome.  Deduplicated by genome."""
        k = self._key(ind)
        if k in self._keys:
            return False
        self._keys.add(k)
        tok, msk, glb = self._encode(ind)
        self._tokens.append(tok)
        self._masks.append(msk)
        self._globs.append(glb)
        self._feas.append(1.0 if passed else 0.0)
        self._score.append(float(score) if passed else 0.0)
        if len(self._feas) > self.max_buffer:  # drop oldest
            for buf in (self._tokens, self._masks, self._globs, self._feas, self._score):
                buf.pop(0)
        return True

    @property
    def n_labels(self) -> int:
        return len(self._feas)

    @property
    def n_pos(self) -> int:
        return int(sum(self._feas))

    # -- batching (variable N -> pad) --------------------------------
    def _make_batch(self, idx: Sequence[int]):
        max_n = max(self._tokens[i].shape[0] for i in idx)
        b = len(idx)
        toks = np.zeros((b, max_n, SURR_PART_DIM), dtype=np.float32)
        msks = np.zeros((b, max_n), dtype=np.float32)
        globs = np.zeros((b, SURR_GLOBAL_DIM), dtype=np.float32)
        feas = np.zeros(b, dtype=np.float32)
        score = np.zeros(b, dtype=np.float32)
        for bi, i in enumerate(idx):
            n = self._tokens[i].shape[0]
            toks[bi, :n] = self._tokens[i]
            msks[bi, :n] = self._masks[i]
            globs[bi] = self._globs[i]
            feas[bi] = self._feas[i]
            score[bi] = self._score[i]
        dev = self.device
        return (torch.from_numpy(toks).to(dev), torch.from_numpy(globs).to(dev),
                torch.from_numpy(msks).to(dev), torch.from_numpy(feas).to(dev),
                torch.from_numpy(score).to(dev))

    def fit(self, epochs: int = 40, batch_size: int = 64) -> Dict[str, float]:
        """Fit the surrogate on the current buffer (class-balanced BCE + score MSE)."""
        n = self.n_labels
        if n < self.min_labels:
            return {"trained": 0.0, "n": float(n)}
        n_pos = self.n_pos
        n_neg = n - n_pos
        pos_w = float(max(n_neg, 1)) / float(max(n_pos, 1))
        pos_w = float(np.clip(pos_w, 0.2, 20.0))
        pos_weight = torch.tensor([pos_w], device=self.device)
        self.net.train()
        last = {"bce": 0.0, "mse": 0.0}
        for _ in range(int(epochs)):
            perm = self._rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                toks, globs, msks, feas, score = self._make_batch(idx)
                out = self.net(toks, globs, msks)
                bce = Fn.binary_cross_entropy_with_logits(
                    out["feas_logit"], feas, pos_weight=pos_weight)
                # score regression only where feasible
                w = feas
                if float(w.sum()) > 0:
                    mse = (((out["score_pred"] - score) ** 2) * w).sum() / w.sum().clamp_min(1.0)
                else:
                    mse = torch.zeros((), device=self.device)
                loss = bce + 0.5 * mse
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                last = {"bce": float(bce.item()), "mse": float(mse.item())}
        self.net.eval()
        self.ready = True
        self.n_fit += 1
        last.update({"trained": 1.0, "n": float(n), "n_pos": float(n_pos), "pos_w": pos_w})
        return last

    # -- inference ---------------------------------------------------
    def predict(self, inds: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Batched surrogate outputs for a list of individuals.

        Returns (feas_prob [B], score_pred [B]).  If not ready, returns 0.5 /
        0.0 so callers can blend safely before the first fit.
        """
        b = len(inds)
        if not inds:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if not self.ready:
            return np.full(b, 0.5, dtype=np.float32), np.zeros(b, dtype=np.float32)
        enc = [self._encode(ind) for ind in inds]
        max_n = max(e[0].shape[0] for e in enc)
        toks = np.zeros((b, max_n, SURR_PART_DIM), dtype=np.float32)
        msks = np.zeros((b, max_n), dtype=np.float32)
        globs = np.zeros((b, SURR_GLOBAL_DIM), dtype=np.float32)
        for bi, (t, m, g) in enumerate(enc):
            nn_ = t.shape[0]
            toks[bi, :nn_] = t
            msks[bi, :nn_] = m
            globs[bi] = g
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(toks).to(self.device),
                           torch.from_numpy(globs).to(self.device),
                           torch.from_numpy(msks).to(self.device))
        feas = torch.sigmoid(out["feas_logit"]).cpu().numpy().astype(np.float32)
        score = out["score_pred"].cpu().numpy().astype(np.float32)
        return feas, score
