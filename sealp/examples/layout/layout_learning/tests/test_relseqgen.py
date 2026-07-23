"""RelSeqGen unit and compatibility tests."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from layout_learning import features as F
from layout_learning.dataset import LayoutDataset, collate_items, move_batch
from layout_learning.generator_dataset import geometry_holdout_split
from layout_learning.infer import LayoutModelRunner
from layout_learning.models import build_model, is_generator
from layout_learning.models.base import static_only
from layout_learning.projection import project_proposal
from layout_learning.losses import LossWeights
from layout_learning.relseqgen_losses import RelSeqGenLossWeights, compute_relseqgen_loss


def _sample(l2=True, score=0.6):
    parts = []
    for i, pid in enumerate(["base_plate", "post_a", "post_b"]):
        parts.append({
            "part_id": pid,
            "order_index": i,
            "is_first": i == 0,
            "extent": [0.12, 0.1, 0.02],
            "footprint": [0.12, 0.1],
            "goal_pos": [0.22, -0.2 + 0.01 * i, 0.0],
            "goal_rotmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "parent": "fixture" if i == 0 else "base_plate",
            "topdown_count": 2,
            "grasp_total": 12,
            "staging_xy": [0.22, -0.2] if i == 0 else [0.24 + 0.02 * i, -0.15],
            "pose_tag": "preassembled" if i == 0 else "fs_up",
            "rot_name": "goal_pose" if i == 0 else "fs_0",
            "pose_candidates": [{
                "pose_id": "fs_up", "pose_tag": "fs_up", "rot_name": "fs_0",
                "rotmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "footprint": [0.12, 0.1], "support_area": 0.012,
                "support_area_ratio": 1.0, "center_of_mass_height": 0.01,
                "stable_probability": 1.0, "grasp_total": 12, "topdown_count": 2,
            }],
            "target_pose_index": 0,
            "target_rotation_index": 0,
        })
    return {
        "seed": 0,
        "assembly_id": "tower",
        "geometry_domain": "tower",
        "mesh_sha256_bundle": "abc123",
        "assembly_station_pos": [0.22, -0.2, 0.0],
        "table_x_range": [0.0, 0.48],
        "table_y_range": [-0.88, 0.18],
        "parts": parts,
        "l2_pass": l2,
        "layout_score": score if l2 else 0.0,
    }


class TestRelSeqGen(unittest.TestCase):
    def setUp(self):
        self.model = build_model("relseqgen", flat_dim=F.flatten_feature_dim(), model_size="small")
        self.model.eval()
        self.sample = _sample()
        self.batch = move_batch(collate_items([LayoutDataset([self.sample], feature_version="v2")[0]]), "cpu")

    def test_is_generator(self):
        self.assertTrue(self.model.is_generator)
        self.assertFalse(is_generator("dynaseqrel_dynedge"))

    def test_static_only_input(self):
        part_h, _ = self.model._encode_parts(self.batch)
        batch2 = copy.deepcopy(self.batch)
        batch2["node_feat"] = static_only(batch2["node_feat"], batch2["static_mask"])
        part_h2, _ = self.model._encode_parts(batch2)
        self.assertTrue(torch.allclose(part_h, part_h2, atol=1e-5))

    def test_forward_shapes(self):
        out = self.model(self.batch)
        self.assertEqual(out["station_mean"].shape, (1, 2))
        self.assertEqual(out["pose_logits"].shape[0], 1)
        self.assertEqual(out["mix_logits"].shape[-1], 5)

    def test_propose_k(self):
        props = self.model.propose(self.batch, k=8)
        self.assertEqual(props.shape[0], 8)

    def test_propose_structured(self):
        props = self.model.propose_structured(self.batch, k=4, seed=0)
        self.assertEqual(len(props), 4)
        self.assertIn("station_xy_norm", props[0])
        self.assertTrue(np.all(np.isfinite(props[0]["station_xy_norm"])))

    def test_first_part_no_staging(self):
        props = self.model.propose_structured(self.batch, k=1, seed=0)
        first = next(p for p in props[0]["parts"] if p["part_index"] == 0)
        self.assertEqual(first["offset_xy_norm"], [0.0, 0.0])

    def test_projection_bounds(self):
        xy = {"post_a": np.array([10.0, 10.0]), "post_b": np.array([-1.0, -1.0])}
        fp = {"post_a": np.array([0.12, 0.1]), "post_b": np.array([0.12, 0.1])}
        rep = project_proposal(xy, fp, (0.0, 0.48, -0.88, 0.18))
        for v in rep["projected_xy"].values():
            self.assertTrue(0.0 <= v[0] <= 0.48)
            self.assertTrue(-0.88 <= v[1] <= 0.18)

    def test_empty_elite_no_nan(self):
        batch = copy.deepcopy(self.batch)
        batch["feas"] = torch.zeros_like(batch["feas"])
        batch["score"] = torch.zeros_like(batch["score"])
        out = self.model(batch)
        res = compute_relseqgen_loss(out, batch, LossWeights(), RelSeqGenLossWeights())
        self.assertTrue(torch.isfinite(res["loss"]))

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "relseqgen_test.pt")
            ckpt = {
                "model_name": "relseqgen",
                "state_dict": self.model.state_dict(),
                "flat_dim": F.flatten_feature_dim(),
                "max_parts": F.MAX_PARTS_DEFAULT,
                "feature_version": "v2",
                "is_generator": True,
                "model_kwargs": {"hidden": 64, "layers": 2, "heads": 2, "mixtures": 5},
            }
            torch.save(ckpt, path)
            runner = LayoutModelRunner(path)
            self.assertTrue(runner.is_generator)
            layouts = runner.propose_layouts(self.sample, k=2)
            self.assertEqual(len(layouts), 2)

    def test_seed_reproducibility(self):
        a = self.model.propose_structured(self.batch, k=3, seed=42)
        b = self.model.propose_structured(self.batch, k=3, seed=42)
        self.assertEqual(a[0]["parts"][1]["offset_xy_norm"], b[0]["parts"][1]["offset_xy_norm"])

    def test_different_seeds_differ(self):
        a = self.model.propose_structured(self.batch, k=3, seed=1, temperature=1.5)
        b = self.model.propose_structured(self.batch, k=3, seed=2, temperature=1.5)
        diffs = [
            a[i]["parts"][1]["offset_xy_norm"] != b[i]["parts"][1]["offset_xy_norm"]
            for i in range(3)
        ]
        self.assertTrue(any(diffs))

    def test_geometry_holdout_no_leak(self):
        samples = [_sample(), _sample()]
        samples[1]["geometry_domain"] = "pavilion_v1"
        samples[1]["mesh_sha256_bundle"] = "other_mesh"
        train, val, _ = geometry_holdout_split(samples, 0.5, 0, holdout_domains={"pavilion_v1"})
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)

    def test_smoke_legacy_models(self):
        for name in ("deepsets", "seqrel", "sagpn"):
            m = build_model(name, flat_dim=F.flatten_feature_dim())
            out = m(self.batch)
            self.assertIn("feas_logit", out)


if __name__ == "__main__":
    unittest.main()
