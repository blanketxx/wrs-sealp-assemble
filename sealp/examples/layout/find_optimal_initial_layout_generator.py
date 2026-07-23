"""Generator-driven layout search (RelSeqGen and compatible generators).

Flow:
    1. Build condition from asmdef / STL / stable poses
    2. Generator proposes K structured layouts
    3. Deterministic projection
    4. evaluate_layout (L2)
    5. pattern_refine on elites
    6. L3 validation on top candidates

用法示例::

    python -m sealp.examples.layout.find_optimal_initial_layout_generator \\
        --checkpoint checkpoints/layout_models_multitask/relseqgen/stratified/seed0/relseqgen_best.pt \\
        --k-proposals 32 --n-samples 32 --output-name generator_layout

其余参数 (--asmdef / --config / --grasp-dir / --cdprim-type 等)
与 find_optimal_initial_layout_tower_strict_pycharm 完全一致。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod
import generate_layout_dataset as gends

from layout_learning.infer import LayoutModelRunner
from layout_learning.projection import project_proposal

LayoutCandidate = fol.LayoutCandidate

GCFG: Dict[str, object] = {
    "model": "relseqgen",
    "checkpoint": None,
    "k_proposals": 32,
    "temperature": 1.0,
    "proposal_seed": 0,
    "device": None,
    "rerank_checkpoint": None,
    "use_structured": False,
    "min_spacing": 0.02,
    "keepout_radius": 0.14,
}


def _build_cond(searcher, region: Tuple[str, Tuple[int, int], np.ndarray]) -> Dict:
    searcher._set_region_from_tuple(region)
    cand = LayoutCandidate(xy={})
    cand.assembly_region_id = region[0]
    cand.assembly_region_rc = region[1]
    cand.assembly_station_pos = np.asarray(region[2], dtype=float)
    return gends.sample_from_candidate(
        searcher, cand, seed=0, region=region,
        sample_index=0, generation_signature="generator_condition_v1")


class GeneratorLayoutSearcher(gmod.GlobalLayoutSearcher):
    """RelSeqGen 生成器搜索: 复用 fol 初始化, 仅替换 random_search。"""

    _runner: Optional[LayoutModelRunner] = None
    _rerank_runner: Optional[LayoutModelRunner] = None

    def _load_reranker(self) -> None:
        ckpt = GCFG.get("rerank_checkpoint")
        if ckpt and self._rerank_runner is None:
            self._rerank_runner = LayoutModelRunner(str(ckpt), device=GCFG.get("device"))

    def _project_and_evaluate(
        self,
        layout_xy: Dict[str, np.ndarray],
        region: Tuple[str, Tuple[int, int], np.ndarray],
        footprints: Dict[str, np.ndarray],
        goal_xy: Dict[str, np.ndarray],
        first_pid: Optional[str],
        meta: Optional[Dict] = None,
        verbose: bool = True,
        tag_idx: int = 0,
        tag_total: int = 0,
    ) -> LayoutCandidate:
        bounds = (
            float(self.table_x_range[0]), float(self.table_x_range[1]),
            float(self.table_y_range[0]), float(self.table_y_range[1]),
        )
        station_xy = np.asarray(region[2], dtype=np.float32)[:2]
        keepout = station_xy.reshape(1, 2)
        min_gap = float(GCFG.get("min_spacing", 0.02))
        clearance = float(getattr(self, "min_staging_mesh_clearance", min_gap))
        min_gap = max(min_gap, clearance)
        keepout_radius = float(GCFG.get("keepout_radius", 0.14))
        arm_keepouts = [
            (np.asarray([ax, ay], dtype=np.float32),
             float(self.staging_arm_x_clearance),
             float(self.staging_arm_y_clearance))
            for _name, (ax, ay) in self._arm_base_xy_map().items()
        ]

        proj = project_proposal(
            layout_xy, footprints, bounds=bounds,
            preassembled=first_pid, goal_xy=goal_xy,
            keepout_centers=keepout, keepout_radius=keepout_radius,
            arm_keepouts=arm_keepouts,
            min_spacing=min_gap, max_iters=32, spread_if_clustered=True,
        )
        xy = {}
        for pid in self.part_order:
            if pid == first_pid:
                continue
            if pid not in proj["projected_xy"]:
                continue
            xy[pid] = self._clip_xy_for_part(
                pid, np.asarray(proj["projected_xy"][pid], dtype=float))
        cand = self._evaluate_gene(xy, region)
        cand.generator_meta = {
            "raw_xy": proj["raw_xy"],
            "projection": proj,
            "pose_choice": (meta or {}).get("pose_choice", {}),
            "rotation_choice": (meta or {}).get("rotation_choice", {}),
            "proposal_logprob": (meta or {}).get("proposal_logprob", 0.0),
        }
        if verbose and tag_total > 0:
            tag = "L2_OK" if cand.l2_pass else "FAIL"
            proj_tag = proj.get("failure_reason") or "ok"
            overlap = float(proj.get("residual_overlap", 0.0))
            reason = "" if cand.l2_pass else f" reason={cand.fail_reason}"
            print(f"[gen] {tag_idx:03d}/{tag_total} {tag} score={cand.layout_score:.4f} "
                  f"proj={proj_tag} overlap={overlap:.5f}{reason}")
        return cand

    def _generator_propose(
        self,
        rng,
        verbose: bool = True,
    ) -> Tuple[Tuple, List[LayoutCandidate], object]:
        runner = self._runner
        if runner is None or not runner.is_generator:
            raise RuntimeError("Generator search requires a generator checkpoint.")

        k = int(GCFG["k_proposals"])
        regions = self._order_regions_center_first(
            self._assembly_region_candidates(), verbose=verbose)
        region = regions[0]
        self._set_region_from_tuple(region)
        cond = _build_cond(self, region)
        first_pid = self._first_part_id() if self.preassemble_first_part else None
        footprints = {
            p["part_id"]: np.asarray(p.get("footprint", [0.05, 0.05]), float)
            for p in cond.get("parts", [])
        }
        goal_xy = {
            p["part_id"]: np.asarray(p.get("goal_pos", [0, 0, 0]), float)[:2]
            for p in cond.get("parts", [])
        }
        station_xy = np.asarray(region[2], dtype=np.float32)[:2]
        mode = str(GCFG.get("proposal_mode", "layouts")).lower()
        base_seed = int(GCFG.get("proposal_seed", 0))
        temperature = float(GCFG.get("temperature", 1.0))

        t0 = time.time()
        feasible: List[LayoutCandidate] = []

        if mode == "structured":
            proposals = runner.propose_structured_layouts(
                cond, k=k, part_ids=list(self.part_order),
                seed=base_seed, temperature=temperature,
            )
            if verbose:
                print(f"[gen] structured proposed {len(proposals)} layouts "
                      f"in {time.time()-t0:.2f}s")
            for i, prop in enumerate(proposals):
                cand = self._project_and_evaluate(
                    prop["xy"], region, footprints, goal_xy, first_pid,
                    meta=prop, verbose=verbose, tag_idx=i + 1, tag_total=k)
                if cand.l2_pass:
                    feasible.append(cand)
        else:
            # 默认: propose_layouts，与 compare_generators / neural 路径一致
            if verbose:
                print(f"[gen] proposal_mode=layouts station="
                      f"({station_xy[0]:.3f},{station_xy[1]:.3f})")
            layouts = runner.propose_layouts(
                cond, k=k, part_ids=list(self.part_order), station_xy=station_xy)
            if verbose:
                print(f"[gen] layouts proposed {len(layouts)} in {time.time()-t0:.2f}s")
            for i, layout_xy in enumerate(layouts):
                cand = self._project_and_evaluate(
                    layout_xy, region, footprints, goal_xy, first_pid,
                    verbose=verbose, tag_idx=i + 1, tag_total=k)
                if cand.l2_pass:
                    feasible.append(cand)

        return region, feasible, cond

    def random_search(
        self,
        n_samples: int,
        seed: int,
        max_resample_layout: int = 80,
        verbose: bool = True,
        enable_l3: bool = False,
        l3_top_k: int = 3,
        l3_obstacle_mode: str = "staging_aware",
        require_l3: bool = True,
    ) -> Optional[LayoutCandidate]:
        del n_samples  # generator path ignores random explore budget
        self._reset_eval_progress_stats()
        gmod.GCFG["max_resample_layout"] = int(max_resample_layout)

        elite_k = max(1, int(gmod.GCFG["elite"]))
        steps = [float(s) for s in gmod.GCFG["refine_steps"]]
        rounds = int(gmod.GCFG["refine_rounds"])
        diagonal = bool(gmod.GCFG["refine_diagonal"])
        refine_enabled = bool(gmod.GCFG["refine_enabled"]) and len(steps) > 0 and rounds > 0

        print("\n========== Generator Layout Search ==========")
        print(f"model           = {GCFG['model']}")
        print(f"checkpoint      = {GCFG['checkpoint']}")
        print(f"k_proposals     = {GCFG['k_proposals']}")
        print(f"temperature     = {GCFG['temperature']}")
        print(f"proposal_seed   = {GCFG['proposal_seed']}")
        print(f"proposal_mode   = {GCFG.get('proposal_mode', 'layouts')}")
        print(f"min_spacing     = {GCFG.get('min_spacing', 0.02)}")
        print(f"keepout_radius  = {GCFG.get('keepout_radius', 0.14)}")
        print(f"rerank_ckpt     = {GCFG['rerank_checkpoint']}")
        print(f"elite (refine)  = {elite_k}, refine enabled={refine_enabled}")
        if refine_enabled:
            print(f"refine steps    = {steps}, rounds={rounds}, diagonal={diagonal}")
        print(f"max_evals       = {nsga2._CFG.get('max_evals')}")
        print(f"L3 default/cur  = {'ON' if enable_l3 else 'OFF'}")

        t0 = time.time()
        rng = np.random.default_rng(seed)

        print("\n---------- Phase A: generator propose + evaluate ----------")
        try:
            _region, feasible, _cond = self._generator_propose(rng, verbose=verbose)
        except Exception as exc:
            print(f"[FAIL] generator propose failed: {type(exc).__name__}: {exc!r}")
            return None

        if not feasible:
            print("[FAIL] no L2-feasible generator proposals.")
            return None

        self._load_reranker()
        if self._rerank_runner is not None:
            region = _region
            samples = [gends.sample_from_candidate(
                self, c, seed=0, region=region, sample_index=j,
                generation_signature="generator_rerank_v1")
                for j, c in enumerate(feasible)]
            pred = self._rerank_runner.score_layouts(samples)
            order = np.argsort(-pred["score"])
            feasible = [feasible[int(j)] for j in order]

        elites = self._unique_elites(feasible, limit=max(elite_k, int(l3_top_k)))
        print(f"\nPhase A 完成: 可行 {len(feasible)} 个, 去重精英 {len(elites)} 个。")
        for r, c in enumerate(elites[:elite_k], 1):
            print(f"  elite#{r} score={c.layout_score:.4f} region={c.assembly_region_id}")

        refined: List[LayoutCandidate] = []
        if refine_enabled:
            print("\n---------- Phase B: pattern refine ----------")
            for r, c in enumerate(elites[:elite_k], 1):
                if self._eval_budget_exhausted():
                    refined.append(c)
                    continue
                print(f"[refine] elite#{r} (start score={c.layout_score:.4f}) ...")
                refined.append(self._pattern_refine(c, steps, rounds, diagonal, verbose))
        else:
            print("\n---------- Phase B: SKIPPED (--no-refine) ----------")

        pool = self._unique_elites(list(feasible) + refined, limit=max(int(l3_top_k) * 2, elite_k))
        pool.sort(key=lambda c: c.layout_score, reverse=True)
        best = pool[0]

        print("\n========== Generator Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"real evaluations    = {self._nsga_eval_count}")
        print(f"feasible found      = {len(feasible)}")
        print(f"[BEST-L2] score={best.layout_score:.4f} region={best.assembly_region_id} "
              f"rc={best.assembly_region_rc}")

        if enable_l3:
            print("\n========== Optional L3 full-process validation ==========")
            k = min(int(l3_top_k), len(pool))
            for rank, cand in enumerate(pool[:k], start=1):
                print(f"[L3] rank {rank}/{k} score={cand.layout_score:.4f} "
                      f"region={cand.assembly_region_id}")
                if self.validate_full_sequence_l3(
                        cand, obstacle_mode=l3_obstacle_mode, verbose=True):
                    print(f"[OK] L3 passed rank={rank}")
                    return cand
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")
            if require_l3:
                print("\n[FAIL] L3 top-k all failed.")
                return None
            print("\n[WARN] L3 failed, require_l3=False, fallback to L2 best.")

        return best


def _consume_generator_args() -> None:
    v = fast._consume_extra_value("--model")
    if v is not None:
        GCFG["model"] = str(v)
    v = fast._consume_extra_value("--checkpoint")
    if v is None:
        raise SystemExit("--checkpoint is required for generator search")
    GCFG["checkpoint"] = str(v)
    v = fast._consume_extra_value("--k-proposals")
    if v is not None:
        GCFG["k_proposals"] = int(v)
    v = fast._consume_extra_value("--temperature")
    if v is not None:
        GCFG["temperature"] = float(v)
    v = fast._consume_extra_value("--proposal-seed")
    if v is not None:
        GCFG["proposal_seed"] = int(v)
    v = fast._consume_extra_value("--device")
    if v is not None:
        GCFG["device"] = str(v)
    v = fast._consume_extra_value("--rerank-checkpoint")
    if v is not None:
        GCFG["rerank_checkpoint"] = str(v)
    v = fast._consume_extra_value("--proposal-mode")
    if v is not None:
        GCFG["proposal_mode"] = str(v).strip().lower()
    v = fast._consume_extra_value("--min-spacing")
    if v is not None:
        GCFG["min_spacing"] = float(v)


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = GeneratorLayoutSearcher


def main() -> None:
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()
    gmod._consume_global_args()
    _consume_generator_args()

    if fast._consume_extra_flag("--no-refine"):
        gmod.GCFG["refine_enabled"] = False

    runner = LayoutModelRunner(
        str(GCFG["checkpoint"]), device=GCFG.get("device"))
    if not runner.is_generator:
        raise RuntimeError(
            f"checkpoint is not a generator model: {GCFG['checkpoint']}")
    GCFG["model"] = runner.model_name
    GeneratorLayoutSearcher._runner = runner
    GeneratorLayoutSearcher._rerank_runner = None

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    print("[generator] config:")
    for key, val in GCFG.items():
        print(f"    {key:18s} = {val}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[generator] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
