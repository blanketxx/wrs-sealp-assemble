"""Offline STL geometry descriptors for RelSeqGen.

Deterministic mesh statistics cached under ``sealp/assets/model_features/``.
Cache key = STL SHA256 + feature schema version + normalization version.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

FEATURE_SCHEMA_VERSION = "geo_v1"
NORMALIZATION_VERSION = "norm_v1"
GEO_FEATURE_DIM = 20

_SEALP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CACHE_ROOT = os.path.join(_SEALP_ROOT, "assets", "model_features")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_scale(values: np.ndarray, scale: float) -> np.ndarray:
    s = max(float(scale), 1e-9)
    return (values / s).astype(np.float32)


def _extent_from_part(part: Dict) -> np.ndarray:
    ext = np.asarray(part.get("extent", [0.0, 0.0, 0.0]), dtype=np.float32)
    if ext.size < 3:
        ext = np.pad(ext, (0, 3 - ext.size))
    return ext[:3]


def _fallback_descriptor(part: Dict, norm_scale: float) -> np.ndarray:
    """Use extent / footprint when mesh_path is unavailable."""
    warnings.warn(
        f"geometry fallback for part={part.get('part_id', '?')}: "
        "no mesh_path; using extent/footprint only.",
        stacklevel=2,
    )
    ext = _extent_from_part(part)
    fp = np.asarray(part.get("footprint", ext[:2]), dtype=np.float32)
    vol_proxy = float(ext[0] * ext[1] * ext[2])
    area_proxy = float(2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2]))
    diag = float(np.linalg.norm(ext))
    scale = max(norm_scale, diag, 1e-6)
    feat = np.zeros(GEO_FEATURE_DIM, dtype=np.float32)
    feat[0:3] = _safe_scale(ext, scale)
    feat[3:5] = _safe_scale(fp[:2], scale)
    feat[5] = vol_proxy / max(scale ** 3, 1e-12)
    feat[6] = area_proxy / max(scale ** 2, 1e-12)
    feat[7] = 1.0
    feat[8] = min(ext) / max(max(ext), 1e-9)
    feat[9:11] = 0.0
    feat[11:14] = ext / max(np.linalg.norm(ext), 1e-9)
    feat[14] = 1.0
    feat[15] = 1.0
    feat[16] = 0.5
    feat[17] = 0.25
    feat[18] = 0.0
    feat[19] = 0.0
    return feat


def _compute_from_mesh(mesh, norm_scale: float) -> np.ndarray:
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    bounds = mesh.bounds
    ext_aabb = (bounds[1] - bounds[0]).astype(np.float32)
    try:
        obb = mesh.bounding_box_oriented
        ext_obb = (obb.extents).astype(np.float32)
    except Exception:
        ext_obb = ext_aabb.copy()

    vol = float(mesh.volume) if mesh.is_watertight else float(ext_aabb.prod())
    area = float(mesh.area)
    try:
        hull = mesh.convex_hull
        hull_vol = float(hull.volume) if hull.volume > 0 else vol
    except Exception:
        hull_vol = vol
    compactness = vol / max(area ** 1.5, 1e-12)
    centroid = mesh.centroid.astype(np.float32)
    origin_offset = centroid.copy()
    try:
        inertia = mesh.moment_inertia
        eigvals = np.linalg.eigvalsh(inertia).astype(np.float32)
        eigvals = np.sort(eigvals)[::-1]
    except Exception:
        eigvals = ext_aabb ** 2

    diag = float(max(np.linalg.norm(ext_aabb), norm_scale, 1e-6))
    scale = diag
    n_comp = 1
    try:
        n_comp = len(mesh.split())
    except Exception:
        n_comp = 1

    feat = np.zeros(GEO_FEATURE_DIM, dtype=np.float32)
    feat[0:3] = _safe_scale(ext_aabb, scale)
    feat[3:5] = _safe_scale(ext_obb[:2], scale)
    feat[5] = vol / max(scale ** 3, 1e-12)
    feat[6] = area / max(scale ** 2, 1e-12)
    feat[7] = vol / max(hull_vol, 1e-12)
    feat[8] = compactness
    feat[9:11] = _safe_scale(origin_offset[:2], scale)
    feat[11:14] = _safe_scale(np.sqrt(np.clip(eigvals, 0, None)), scale)
    feat[14] = float(n_comp)
    feat[15] = float(mesh.is_watertight)
    feat[16] = min(ext_aabb) / max(max(ext_aabb), 1e-9)
    feat[17] = float(len(mesh.vertices)) / 50000.0
    feat[18] = float(len(mesh.faces)) / 100000.0
    feat[19] = float(np.linalg.norm(origin_offset)) / scale
    return feat


def cache_path(mesh_sha256: str) -> str:
    return os.path.join(
        _CACHE_ROOT,
        f"{mesh_sha256}_{FEATURE_SCHEMA_VERSION}_{NORMALIZATION_VERSION}.npz",
    )


def load_or_compute_geometry(
    part: Dict,
    mesh_path: Optional[str] = None,
    norm_scale: float = 0.5,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, str, Optional[str]]:
    """Return (descriptor, mesh_sha256, cache_file)."""
    path = mesh_path or part.get("mesh_path")
    sha = part.get("mesh_sha256")
    if path and os.path.isfile(path):
        sha = sha or _sha256_file(path)
        cpath = cache_path(str(sha))
        if not force_recompute and os.path.isfile(cpath):
            data = np.load(cpath)
            return data["descriptor"].astype(np.float32), str(sha), cpath
        import trimesh
        mesh = trimesh.load(path, force="mesh", process=False)
        feat = _compute_from_mesh(mesh, norm_scale)
        os.makedirs(_CACHE_ROOT, exist_ok=True)
        np.savez_compressed(
            cpath,
            descriptor=feat,
            mesh_sha256=str(sha),
            feature_schema=FEATURE_SCHEMA_VERSION,
            normalization=NORMALIZATION_VERSION,
            mesh_path=os.path.abspath(path),
        )
        return feat, str(sha), cpath

    if sha:
        cpath = cache_path(str(sha))
        if os.path.isfile(cpath):
            data = np.load(cpath)
            return data["descriptor"].astype(np.float32), str(sha), cpath
    return _fallback_descriptor(part, norm_scale), str(sha or "fallback"), None


def geometry_domain_key(sample: Dict) -> str:
    return str(sample.get("geometry_domain") or sample.get("assembly_id") or "unknown")


def mesh_sha256_for_sample(sample: Dict) -> str:
    parts = sample.get("parts", [])
    hashes = []
    for p in parts:
        h = p.get("mesh_sha256")
        if h:
            hashes.append(str(h))
    if hashes:
        joined = "|".join(sorted(hashes))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    bundle = sample.get("mesh_sha256_bundle")
    if bundle:
        return str(bundle)
    domain = sample.get("geometry_domain") or sample.get("assembly_id", "no_mesh")
    return str(domain)
