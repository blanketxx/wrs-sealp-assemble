"""Read shared workspace settings (table / robot mode) from sample_config.yaml."""
from __future__ import annotations

from typing import Any, Dict

import yaml


def load_workspace_settings(config_yaml: str) -> Dict[str, Any]:
    with open(config_yaml, "r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}

    robot_cfg = cfg.get("robot") or {}
    mode = str(robot_cfg.get("mode", "dual")).strip().lower()
    single_arm = mode in ("single", "sgl", "one", "single_arm")

    return {
        "robot_mode": "single" if single_arm else "dual",
        "single_arm": single_arm,
        "dual_arm_y_offset": float(robot_cfg.get("dual_arm_y_offset", 0.62)),
        "robot_type": str(robot_cfg.get("type", "panthera_ht")),
    }
