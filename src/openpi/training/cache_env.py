"""Pin large on-disk caches to a project disk (not the small overlay root)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Large shared disk; keeps HF / uv / tmp / openpi / wandb / torch caches under
# ``TGY_DISK_ROOT/.cache`` (not the container overlay).
TGY_DISK_ROOT = Path("/mnt/project_rlinf/tgy")


def apply_tgy_disk_caches(*, log: bool = True) -> dict[str, str]:
    """Set cache-related env vars under ``TGY_DISK_ROOT`` and optionally log them.

    Called at PyTorch training entry startup so HuggingFace, temp files, and
    related tooling do not fill the container overlay (``/``).
    """
    root = TGY_DISK_ROOT
    cache_home = root / ".cache"
    hf = cache_home / "huggingface"
    layout: list[tuple[str, Path]] = [
        # Broad defaults (uv falls back to $XDG_CACHE_HOME/uv when UV_CACHE_DIR is unset).
        ("XDG_CACHE_HOME", cache_home),
        ("HF_HOME", hf),
        ("HF_DATASETS_CACHE", hf / "datasets"),
        ("HF_HUB_CACHE", hf / "hub"),
        ("HF_ASSETS_CACHE", hf / "assets"),
        ("HF_MODULES_CACHE", hf / "modules"),
        ("HF_LEROBOT_HOME", hf / "lerobot"),
        ("OPENPI_DATA_HOME", cache_home / "openpi"),
        ("TMPDIR", cache_home / "tmp"),
        ("WANDB_DIR", cache_home / "wandb"),
        ("TORCH_HOME", cache_home / "torch"),
        # uv wheel/sdist extraction (archive-v0) lives under UV_CACHE_DIR; set it before
        # ``uv sync`` / ``uv run`` so the venv is not tied to /opt/venv/.cache.
        ("UV_CACHE_DIR", cache_home / "uv"),
    ]
    resolved: dict[str, str] = {}
    for key, path in layout:
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        resolved[key] = str(path)

    (cache_home / "jax").mkdir(parents=True, exist_ok=True)

    if log:
        log_lines = [f"Disk cache layout (root={root}):"]
        for key, val in resolved.items():
            log_lines.append(f"  {key}={val}")
        logging.info("\n".join(log_lines))

    return resolved
