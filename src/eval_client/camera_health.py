"""Detect black or blank camera frames in eval observations.

A camera that renders black (renderer not warmed up, wrist camera not
delivering frames, ...) silently turns a policy failure into an
infrastructure failure. This monitor inspects the colour image of every
camera in every observation, keeps per-episode counters, and lets the eval
loop either log the problem or drop the episode as invalid.

Mode is set with ``ROBODOJO_CAMERA_CHECK``:

* ``off``    - no checks.
* ``warn``   - (default) log bad cameras and record stats in ``_result.json``.
* ``strict`` - like ``warn``, and an episode whose bad-frame fraction for any
  camera exceeds ``ROBODOJO_CAMERA_BAD_FRAC`` (default 0.5) is marked unstable,
  so it is excluded from the score instead of counting as a policy failure.

A frame is "bad" when its mean intensity is below ``ROBODOJO_CAMERA_MIN_MEAN``
(default 10, on a 0-255 scale) or its standard deviation is below
``ROBODOJO_CAMERA_MIN_STD`` (default 2), i.e. black or uniform.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_MODES = ("off", "warn", "strict")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class CameraHealthMonitor:
    def __init__(self):
        mode = os.environ.get("ROBODOJO_CAMERA_CHECK", "warn").strip().lower()
        self.mode = mode if mode in _MODES else "warn"
        self.min_mean = _env_float("ROBODOJO_CAMERA_MIN_MEAN", 10.0)
        self.min_std = _env_float("ROBODOJO_CAMERA_MIN_STD", 2.0)
        self.max_bad_frac = _env_float("ROBODOJO_CAMERA_BAD_FRAC", 0.5)
        # {env_idx: {camera_name: [n_frames, n_bad, sum_mean]}}
        self._stats: dict[int, dict[str, list[float]]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def strict(self) -> bool:
        return self.mode == "strict"

    def reset(self, env_idx_list=None):
        if env_idx_list is None:
            self._stats = {}
            return
        for env_idx in env_idx_list:
            self._stats.pop(env_idx, None)

    @staticmethod
    def _color(cam_data):
        if isinstance(cam_data, dict):
            for key in ("color", "rgb"):
                if key in cam_data:
                    return cam_data[key]
            return None
        return cam_data

    def update(self, env_idx: int, obs: dict):
        if not self.enabled:
            return
        vision = obs.get("vision") or {}
        env_stats = self._stats.setdefault(env_idx, {})
        for cam_name, cam_data in vision.items():
            image = self._color(cam_data)
            if image is None:
                continue
            image = np.asarray(image)
            if image.size == 0:
                continue
            # Subsample for speed; enough pixels to judge blank vs. textured.
            sample = image[::4, ::4].astype(np.float32)
            mean = float(sample.mean())
            if np.issubdtype(image.dtype, np.floating) and float(image.max(initial=0.0)) <= 1.0:
                sample = sample * 255.0
                mean *= 255.0
            bad = mean < self.min_mean or float(sample.std()) < self.min_std
            s = env_stats.setdefault(cam_name, [0, 0, 0.0])
            s[0] += 1
            s[1] += int(bad)
            s[2] += mean

    def summary(self, env_idx: int) -> dict:
        out = {}
        for cam_name, (n, n_bad, sum_mean) in sorted(self._stats.get(env_idx, {}).items()):
            if n == 0:
                continue
            out[cam_name] = {
                "frames": int(n),
                "bad_frac": round(n_bad / n, 4),
                "mean": round(sum_mean / n, 2),
            }
        return out

    def bad_cameras(self, env_idx: int) -> list[str]:
        return [name for name, s in self.summary(env_idx).items() if s["bad_frac"] > self.max_bad_frac]

    def report(self, env_idx: int, task_name: str = "") -> tuple[dict, list[str]]:
        """Return (summary, bad camera names) for an env and log problems."""
        summary = self.summary(env_idx)
        bad = self.bad_cameras(env_idx)
        if bad:
            logger.warning(
                "[camera_health] task=%s env=%d bad cameras (bad_frac > %.2f): %s; stats=%s",
                task_name,
                env_idx,
                self.max_bad_frac,
                ",".join(bad),
                summary,
            )
        return summary, bad
