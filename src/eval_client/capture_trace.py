"""Opt-in per-capture trace for debugging black camera frames.

Enabled when ``ROBODOJO_CAPTURE_TRACE`` names an output directory. For each
observation capture it appends one JSON line per env with the render-sync
result (passes, delivered frames, timeout, wall time) and, per camera, the
image mean/std plus the camera prim world position. Off by default; zero cost
when disabled.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np


class CaptureTrace:
    def __init__(self, task_name: str, run_id: str):
        out_dir = os.environ.get("ROBODOJO_CAPTURE_TRACE", "").strip()
        self.enabled = bool(out_dir)
        self._fh = None
        self._capture_idx = 0
        if self.enabled:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{task_name}_{run_id}_{os.getpid()}.jsonl")
            self._fh = open(path, "a", buffering=1)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @staticmethod
    def _render_side_pose(prim_path: str) -> dict:
        """World position of a prim as seen by USD and by Fabric (the renderer's copy)."""
        out = {}
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            out["usd"] = [round(float(x), 4) for x in t]
        except Exception as e:
            out["usd_err"] = str(e)[:80]
        try:
            import omni.usd
            import usdrt

            rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
            rt_prim = rt_stage.GetPrimAtPath(prim_path)
            if not rt_prim or not rt_prim.IsValid():
                out["fabric"] = "missing"
            else:
                for name in ("omni:fabric:worldMatrix", "_worldPosition"):
                    attr = rt_prim.GetAttribute(name)
                    if attr and attr.HasValue():
                        v = attr.Get()
                        if name == "omni:fabric:worldMatrix":
                            v = [v[3][0], v[3][1], v[3][2]]
                        out["fabric"] = [round(float(x), 4) for x in v]
                        out["fabric_attr"] = name
                        break
                else:
                    out["fabric"] = "no_world_attr"
        except Exception as e:
            out["fabric_err"] = str(e)[:80]
        return out

    def probe(self, env, data: dict):
        """Render each camera of env 0 through a fresh, standalone render product.

        Runs once, at capture ``ROBODOJO_TRACE_PROBE_AT`` (default 3). Compares the
        tiled-product image with a brand-new product for the same camera prim:
        a dark tiled image with a bright fresh image means the camera prim is fine
        and the tiled render product is stale.
        """
        probe_at = int(os.environ.get("ROBODOJO_TRACE_PROBE_AT", "3"))
        if not self.enabled or self._capture_idx != probe_at:
            return
        cam_mgr = getattr(env, "camera_manager", None)
        if cam_mgr is None:
            return
        row = {"probe": True, "capture": self._capture_idx, "cams": {}}
        try:
            import omni.replicator.core as rep
        except Exception as e:
            row["err"] = str(e)[:120]
            self._fh.write(json.dumps(row) + "\n")
            return
        vision = data.get(0, {}).get("vision") or {}
        for cam_id, cam_name in enumerate(cam_mgr.camera_names[0]):
            entry = {}
            tiled = vision.get(cam_name, {}).get("color")
            if tiled is not None:
                entry["tiled_mean"] = round(float(np.asarray(tiled)[::8, ::8].mean()), 2)
            try:
                cam = cam_mgr.cameras[0][cam_id]
                entry["path"] = cam.prim_path
                rp = rep.create.render_product(cam.prim_path, (640, 480), force_new=True)
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach([rp.path])
                means = []
                for _ in range(6):
                    env.render()
                    img = ann.get_data()
                    if img is not None and getattr(img, "size", 0):
                        means.append(round(float(np.asarray(img)[::8, ::8, :3].mean()), 2))
                entry["fresh_means"] = means
                ann.detach()
                rp.destroy()
            except Exception as e:
                entry["err"] = str(e)[:160]
            row["cams"][cam_name] = entry
        self._fh.write(json.dumps(row) + "\n")

    def record(self, env, data: dict, env_idx_list, render_seconds: float):
        if not self.enabled:
            return
        self.probe(env, data)
        from env.camera_manager.capture import render_sync

        sync = dict(render_sync.last_sync_info)
        cam_mgr = getattr(env, "camera_manager", None)
        now = time.time()
        for env_idx in env_idx_list:
            obs = data.get(env_idx, {})
            cams = {}
            for cam_name, cam_data in (obs.get("vision") or {}).items():
                image = cam_data.get("color") if isinstance(cam_data, dict) else None
                if image is None:
                    continue
                sample = np.asarray(image)[::8, ::8].astype(np.float32)
                cams[cam_name] = {"mean": round(float(sample.mean()), 2), "std": round(float(sample.std()), 2)}
            if cam_mgr is not None:
                for cam_id, cam_name in enumerate(cam_mgr.camera_names[env_idx]):
                    if cam_name not in cams:
                        continue
                    try:
                        pos, _ = cam_mgr.cameras_xform[env_idx][cam_id].get_world_pose()
                        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
                        cams[cam_name]["pos"] = [round(float(x), 4) for x in pos]
                    except Exception as e:  # diagnostics must never break eval
                        cams[cam_name]["pos_err"] = str(e)[:80]
                    # Renderer-side pose is expensive; sample env 0 every 25 captures.
                    if env_idx == 0 and self._capture_idx % 25 == 0:
                        try:
                            cam_path = cam_mgr.cameras[env_idx][cam_id].prim_path
                            cams[cam_name]["cam_path"] = cam_path
                            cams[cam_name]["render_pose"] = self._render_side_pose(cam_path)
                        except Exception as e:
                            cams[cam_name]["render_pose_err"] = str(e)[:80]
            row = {
                "t": round(now, 3),
                "capture": self._capture_idx,
                "env": int(env_idx),
                "render_s": round(render_seconds, 4),
                "sync": sync,
                "cams": cams,
            }
            self._fh.write(json.dumps(row) + "\n")
        self._capture_idx += 1
