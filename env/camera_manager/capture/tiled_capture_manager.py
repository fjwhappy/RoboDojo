"""
Tiled capture manager: initializes tiled render products and annotators for eval.
"""

from copy import deepcopy
import logging
import os
from typing import List

from isaacsim.sensors.camera import Camera
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omni.replicator.core.scripts.annotators import Annotator
import torch

from env.camera_manager.camera_manager import CameraManager
from env.camera_manager.capture.camera_view import CameraView
from env.environment.isaac.isaac_rl_env import IsaacRLEnv

logger = logging.getLogger(__name__)

# A tile whose mean RGB intensity (0-255) is below this is treated as dark.
_DARK_MEAN = 10.0
# A standalone render of the same camera must exceed this to prove the camera
# itself can see something (so darkness is the tiled product's fault).
_LIT_MEAN = 20.0


class TiledCaptureManager:
    """
    Manages tiled camera capture: initializes cameras, render products, and annotators, and handles video recording.
    """

    def __init__(
        self,
        num_envs: int,
        config: DictConfig,
        camera_manager: CameraManager,
        device: torch.device,
    ):
        self.sim: IsaacRLEnv = None
        self.config = config
        self.camera_manager = camera_manager
        self.num_envs = num_envs
        self.device = device
        self.tiled_render_products: List[str] = []  # all the render products
        self.annotator: List[
            List[List[Annotator]]
        ] = []  # Defines the types and devices of annotator. See yaml for camera for more detail
        self.annotator_type: List[List[List[str]]] = []
        self.annotator_device: List[List[List[str]]] = []
        self.cameras: List[List[Camera]] = camera_manager.cameras
        self.camera_names: List[List[str]] = camera_manager.camera_names
        self.tiled_cameras: List[CameraView] = []
        self.camera_prim_paths: List[List[str]] = []
        # Pre-allocated output buffers for each camera and annotator to reduce memory allocation
        # Format: {cam_id: {annotator_name: wp.array}}
        self._output_buffers: dict = {}
        # Camera slots served by per-env standalone render products: {cam_id: [(rp, annotator), ...]}
        self._standalone_slots: dict = {}

    def initialize(self, sim: IsaacRLEnv):
        """
        Initialize the capture manager.
        This method should be called before the simulation context is created.
        """
        if len(self.cameras) == 0:
            self.cameras = self.camera_manager.cameras
            self.camera_names = self.camera_manager.camera_names
        self.num_cams = len(self.cameras[0])
        self.sim = sim

    def init_cameras(self):
        """
        Initialize cameras based on the configuration.
        1. Initialize the cameras
        2. Get all render_product control
        3. Attach Annotator
        """
        self.camera_prim_paths.clear()
        for env_id in range(self.num_envs):
            self.camera_prim_paths.append([])
            for camera in self.cameras[env_id]:
                self.camera_prim_paths[env_id].append(camera.prim_path)

        for cam_id in range(self.num_cams):
            self.annotator.append([])
            self.annotator_type.append([])
            self.annotator_device.append([])

        config = OmegaConf.to_container(self.config, resolve=True)
        if "colorize_depth" in config:
            self.colorize_depth = config["colorize_depth"]
            config.pop("colorize_depth")
        if "default_frequency" in config:
            config.pop("default_frequency")

        self.config = OmegaConf.create(config)
        annotator_config = OmegaConf.to_container(self.config.annotator, resolve=True)
        if annotator_config is None:
            print("[TiledCaptureManager] No annotator enabled in config. Please check your config file.")
            return
        for cam_id, camera_name in enumerate(self.camera_names[0]):
            if camera_name in annotator_config:
                capture_config = annotator_config[camera_name]
            else:
                capture_config = annotator_config.get("common", None)  # check if there is common annotator config

            if capture_config.get("enabled", False):
                annotators = deepcopy(capture_config)
                annotators.pop("enabled")
                for annotator_name, annotator_setting in annotators.items():
                    type_annotator = annotator_setting["type"]
                    self.annotator_type[cam_id].append(type_annotator)
            tiled_camera = self._create_tiled_camera(cam_id)
            self.tiled_cameras.append(tiled_camera)
            self.tiled_render_products.append(tiled_camera._render_product)

    def _create_tiled_camera(self, cam_id: int) -> CameraView:
        """Create the tiled render product for one camera slot and its output buffers."""
        prim_paths_by_cam_id = [row[cam_id] for row in self.camera_prim_paths]
        camera_resolution = self.cameras[0][cam_id]._resolution
        height, width = camera_resolution[1], camera_resolution[0]
        tiled_camera = CameraView(
            prim_paths_by_cam_id,
            camera_resolution=[width, height],
            output_annotators=self.annotator_type[cam_id],
        )

        # Pre-allocate output buffers for this camera's annotators
        self._output_buffers[cam_id] = {}
        import warp as wp

        from env.camera_manager.capture.camera_view import ANNOTATOR_SPEC

        for annotator_name in self.annotator_type[cam_id]:
            spec = ANNOTATOR_SPEC.get(annotator_name)
            if spec is None:
                continue
            shape = (self.num_envs, height, width, spec["channels"])
            # Pre-allocate warp array on CUDA to reuse memory
            self._output_buffers[cam_id][annotator_name] = wp.zeros(shape, dtype=spec["dtype"], device="cuda:0")
        return tiled_camera

    def _tile_means(self, cam_id: int):
        """Per-env mean RGB intensity of the current tiled frame, or None if no rgb annotator."""
        name = self._rgb_annotator_name(cam_id)
        if name is None:
            return None
        out, _ = self.tiled_cameras[cam_id].get_data(name, out=self._output_buffers[cam_id].get(name))
        img = out.numpy() if hasattr(out, "numpy") else out
        return img[:, ::8, ::8, :3].reshape(img.shape[0], -1).mean(axis=1)

    def _rgb_annotator_name(self, cam_id: int):
        for name in ("rgb", "rgba"):
            if name in self.annotator_type[cam_id]:
                return name
        return None

    def _create_standalone_slot(self, cam_id: int) -> list:
        """One single-camera render product + rgb annotator per env for a camera slot."""
        import omni.replicator.core as rep

        slot = []
        for env_id in range(self.num_envs):
            camera = self.cameras[env_id][cam_id]
            width, height = camera._resolution
            rp = rep.create.render_product(camera.prim_path, (width, height), force_new=True)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp.path])
            slot.append((rp, ann))
        return slot

    def _destroy_standalone_slot(self, slot: list):
        for rp, ann in slot:
            try:
                ann.detach()
                rp.destroy()
            except Exception as e:
                logger.warning("[TiledCaptureManager] standalone cleanup failed: %s", e)

    @staticmethod
    def _standalone_image(ann):
        data = ann.get_data()
        if data is None or not getattr(data, "size", 0):
            return None
        return np.asarray(data)

    def repair_dark_cameras(self, render_fn) -> dict:
        """Switch camera slots whose tiled render product delivers dark tiles to per-camera products.

        Under several concurrent Isaac Sim processes, the tiled render product of a
        camera slot is sometimes created in a state where its tiles stay black (or
        flicker black) for the whole run, although the camera prims are correctly
        posed: a standalone render product for the same camera is lit, and
        destroying and recreating the tiled product does not help. This checks each
        slot; if any env's tile is dark while a standalone render of that env's
        camera is lit, the slot is served from one standalone render product per env
        from then on (slower, but correct).

        Disable with ``ROBODOJO_CAMERA_REPAIR=0``. Returns a report per camera name.
        """
        if os.environ.get("ROBODOJO_CAMERA_REPAIR", "1").strip().lower() in ("0", "false", "off", "no"):
            return {}
        report = {}
        for cam_id, camera_name in enumerate(self.camera_names[0]):
            entry = {}
            if cam_id in self._standalone_slots:
                report[camera_name] = {"mode": "standalone"}
                continue
            try:
                # Several renders so a flickering slot is caught in a dark phase.
                dark_envs = set()
                for _ in range(8):
                    render_fn()
                    means = self._tile_means(cam_id)
                    if means is None:
                        break
                    dark_envs.update(int(i) for i in np.flatnonzero(means < _DARK_MEAN))
                if not dark_envs:
                    report[camera_name] = entry
                    continue
                entry["dark_envs"] = sorted(dark_envs)
                slot = self._create_standalone_slot(cam_id)
                means = None
                for _ in range(4):
                    render_fn()
                    imgs = [self._standalone_image(ann) for _, ann in slot]
                    means = [float(im[::8, ::8, :3].mean()) if im is not None else 0.0 for im in imgs]
                entry["standalone_means"] = [round(m, 1) for m in means]
                if all(means[e] < _LIT_MEAN for e in dark_envs):
                    # The cameras themselves see nothing; not a render-product fault.
                    self._destroy_standalone_slot(slot)
                    entry["mode"] = "tiled"
                else:
                    self._standalone_slots[cam_id] = slot
                    entry["mode"] = "standalone"
                print(f"[camera_repair] {camera_name}: {entry}", flush=True)
            except Exception as e:  # repair must never break eval
                entry["error"] = str(e)[:200]
                print(f"[camera_repair] {camera_name}: error {entry['error']}", flush=True)
            report[camera_name] = entry
        return report

    def step(self, env_ids: List[int] = None, cam_ids: List[int] = None) -> List[List[List[any]]]:
        """
        Step the annotator. When env_id and cam_id is given, use the given. Otherwise apply to all cameras.
        Args:
            env_ids: List[int] - list of environment IDs to process
            cam_ids: List[int] - list of camera IDs to process
        Returns:
            List[List[List[Any]]] : returns the required data from annotators for required env_ids and cam_ids
            Format: data[cam_id][annotator_name] = [env_0_data, env_1_data, ...]
            where each env_i_data is {data: numpy_array, info: dict}
        """

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if cam_ids is None:
            cam_ids = list(range(len(self.cameras[0])))

        data = []
        for cam_id in cam_ids:
            cam_data = {}
            annotator_names = self.annotator_type[cam_id]
            if cam_id in self._standalone_slots:
                slot = self._standalone_slots[cam_id]
                rgb_name = self._rgb_annotator_name(cam_id)
                if rgb_name is not None:
                    env_list = []
                    for env_id in env_ids:
                        img = self._standalone_image(slot[env_id][1])
                        if img is None:
                            camera = self.cameras[env_id][cam_id]
                            width, height = camera._resolution
                            img = np.zeros((height, width, 4), dtype=np.uint8)
                        env_list.append({"data": img, "info": {}})
                    cam_data[rgb_name] = env_list
                data.append(cam_data)
                continue
            for annotator_name in annotator_names:
                pre_allocated_out = None
                if cam_id in self._output_buffers and annotator_name in self._output_buffers[cam_id]:
                    pre_allocated_out = self._output_buffers[cam_id][annotator_name]

                out, info = self.tiled_cameras[cam_id].get_data(annotator_name, out=pre_allocated_out)

                # Convert out to numpy if it's a warp array (only convert once, reuse buffer)
                if hasattr(out, "numpy"):
                    out_np = out.numpy()
                elif hasattr(out, "cpu"):
                    out_np = out.cpu().numpy()
                else:
                    out_np = out

                env_list = []
                for env_id in env_ids:
                    env_list.append({"data": out_np[env_id], "info": info})

                cam_data[annotator_name] = env_list
            data.append(cam_data)
        return data

    def reset(
        self,
    ):
        """
        Soft Reset do not need to reset the replicator writer and camera.
        Only Hard Reset need which means if we reset simulation backend we need to initialize camera again
        Since Render product change, we also need to attch a new writer maybe
        """
        for slot in self._standalone_slots.values():
            self._destroy_standalone_slot(slot)
        self._standalone_slots.clear()
        self.init_cameras()

    def destroy(self):
        """
        Destroy the capture manager.
        This function will be called when we close the environment.
        """

        self.annotator.clear()
        self.annotator_type.clear()
        self.annotator_device.clear()
        self.tiled_cameras.clear()
        self.cameras.clear()
        self.camera_names.clear()
        self.sim = None
        for rp in self.tiled_render_products:
            rp.destroy()
        for slot in self._standalone_slots.values():
            self._destroy_standalone_slot(slot)
        self._standalone_slots.clear()
        self.camera_prim_paths.clear()
