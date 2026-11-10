"""Client script for running Cosmos-Policy on Franka robot.

This script runs on the host workstation and:
1. Captures images from RealSense cameras
2. Connects to the Franka NUC via ZeroRPC
3. Sends observations to the Cosmos-Policy server (Docker container) via HTTP
4. Receives actions and executes them on the robot

Usage:
    # With real hardware:
    python -m cosmos_policy.experiments.robot.franka.run_franka_eval \
        --nuc-ip <NUC_IP> \
        --external-camera <EXTERNAL_CAMERA_SERIAL> \
        --wrist-camera <WRIST_CAMERA_SERIAL> \
        --left-camera <LEFT_CAMERA_SERIAL> \
        --server-ip 127.0.0.1 --server-port 8777

    # With mock hardware (testing):
    python -m cosmos_policy.experiments.robot.franka.run_franka_eval \
        --use-mock-cameras --use-mock-robot
"""

import base64
import contextlib
import datetime
import json
import os
import signal
import time
from typing import Callable, Optional, Literal

import cv2
import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation as R
import tyro
try:
    from moviepy import ImageSequenceClip
except ImportError:
    ImageSequenceClip = None

from cosmos_policy.datasets.dataset_utils import resize_images
from cosmos_policy.experiments.robot.franka import camera_utils
from cosmos_policy.experiments.robot.franka import config as cfg
from cosmos_policy.experiments.robot.franka.franka_interface import FrankaInterface, MockRobot

IMAGE_RESOLUTION = 224

GRIPPER_CLOSE_THRESHOLD = 0.03  # meters (30mm)
EXPECTED_START_GRIPPER_WIDTH = 0.08  # meters
EXPECTED_START_GRIPPER_WIDTH_TOLERANCE = 0.001  # meters
OPENVLA_MIN_Y = -0.2  # Workspace guard copied from OpenVLA Franka control.


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until after policy inference completes."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to Euler [roll, pitch, yaw]."""
    rot = R.from_quat(quat)
    return rot.as_euler("xyz", degrees=False)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler [roll, pitch, yaw] to quaternion [qx, qy, qz, qw]."""
    rot = R.from_euler("xyz", euler, degrees=False)
    return rot.as_quat()


def wrap_to_pi(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return ((array + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def canonicalize_lerobot_roll(roll: np.ndarray) -> np.ndarray:
    return wrap_to_pi(np.asarray(roll, dtype=np.float32) - np.pi)


def canonicalize_lerobot_eef_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).copy()
    if state.shape[-1] < 4:
        raise ValueError(f"Expected EEF state with at least 4 dims, got shape {state.shape}")
    state[..., 3] = canonicalize_lerobot_roll(state[..., 3])
    return state.astype(np.float32)


def apply_euler_delta(
    current_euler: np.ndarray,
    delta_euler: np.ndarray,
    mode: Literal["add", "multiply"] = "add",
) -> np.ndarray:
    """Apply an Euler delta to the current Euler angles ("add": elementwise sum, "multiply": rotation composition)."""
    if mode == "add":
        return current_euler + delta_euler
    if mode == "multiply":
        current_rot = R.from_euler("xyz", current_euler, degrees=False)
        delta_rot = R.from_euler("xyz", delta_euler, degrees=False)
        return (delta_rot * current_rot).as_euler("xyz", degrees=False)
    raise ValueError(f"Unknown Euler delta mode: {mode}")


def apply_openvla_delta_pose(
    current_pose: np.ndarray,
    delta: np.ndarray,
    position_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """Apply OpenVLA Franka delta control semantics to an EE pose.

    Mirrors openvla-oft Franka control: deltas are integrated from the previous
    commanded target pose, position is added directly, RPY deltas are added in
    Euler space, and only the y-axis lower workspace guard is applied.
    """
    current_pose = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    if current_pose.shape[0] != 7:
        raise ValueError(f"Expected 7D pose [x,y,z,qx,qy,qz,qw], got shape {current_pose.shape}")
    if delta.shape[0] != 6:
        raise ValueError(f"Expected 6D delta [dx,dy,dz,droll,dpitch,dyaw], got shape {delta.shape}")

    position = current_pose[:3].copy()
    quat = normalize_quaternion(current_pose[3:7])

    new_position = position + delta[:3] * position_scale
    new_position[1] = max(new_position[1], OPENVLA_MIN_Y)

    current_rot_euler = R.from_quat(quat).as_euler("xyz", degrees=False)
    delta_rpy = delta[3:6] * rotation_scale
    new_rot_euler = current_rot_euler + delta_rpy
    new_quat = R.from_euler("xyz", new_rot_euler, degrees=False).as_quat()

    return np.concatenate([new_position, new_quat]).astype(np.float32)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion [qx, qy, qz, qw]."""
    quat = np.asarray(quat, dtype=np.float32)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-8:
        raise ValueError(f"Invalid quaternion with near-zero norm: {quat}")
    return quat / quat_norm


def clip_quaternion_delta(current_quat: np.ndarray, target_quat: np.ndarray, max_rotation_delta: float) -> np.ndarray:
    """Move toward target quaternion while clipping the relative rotation magnitude."""
    current_rot = R.from_quat(normalize_quaternion(current_quat))
    target_rot = R.from_quat(normalize_quaternion(target_quat))
    delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
    delta_angle = np.linalg.norm(delta_rotvec)
    if delta_angle > max_rotation_delta and delta_angle > 1e-8:
        delta_rotvec = delta_rotvec * (max_rotation_delta / delta_angle)
        target_rot = R.from_rotvec(delta_rotvec) * current_rot
    return target_rot.as_quat().astype(np.float32)


def resize_policy_image(image: np.ndarray, target_size: int = IMAGE_RESOLUTION) -> np.ndarray:
    """Resize a single RGB image the same way training does: direct square resize, no padding."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    return resize_images(image[None, ...], target_size)[0]


def apply_policy_image_aug_transform(image: np.ndarray, target_size: int = IMAGE_RESOLUTION) -> np.ndarray:
    """Mirror the deterministic 90% center-crop image transform used by the deploy server."""
    image = resize_policy_image(image, target_size)
    crop_size = int(target_size * (0.9 ** 0.5))
    top = (target_size - crop_size) // 2
    left = (target_size - crop_size) // 2
    cropped = image[top : top + crop_size, left : left + crop_size]
    resample = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(Image.fromarray(cropped).resize((target_size, target_size), resample=resample), dtype=np.uint8)


def _parse_csv_vector(raw: str, *, expected_len: int, name: str) -> np.ndarray:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated {name} values, got {len(values)}: {raw!r}")
    return np.asarray(values, dtype=np.float64)


def _sample_uniform_range(low: np.ndarray, high: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if low.shape != high.shape:
        raise ValueError(f"Range shape mismatch: low={low.shape}, high={high.shape}")
    if np.any(high < low):
        raise ValueError(f"Range high must be >= low for every axis: low={low}, high={high}")
    return rng.uniform(low=low, high=high).astype(np.float64)


def _resolve_instruction(raw: str, default_prompt: Optional[str]) -> Optional[str]:
    raw = raw.strip()
    if not raw and default_prompt:
        return default_prompt
    aliases = {
        "default": default_prompt or "Sweep the green block into the dustpan",
        "defalut": default_prompt or "Sweep the green block into the dustpan",
        "sweep": "Sweep the green block into the dustpan",
        "d_sweep": "Sweep the green block into the dustpan",
        "cup": "Place the cup on the green coaster",
        "d_cup": "Place the cup on the green coaster",
        "build": "Place the orange block on top of the gray block",
        "build_block": "Place the orange block on top of the gray block",
        "d_build": "Place the orange block on top of the gray block",
    }
    return aliases.get(raw.lower(), raw or None)


def _move_to_start_joints(
    runner: "FrankaPolicyRunner",
    joints: np.ndarray,
    move_time: float,
    settle_time: float,
    label: str = "randomized start",
) -> None:
    if runner.robot is None:
        raise RuntimeError(f"Robot is not initialized; call runner.setup() before moving to {label} joints.")
    joint_csv = ", ".join(f"{value:.9f}" for value in joints)
    print(f"[Franka eval] Moving to {label} joints: {joint_csv}")
    try:
        runner.robot.terminate_current_policy()
    except Exception:
        pass
    runner.robot.move_to_joint_positions(joints, time_to_go=float(move_time))
    if settle_time > 0:
        time.sleep(float(settle_time))
    try:
        runner.robot.terminate_current_policy()
    except Exception:
        pass
    runner._start_robot_controller()


class FrankaPolicyRunner:
    """Manages robot control with remote Cosmos-Policy inference."""

    def __init__(self, config: cfg.Config, server_url: str):
        self.config = config
        self.server_url = server_url
        self.robot: Optional[FrankaInterface] = None
        self.external_cam = None
        self.wrist_cam = None
        self.left_cam = None

        # Safety limits
        self.max_position_delta = 0.05   # 5cm per step
        self.max_vertical_delta = 0.01  # 10mm per step, symmetric up/down safety clamp
        self.max_rotation_delta = 0.2    # ~11 degrees per step

        # Gripper state management
        self.last_gripper_state = None
        self.prev_gripper_cmd = None
        self.current_gripper_state = None
        self.gripper_cooldown_steps = 10
        self.gripper_release_confirm_steps = self._get_gripper_release_confirm_steps()
        self.consecutive_gripper_open_cmds = 0
        self.consecutive_gripper_close_cmds = 0
        self.pending_gripper_release_step = None
        self.pending_gripper_release_source_step = None
        self.pending_gripper_release_offset = None
        self.pending_gripper_close_step = None
        self.pending_gripper_close_source_step = None
        self.pending_gripper_close_offset = None
        self.steps_since_last_gripper_cmd = 0
        self.active_min_target_z = self.config.policy.min_target_z
        self.active_closed_gripper_min_target_z = self.config.policy.closed_gripper_min_target_z
        self.active_z_floor_task = "global"

        # Trajectory tracking
        self.chunk_predicted_actions = []
        self.chunk_actual_states = []
        self.chunk_initial_state = None
        self._last_target_pose = None
        self._last_execution_debug = None

        # Camera display
        self._frame_dir = None
        self._viewer_proc = None
        self._debug_dir = None
        self._debug_query_index = 0
        self._debug_stems_by_step = {}
        self._debug_predictions_by_step = {}
        self._warned_openvla_delta_base = False

        # Optional per-step callback for external data collectors. Receives a dict per
        # control step after the action has been issued; see run_rollout for keys.
        self._on_frame: Optional[Callable[[dict], None]] = None
        # External callers (e.g. data collector watching keyboard) can set this to
        # request a graceful early break from the run_rollout loop after the
        # current step finishes.
        self._early_stop_requested: bool = False

    def _close_gripper_before_rollout(self) -> None:
        """Optionally close the gripper once before the policy loop begins."""
        if not self.config.close_gripper_on_rollout_start:
            return

        print("Closing gripper before rollout start...")
        self.robot.control_gripper(True)
        time.sleep(2.0)
        current_gripper_width = self._get_current_gripper_width()
        self.current_gripper_state = True
        self.last_gripper_state = True
        self.steps_since_last_gripper_cmd = self.gripper_cooldown_steps
        print(
            f"  [GRIPPER] Startup state: {'CLOSED' if self.last_gripper_state else 'OPEN'} "
            f"(width: {current_gripper_width*1000:.1f}mm)"
        )

    def _set_start_gripper_width_before_rollout(self) -> None:
        """Optionally command a measured gripper width before the policy loop begins."""
        start_width = self.config.start_gripper_width
        if start_width is None:
            return

        start_width = float(start_width)
        print(f"Setting gripper width before rollout start: {start_width*1000:.1f}mm")
        self.robot.set_gripper_position(start_width)
        time.sleep(2.0)
        current_gripper_width = self._get_current_gripper_width()
        self.current_gripper_state = current_gripper_width < GRIPPER_CLOSE_THRESHOLD
        self.last_gripper_state = self.current_gripper_state
        self.steps_since_last_gripper_cmd = self.gripper_cooldown_steps
        print(
            f"  [GRIPPER] Startup state: {'CLOSED' if self.last_gripper_state else 'OPEN'} "
            f"(width: {current_gripper_width*1000:.1f}mm)"
        )

    def _get_current_gripper_width(self) -> float:
        """Read the current gripper opening width from either width or per-finger measurements."""
        if self.robot is None:
            raise RuntimeError("Robot is not initialized; call setup() before reading gripper width.")

        gripper_measurement = np.asarray(self.robot.get_gripper_position(), dtype=np.float32).reshape(-1)
        if gripper_measurement.size == 1:
            return float(abs(gripper_measurement[0]))
        if gripper_measurement.size >= 2:
            return self._get_gripper_width_from_state(gripper_measurement[:2])
        raise ValueError(f"Unexpected gripper measurement shape: {gripper_measurement.shape}")

    def _verify_start_gripper_width_before_rollout(self) -> None:
        """Fail fast if the rollout does not begin from the expected open gripper width."""
        if self.config.robot.use_mock_robot:
            print("  [GRIPPER CHECK] Skipping start-width check for mock robot.")
            return
        if self.robot is None:
            raise RuntimeError("Robot is not initialized; call setup() before checking gripper width.")

        current_gripper_width = self._get_current_gripper_width()
        expected_width = float(EXPECTED_START_GRIPPER_WIDTH)
        tolerance = float(EXPECTED_START_GRIPPER_WIDTH_TOLERANCE)
        width_error = abs(current_gripper_width - expected_width)
        print(
            "  [GRIPPER CHECK] Start width "
            f"{current_gripper_width:.4f}m, expected {expected_width:.4f}m +/- {tolerance:.4f}m"
        )
        if width_error > tolerance:
            raise RuntimeError(
                "Gripper start width check failed: "
                f"measured {current_gripper_width:.4f}m, expected {expected_width:.4f}m +/- {tolerance:.4f}m. "
                "Adjust the gripper to about 0.08m before starting the rollout."
            )

    def setup(self):
        """Initialize all components."""
        print("=" * 70)
        print("Setting up Franka Cosmos-Policy Runner (Client)")
        print("=" * 70)

        # 1. Setup cameras
        self._setup_cameras()

        # 2. Connect to robot
        self._connect_robot()

        # 3. Verify server connection (replaces _load_policy)
        # self._verify_server()

        # 4. Start camera display if enabled
        if self.config.show_cameras:
            self._start_display_thread()

        print("\n" + "=" * 70)
        print("Setup complete! Ready to run.")
        print("=" * 70 + "\n")

    def _verify_server(self):
        """Verify the inference server is reachable."""
        print(f"\n[3/3] Verifying inference server at {self.server_url}...")
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            print(f"  Server status: {data.get('status', 'unknown')}")
            print(f"  Model loaded: {data.get('model_loaded', 'unknown')}")
            print("  ✓ Server is reachable")
        except requests.exceptions.ConnectionError:
            print(f"  [WARNING] Server not reachable at {self.server_url}")
            print("  Make sure the server is running in the Docker container.")
            print("  Continuing anyway (will retry on first inference)...")
        except Exception as e:
            print(f"  [WARNING] Server health check failed: {e}")
            print("  Continuing anyway...")

    def _send_to_server(self, obs: dict) -> dict:
        """Send observation to inference server and get actions.

        Args:
            obs: Observation dict with keys:
                primary_image, left_wrist_image, right_wrist_image, proprio, task_description

        Returns:
            dict with "actions" key containing action chunk as np.ndarray
        """
        # Convert numpy arrays to lists for JSON serialization
        payload = {
            "primary_image": obs["primary_image"].tolist(),
            "left_wrist_image": obs["left_wrist_image"].tolist(),
            "right_wrist_image": obs["right_wrist_image"].tolist(),
            "proprio": obs["proprio"].tolist(),
            "task_description": obs["task_description"],
        }
        if self.config.debug_predictions:
            payload["debug_predictions"] = True

        resp = requests.post(f"{self.server_url}/act", json=payload, timeout=6000)
        resp.raise_for_status()
        result = resp.json()

        inference_time_ms = result.get("inference_time_ms", 0)

        return {
            "actions": result["actions"],
            "inference_time_ms": inference_time_ms,
            "future_image_predictions": result.get("future_image_predictions"),
            "current_proprio_normalized": result.get("current_proprio_normalized"),
            "future_proprio": result.get("future_proprio"),
            "future_proprio_normalized": result.get("future_proprio_normalized"),
            "value_prediction": result.get("value_prediction"),
        }

    def _start_display_thread(self):
        """Start the camera display by creating shared frame directory and launching viewer."""
        import os
        import subprocess

        if self._viewer_proc is not None:
            return

        self._frame_dir = os.path.join("/tmp", f"robot_camera_frames_{os.getuid()}")
        os.makedirs(self._frame_dir, exist_ok=True)

        viewer_script = os.path.join(os.path.dirname(__file__), "camera_viewer.py")
        if os.path.exists(viewer_script):
            try:
                self._viewer_proc = subprocess.Popen(
                    ["python3", viewer_script],
                    start_new_session=True
                )
                print(f"  ✓ Camera viewer launched (PID: {self._viewer_proc.pid})")
            except Exception as e:
                print(f"  [WARNING] Failed to launch camera viewer: {e}")
                self._viewer_proc = None
        else:
            print(f"  [WARNING] Camera viewer script not found: {viewer_script}")

        print(f"  ✓ Camera frames saved to: {self._frame_dir}")

    def _stop_display_thread(self):
        """Stop the camera display."""
        import os

        if hasattr(self, '_viewer_proc') and self._viewer_proc is not None:
            try:
                self._viewer_proc.terminate()
                self._viewer_proc.wait(timeout=2)
            except Exception:
                try:
                    self._viewer_proc.kill()
                except Exception:
                    pass
            self._viewer_proc = None

        try:
            if self._frame_dir:
                frame_file = os.path.join(self._frame_dir, "combined_frame.npy")
                step_file = os.path.join(self._frame_dir, "step.txt")
                if os.path.exists(frame_file):
                    os.remove(frame_file)
                if os.path.exists(step_file):
                    os.remove(step_file)
        except Exception:
            pass
        print("  ✓ Camera display stopped")

    def _update_display(self, external_img_rgb, wrist_img_rgb, left_img_rgb, t_step, resolution=224):
        """Update the display frame by saving to shared file."""
        import os

        if self._frame_dir is None:
            return False

        try:
            display_external = camera_utils.resize_with_pad(external_img_rgb, 320, 180)
            display_wrist = camera_utils.resize_with_pad(wrist_img_rgb, 320, 180)
            display_left = camera_utils.resize_with_pad(left_img_rgb, 320, 180)

            display_external = cv2.cvtColor(display_external, cv2.COLOR_RGB2BGR)
            display_wrist = cv2.cvtColor(display_wrist, cv2.COLOR_RGB2BGR)
            display_left = cv2.cvtColor(display_left, cv2.COLOR_RGB2BGR)

            combined_view = np.hstack([display_external, display_wrist, display_left])

            cv2.putText(combined_view, "External", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined_view, "Wrist", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined_view, "Left", (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            combined_view = np.ascontiguousarray(combined_view, dtype=np.uint8)

            frame_file = os.path.join(self._frame_dir, "combined_frame.npy")
            temp_file = frame_file + ".tmp.npy"
            np.save(temp_file, combined_view)
            os.replace(temp_file, frame_file)

            step_file = os.path.join(self._frame_dir, "step.txt")
            with open(step_file, 'w') as f:
                f.write(str(t_step))

            if t_step == 0:
                print(f"    [Display] Saved frame shape: {combined_view.shape}, dtype: {combined_view.dtype}")

            return True
        except Exception as e:
            print(f"  [WARNING] Failed to update display: {e}")
            return False

    def _check_user_quit(self):
        """Check if user requested quit."""
        return False

    def _setup_cameras(self):
        """Initialize RealSense cameras."""
        print("\n[1/3] Setting up cameras...")

        if self.config.camera.use_mock_cameras:
            print("  Using MOCK cameras (no real hardware)")
            self.external_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.wrist_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.left_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
        else:
            devices = camera_utils.list_realsense_devices()
            print(f"  Found {len(devices)} RealSense device(s)")

            if not devices:
                raise RuntimeError(
                    "No RealSense cameras found! "
                    "Run 'python camera_utils.py' to list devices, "
                    "or use --use-mock-cameras for testing."
                )

            for i, dev in enumerate(devices):
                print(f"    [{i}] {dev['name']} (Serial: {dev['serial_number']})")

            print("  Policy camera mapping:")
            print(
                "    primary_image     <- EXTERNAL_CAMERA "
                f"({self.config.camera.external_camera_serial})"
            )
            print(
                "    left_wrist_image  <- LEFT_CAMERA "
                f"({self.config.camera.left_camera_serial})"
            )
            print(
                "    right_wrist_image <- WRIST_CAMERA "
                f"({self.config.camera.wrist_camera_serial})"
            )

            print(f"  Initializing external camera (serial: {self.config.camera.external_camera_serial})...")
            self.external_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.external_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
            print(f"  Initializing wrist camera (serial: {self.config.camera.wrist_camera_serial})...")
            self.wrist_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.wrist_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
                powerline_frequency='60Hz',
                reset_on_release=True,
            )
            print(f"  Initializing left camera (serial: {self.config.camera.left_camera_serial})...")
            self.left_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.left_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )

        print("  ✓ Cameras ready")

    def _connect_robot(self):
        """Connect to Franka robot via ZeroRPC."""
        if self.config.robot.use_mock_robot:
            print("\n[2/3] Using MOCK robot (no real hardware)...")
            self.robot = MockRobot(ip=self.config.robot.nuc_ip, port=self.config.robot.nuc_port)
        else:
            print(f"\n[2/3] Connecting to Franka NUC at {self.config.robot.nuc_ip}:{self.config.robot.nuc_port}...")
            try:
                self.robot = FrankaInterface(ip=self.config.robot.nuc_ip, port=self.config.robot.nuc_port)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to robot: {e}")

        # Terminate any residual policy from a previous crashed session
        try:
            self.robot.terminate_current_policy()
            print("  ✓ Cleared residual controller (if any)")
        except Exception:
            pass  # No residual policy running, safe to ignore

        joint_pos = self.robot.get_joint_positions()
        gripper_pos = self.robot.get_gripper_position()

        # Test connection by getting robot state
        print("  ✓ Connected to robot")
        print(f"    Joint positions: {joint_pos}")
        print(f"    Gripper position: {gripper_pos}")

        self._start_robot_controller()

    def _start_robot_controller(self):
        """Start the robot controller configured for the current control mode."""
        # Initialize robot controller based on control mode
        if self.config.robot.control_mode == "joint":
            print("  Starting joint impedance controller...")
            self.robot.start_joint_impedance(Kq=None, Kqd=None)
        elif self.config.robot.control_mode == "eef":
            print("  Starting Cartesian impedance controller...")
            # Stiffness and damping for position (x,y,z) and orientation (rx,ry,rz)
            Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
            Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
            self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
            # Note: End-effector poses use quaternion format: [x, y, z, qx, qy, qz, qw]

        print("  ✓ Robot controller started")

    def _restart_robot_controller(self):
        """Start a controller after detecting that the previous one is gone."""
        if self.robot is None:
            return
        print("  [WARNING] Robot controller was not running; restarting controller and retrying action.")
        self._start_robot_controller()

    @staticmethod
    def _is_no_controller_running_error(error: Exception) -> bool:
        message = str(error)
        return (
            "no controller running" in message
            or "Tried to perform a controller update with no controller running" in message
        )

    def _get_robot_state(self, target_pose=None) -> np.ndarray:
        """Get raw measured robot state: [x, y, z, roll, pitch, yaw, gripper_1, gripper_2]."""
        del target_pose  # Policy proprio should come from measured robot state, not the previous target command.
        ee_pose = self.robot.get_ee_pose()

        position = ee_pose[:3]
        quat = ee_pose[3:]
        euler = quaternion_to_euler(quat)

        gripper_width = self._get_current_gripper_width()
        gripper_1 = gripper_width / 2.0
        gripper_2 = -gripper_width / 2.0

        state = np.concatenate([position, euler, [gripper_1, gripper_2]]).astype(np.float32)
        return state

    def _pose_to_eef_state(self, ee_pose: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
        """Convert pose [x, y, z, qx, qy, qz, qw] + gripper state into policy 8D EEF state."""
        ee_pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
        gripper_state = np.asarray(gripper_state, dtype=np.float32).reshape(-1)
        if ee_pose.shape[0] != 7:
            raise ValueError(f"Expected 7D EE pose, got shape {ee_pose.shape}")
        if gripper_state.shape[0] != 2:
            raise ValueError(f"Expected 2D gripper state, got shape {gripper_state.shape}")
        return np.concatenate([ee_pose[:3], quaternion_to_euler(ee_pose[3:7]), gripper_state]).astype(np.float32)

    def _summarize_proprio_debug(self, proprio) -> Optional[dict]:
        """Expose the 8D EEF layout and optional 9th padding dim used by the trained model."""
        if proprio is None:
            return None
        arr = np.asarray(self._decode_json_numpy(proprio), dtype=np.float32).reshape(-1)
        payload = {
            "dim": int(arr.shape[0]),
            "values": arr.copy(),
        }
        if arr.shape[0] >= 8:
            payload["xyzrpy_gripper_8d"] = arr[:8].copy()
            payload["gripper_dims"] = arr[6:8].copy()
            payload["padded9_expected_layout"] = np.concatenate([arr[:8], np.zeros(1, dtype=np.float32)]).astype(
                np.float32
            )
        if arr.shape[0] >= 9:
            payload["padding_dim"] = float(arr[8])
            payload["padding_is_close_to_zero"] = bool(np.isclose(arr[8], 0.0, atol=1e-4))
        return payload

    def _get_gripper_width_from_state(self, state: np.ndarray) -> float:
        """Recover gripper width from the 8D EEF state."""
        gripper_state = np.asarray(state[6:8], dtype=np.float32)
        if gripper_state[0] >= 0.0 and gripper_state[1] >= 0.0:
            return float(gripper_state.mean() * 2.0)
        return float(abs(gripper_state[0] - gripper_state[1]))

    def _interpret_gripper_close(self, gripper_cmd: float) -> bool:
        """Interpret the policy gripper scalar as a binary close command."""
        if self.config.policy.gripper_positive_action == "close":
            return bool(float(gripper_cmd) > 0.5)
        if self.config.policy.gripper_positive_action == "open":
            return bool(float(gripper_cmd) <= 0.5)
        raise ValueError(
            "Unsupported policy gripper_positive_action: "
            f"{self.config.policy.gripper_positive_action}"
        )

    def _get_gripper_release_confirm_steps(self) -> int:
        configured_steps = self.config.policy.gripper_release_confirm_steps
        if configured_steps is not None:
            return max(1, int(configured_steps))
        return max(2, 3 * int(self.config.policy.open_loop_horizon))

    def _is_build_block_task(self, instruction: str) -> bool:
        normalized_instruction = " ".join(str(instruction).lower().split())
        return "orange block" in normalized_instruction and "gray block" in normalized_instruction

    def _configure_task_z_floors(self, instruction: str) -> None:
        self.active_min_target_z = self.config.policy.min_target_z
        self.active_closed_gripper_min_target_z = self.config.policy.closed_gripper_min_target_z
        self.active_z_floor_task = "global"

        if self._is_build_block_task(instruction):
            if self.config.policy.build_task_min_target_z is not None:
                self.active_min_target_z = self.config.policy.build_task_min_target_z
                self.active_z_floor_task = "build_block"
            if self.config.policy.build_task_closed_gripper_min_target_z is not None:
                self.active_closed_gripper_min_target_z = self.config.policy.build_task_closed_gripper_min_target_z
                self.active_z_floor_task = "build_block"

    def _active_target_z_floor(self) -> tuple[Optional[float], str]:
        z_floor = self.active_min_target_z
        reason = self.active_z_floor_task if z_floor is not None else "none"
        closed_floor = self.active_closed_gripper_min_target_z
        closed_floor_ready = (
            closed_floor is not None
            and bool(self.current_gripper_state)
            and self.steps_since_last_gripper_cmd >= self.gripper_cooldown_steps
        )
        if closed_floor_ready and self.active_z_floor_task == "build_block":
            # For block stacking, the stricter closed-gripper z floor is meant
            # for the placement phase onto the support block, not the initial
            # descent/grasp. Only arm it once the predicted action chunk already
            # contains a future release/open segment.
            closed_floor_ready = self.pending_gripper_release_step is not None
        if closed_floor_ready and (z_floor is None or closed_floor > z_floor):
            z_floor = closed_floor
            reason = f"{self.active_z_floor_task}:closed_gripper"
        return z_floor, reason

    def _apply_target_z_floor(self, target_pos: np.ndarray, measured_pos: np.ndarray):
        z_floor, z_floor_reason = self._active_target_z_floor()
        target_pos_before_z_floor = target_pos.copy()
        z_floor_clipped = False
        if z_floor is not None and target_pos[2] < z_floor:
            target_pos = target_pos.copy()
            # This guard is intended to be a true absolute lower bound on the
            # commanded target z. Do not soften it with the per-step vertical
            # delta clamp; otherwise build-block placement can still command
            # sub-floor target poses for several steps while "chasing" the floor.
            target_pos[2] = max(float(target_pos[2]), float(z_floor))
            z_floor_clipped = True
        return target_pos, target_pos_before_z_floor, z_floor, z_floor_reason, z_floor_clipped

    def _open_gripper_command_value(self) -> float:
        if self.config.policy.gripper_positive_action == "open":
            return 1.0
        return 0.0

    def _closed_gripper_command_value(self) -> float:
        if self.config.policy.gripper_positive_action == "open":
            return 0.0
        return 1.0

    def _find_chunk_gripper_run_start(self, action_chunk: np.ndarray, desired_close: bool) -> Optional[int]:
        confirm_steps = max(1, int(self.config.policy.gripper_chunk_lookahead_confirm_steps))
        consecutive_matches = 0
        for idx, action in enumerate(np.asarray(action_chunk)):
            gripper_close = self._interpret_gripper_close(float(action[-1]))
            if gripper_close == desired_close:
                consecutive_matches += 1
                if consecutive_matches >= confirm_steps:
                    return idx - confirm_steps + 1
            else:
                consecutive_matches = 0
        return None

    def _find_chunk_open_run_start(self, action_chunk: np.ndarray) -> Optional[int]:
        return self._find_chunk_gripper_run_start(action_chunk, desired_close=False)

    def _find_chunk_close_run_start(self, action_chunk: np.ndarray) -> Optional[int]:
        return self._find_chunk_gripper_run_start(action_chunk, desired_close=True)

    def _gripper_lookahead_min_offset(self) -> int:
        configured_offset = self.config.policy.gripper_chunk_lookahead_min_offset
        if configured_offset is not None:
            return max(0, int(configured_offset))
        return max(0, int(self.config.policy.open_loop_horizon))

    def _update_gripper_close_lookahead(self, action_chunk: np.ndarray, chunk_start_step: int) -> None:
        if not self.config.policy.gripper_chunk_lookahead_close:
            return
        if bool(self.current_gripper_state):
            return

        close_run_start = self._find_chunk_close_run_start(action_chunk)
        if close_run_start is None:
            return
        if close_run_start < self._gripper_lookahead_min_offset():
            return

        close_step = chunk_start_step + close_run_start
        if self.pending_gripper_close_step is None or close_step < self.pending_gripper_close_step:
            self.pending_gripper_close_step = close_step
            self.pending_gripper_close_source_step = chunk_start_step
            self.pending_gripper_close_offset = close_run_start
            print(
                "  [GRIPPER] Scheduled close from chunk lookahead: "
                f"step={close_step}, offset={close_run_start}"
            )

    def _update_gripper_release_lookahead(self, action_chunk: np.ndarray, chunk_start_step: int) -> None:
        if not self.config.policy.gripper_chunk_lookahead_release:
            return
        if not bool(self.current_gripper_state):
            return
        if self.steps_since_last_gripper_cmd < self.gripper_cooldown_steps:
            return

        open_run_start = self._find_chunk_open_run_start(action_chunk)
        if open_run_start is None:
            return
        if open_run_start < self._gripper_lookahead_min_offset():
            return

        release_step = chunk_start_step + open_run_start
        if self.pending_gripper_release_step is None or release_step < self.pending_gripper_release_step:
            self.pending_gripper_release_step = release_step
            self.pending_gripper_release_source_step = chunk_start_step
            self.pending_gripper_release_offset = open_run_start
            print(
                "  [GRIPPER] Scheduled release from chunk lookahead: "
                f"step={release_step}, offset={open_run_start}"
            )

    def _update_gripper_chunk_lookahead(self, action_chunk: np.ndarray, chunk_start_step: int) -> None:
        self._update_gripper_close_lookahead(action_chunk, chunk_start_step)
        self._update_gripper_release_lookahead(action_chunk, chunk_start_step)

    def _should_force_gripper_close(self, t_step: int) -> bool:
        return (
            self.pending_gripper_close_step is not None
            and t_step >= self.pending_gripper_close_step
            and not bool(self.current_gripper_state)
        )

    def _should_force_gripper_open(self, t_step: int) -> bool:
        return (
            self.pending_gripper_release_step is not None
            and t_step >= self.pending_gripper_release_step
            and bool(self.current_gripper_state)
        )

    def _rpy_action_lock_mask(self) -> np.ndarray:
        """Return a [roll, pitch, yaw] boolean mask for orientation action locking."""
        if self.config.policy.lock_rpy_actions:
            return np.ones(3, dtype=bool)

        axes = str(self.config.policy.lock_rpy_action_axes or "").strip()
        mask = np.zeros(3, dtype=bool)
        if not axes:
            return mask

        aliases = {
            "0": 0,
            "r": 0,
            "roll": 0,
            "droll": 0,
            "1": 1,
            "p": 1,
            "pitch": 1,
            "dpitch": 1,
            "2": 2,
            "y": 2,
            "yaw": 2,
            "dyaw": 2,
        }
        normalized_axes = axes.replace(";", ",").replace(" ", ",")
        for token in normalized_axes.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token in ("all", "rpy"):
                mask[:] = True
                continue
            if token in ("none", "false", "off"):
                continue
            if token not in aliases:
                raise ValueError(
                    "Unsupported lock_rpy_action_axes token "
                    f"{token!r}; use roll,pitch,yaw or 0,1,2."
                )
            mask[aliases[token]] = True
        return mask

    def _rpy_action_lock_axes_label(self, mask: Optional[np.ndarray] = None) -> str:
        """Human-readable label for the active orientation action lock mask."""
        if mask is None:
            mask = self._rpy_action_lock_mask()
        names = ("roll", "pitch", "yaw")
        locked = [name for name, is_locked in zip(names, mask) if is_locked]
        return ",".join(locked) if locked else "none"

    def _get_policy_proprio(self, raw_state: np.ndarray) -> np.ndarray:
        """Get the proprio vector expected by the policy server."""
        if self.config.policy.proprio_format == "eef_8d":
            policy_state = canonicalize_lerobot_eef_state(raw_state)
            if self.config.policy.gripper_proprio_mode == "zero":
                policy_state[6:8] = 0.0
            elif self.config.policy.gripper_proprio_mode != "measured_width_signed":
                raise ValueError(
                    "Unsupported policy gripper_proprio_mode: "
                    f"{self.config.policy.gripper_proprio_mode}"
                )
            return policy_state
        if self.config.policy.proprio_format == "joint_9d":
            return self.robot.get_joint_positions_w_gripper().astype(np.float32)
        raise ValueError(f"Unsupported policy proprio_format: {self.config.policy.proprio_format}")

    def _execute_action(
        self,
        pred_action: np.ndarray,
        current_state: np.ndarray,
        target_ee_pose: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        force_gripper_close: bool = False,
        force_gripper_open: bool = False,
    ):
        """
        Execute action on robot using OpenVLA-style target-pose accumulator while
        retaining the original per-step safety clipping (5cm xy / 1cm z / ~11deg).

        Args:
            pred_action (7D or 8D):
                7D delta_eef_7d: [dx, dy, dz, droll, dpitch, dyaw, gripper_binary]
                7D delta_eef_7d_z_positive_down: accepted for CLI compatibility; no axis flip is applied
                7D absolute_eef: [x, y, z, roll, pitch, yaw, gripper_binary]
                8D: [x, y, z, qx, qy, qz, qw, gripper_binary] absolute pose
            current_state (8D): Current robot state [x,y,z, roll,pitch,yaw, grip1,grip2]
            target_ee_pose: Previous commanded target pose. OpenVLA integrates deltas from this accumulator.
            dt: Control timestep (unused; safety limits are per-step constants).
        """
        if dt is None and np.isscalar(target_ee_pose):
            # Backward-compatible call shape: _execute_action(action, state, dt).
            dt = float(target_ee_pose)
            target_ee_pose = self._last_target_pose
        del dt  # Safety limits are applied once per control step.

        pred_action = np.asarray(pred_action, dtype=np.float32).reshape(-1)
        policy_state_before = np.asarray(current_state, dtype=np.float32).copy()
        execution_state = policy_state_before.copy()
        measured_pos = execution_state[:3]
        measured_euler = execution_state[3:6]
        measured_quat = euler_to_quaternion(measured_euler)
        current_pos = measured_pos
        current_euler = measured_euler
        current_quat = measured_quat
        action_mode = self.config.policy.action_mode
        delta_action_base = self.config.policy.delta_action_base
        action_base_state = execution_state
        action_base_source = "measured"
        action_base_pose = np.concatenate([measured_pos, measured_quat]).astype(np.float32)
        target_lead_before_clip = None
        target_lead_after_clip = None
        target_lead_clipped = False
        model_delta_pos = None
        control_delta_pos = None
        delta_pos_clipped = None
        target_pos_abs = None
        model_delta_euler = None
        control_delta_euler = None
        target_quat_abs = None
        rpy_action_lock_mask = self._rpy_action_lock_mask()
        rpy_action_locked = bool(np.any(rpy_action_lock_mask))
        position_clipped = False
        rotation_clipped = False

        if pred_action.shape[0] == 7:
            gripper_cmd = pred_action[6]
            if action_mode in ("delta_eef_7d", "delta_eef_7d_z_positive_down"):
                if delta_action_base != "target" and not self._warned_openvla_delta_base:
                    print(
                        "  [OpenVLA control] Ignoring requested delta_action_base="
                        f"{delta_action_base!r}; OpenVLA integrates deltas from the previous target pose."
                    )
                    self._warned_openvla_delta_base = True

                # Exact OpenVLA behavior: seed the accumulator from the measured pose once,
                # then keep integrating every delta from the previously commanded target.
                if target_ee_pose is None:
                    action_base_pose = np.asarray(self.robot.get_ee_pose(), dtype=np.float32).reshape(-1)
                    action_base_source = "measured_robot_initial_openvla"
                else:
                    action_base_pose = np.asarray(target_ee_pose, dtype=np.float32).reshape(-1)
                    action_base_source = "previous_target_openvla"
                if action_base_pose.shape[0] != 7:
                    raise ValueError(f"Expected 7D target pose accumulator, got shape {action_base_pose.shape}")

                current_pos = action_base_pose[:3]
                current_quat = normalize_quaternion(action_base_pose[3:7])
                current_euler = quaternion_to_euler(current_quat)
                action_base_state = self._pose_to_eef_state(action_base_pose, execution_state[6:8])

                # Per-step safety clipping on the policy-emitted delta.
                model_delta_pos = pred_action[:3].copy()
                if action_mode == "delta_eef_7d_z_positive_down":
                    print(
                        "  [WARNING] delta_eef_7d_z_positive_down is deprecated for "
                        "OpenVLA-style control; no z-axis flip will be applied."
                    )
                control_delta_pos = model_delta_pos.copy()
                delta_pos_clipped = np.clip(control_delta_pos, -self.max_position_delta, self.max_position_delta)
                delta_pos_clipped[2] = np.clip(
                    delta_pos_clipped[2], -self.max_vertical_delta, self.max_vertical_delta
                )
                position_clipped = not np.allclose(control_delta_pos, delta_pos_clipped)
                if position_clipped:
                    print("  [WARNING] Delta position action clipped for safety!")

                model_delta_euler = pred_action[3:6].copy()
                delta_euler = model_delta_euler.copy()
                if rpy_action_locked:
                    delta_euler[rpy_action_lock_mask] = 0.0
                delta_angle = np.linalg.norm(delta_euler)
                if delta_angle > self.max_rotation_delta and delta_angle > 1e-8:
                    delta_euler = delta_euler * (self.max_rotation_delta / delta_angle)
                    rotation_clipped = True
                    print("  [WARNING] Delta rotation action clipped for safety!")
                control_delta_euler = delta_euler.copy()

                # OpenVLA integration with the (safety-clipped) delta.
                clipped_delta = np.concatenate([delta_pos_clipped, control_delta_euler]).astype(np.float32)
                target_ee_pose = apply_openvla_delta_pose(
                    action_base_pose,
                    clipped_delta,
                    position_scale=1.0,
                    rotation_scale=1.0,
                )
                target_pos = target_ee_pose[:3]
                target_quat = target_ee_pose[3:7]
                target_quat_abs = target_quat.copy()

                # Keep target_lead clamp for safety (prevents runaway target vs. measured).
                target_lead_before_clip = target_pos - measured_pos
                target_lead_after_clip = np.clip(
                    target_lead_before_clip,
                    -self.config.policy.max_target_position_lead,
                    self.config.policy.max_target_position_lead,
                )
                target_lead_after_clip[2] = np.clip(
                    target_lead_after_clip[2],
                    -self.config.policy.max_target_vertical_lead,
                    self.config.policy.max_target_vertical_lead,
                )
                target_lead_clipped = not np.allclose(target_lead_before_clip, target_lead_after_clip)
                if target_lead_clipped:
                    print("  [WARNING] Integrated target pose clipped to measured-pose lead limit!")
                target_pos = measured_pos + target_lead_after_clip
            elif action_mode == "absolute_eef":
                target_pos_abs = pred_action[:3]
                target_euler_abs = pred_action[3:6].copy()
                if rpy_action_locked:
                    target_euler_abs[rpy_action_lock_mask] = measured_euler[rpy_action_lock_mask]
                target_quat_abs = euler_to_quaternion(target_euler_abs)
                delta_pos = target_pos_abs - measured_pos
                control_delta_pos = delta_pos.copy()
                delta_pos_clipped = np.clip(delta_pos, -self.max_position_delta, self.max_position_delta)
                delta_pos_clipped[2] = np.clip(
                    delta_pos_clipped[2], -self.max_vertical_delta, self.max_vertical_delta
                )
                position_clipped = not np.allclose(delta_pos, delta_pos_clipped)
                if position_clipped:
                    print("  [WARNING] Absolute position action clipped for safety!")
                target_pos = measured_pos + delta_pos_clipped
                target_quat = clip_quaternion_delta(measured_quat, target_quat_abs, self.max_rotation_delta)
            else:
                raise ValueError(f"Unsupported policy action_mode for 7D action: {action_mode}")
        elif pred_action.shape[0] == 8:
            if action_mode in ("delta_eef_7d", "delta_eef_7d_z_positive_down"):
                print("  [WARNING] Received 8D action; treating it as absolute_eef pose.")
            target_pos_abs = pred_action[:3]
            target_quat_abs = normalize_quaternion(pred_action[3:7])
            if rpy_action_locked:
                target_euler_abs = quaternion_to_euler(target_quat_abs)
                target_euler_abs[rpy_action_lock_mask] = measured_euler[rpy_action_lock_mask]
                target_quat_abs = euler_to_quaternion(target_euler_abs)
            gripper_cmd = pred_action[7]
            delta_pos = target_pos_abs - measured_pos
            control_delta_pos = delta_pos.copy()
            delta_pos_clipped = np.clip(delta_pos, -self.max_position_delta, self.max_position_delta)
            delta_pos_clipped[2] = np.clip(delta_pos_clipped[2], -self.max_vertical_delta, self.max_vertical_delta)
            position_clipped = not np.allclose(delta_pos, delta_pos_clipped)
            if position_clipped:
                print("  [WARNING] Absolute position action clipped for safety!")
            target_pos = measured_pos + delta_pos_clipped
            target_quat = clip_quaternion_delta(measured_quat, target_quat_abs, self.max_rotation_delta)
        else:
            raise ValueError(
                "Unsupported Franka action dimension: "
                f"got {pred_action.shape[0]}, expected 7 or 8"
            )

        raw_gripper_cmd = float(gripper_cmd)
        if force_gripper_close and force_gripper_open:
            force_gripper_open = False
        if force_gripper_close:
            gripper_cmd = self._closed_gripper_command_value()
        elif force_gripper_open:
            gripper_cmd = self._open_gripper_command_value()

        (
            target_pos,
            target_pos_before_z_floor,
            target_z_floor,
            target_z_floor_reason,
            target_z_floor_clipped,
        ) = self._apply_target_z_floor(target_pos, measured_pos)
        if target_z_floor_clipped:
            print(
                "  [Z GUARD] Target z clipped "
                f"from {target_pos_before_z_floor[2]:.4f}m to {target_pos[2]:.4f}m "
                f"(floor={target_z_floor:.4f}m, reason={target_z_floor_reason})"
            )

        target_ee_pose = np.concatenate([target_pos, target_quat]).astype(np.float32)
        try:
            self.robot.update_desired_ee_pose(target_ee_pose)
        except Exception as e:
            if not self._is_no_controller_running_error(e):
                raise
            self._restart_robot_controller()
            time.sleep(0.2)
            self.robot.update_desired_ee_pose(target_ee_pose)
        self._last_target_pose = target_ee_pose.copy()

        # Gripper control uses an asymmetric debounce. Closing is latched on the
        # first close command; opening requires a longer sustained open intent to
        # avoid dropping the object when fresh chunks start with open-like values.
        gripper_close = self._interpret_gripper_close(gripper_cmd)
        gripper_open = not gripper_close
        self.steps_since_last_gripper_cmd += 1
        gripper_state_before = self.last_gripper_state
        prev_gripper_cmd_before = self.prev_gripper_cmd
        pending_gripper_close_step_before = self.pending_gripper_close_step
        pending_gripper_close_source_step_before = self.pending_gripper_close_source_step
        pending_gripper_close_offset_before = self.pending_gripper_close_offset
        pending_gripper_release_step_before = self.pending_gripper_release_step
        pending_gripper_release_source_step_before = self.pending_gripper_release_source_step
        pending_gripper_release_offset_before = self.pending_gripper_release_offset
        prev_gripper_close = None
        if gripper_close:
            self.consecutive_gripper_close_cmds += 1
            self.consecutive_gripper_open_cmds = 0
        else:
            self.consecutive_gripper_open_cmds += 1
            self.consecutive_gripper_close_cmds = 0
        gripper_open_release_ready = False
        gripper_debounce_satisfied = False
        gripper_command_sent = False
        initial_gripper_command_deferred = False

        if self.last_gripper_state is None:
            current_gripper_width = self._get_gripper_width_from_state(execution_state)
            self.last_gripper_state = current_gripper_width < GRIPPER_CLOSE_THRESHOLD
            self.current_gripper_state = self.last_gripper_state
            print(
                f"  [GRIPPER] Initialized: {'CLOSED' if self.last_gripper_state else 'OPEN'} "
                f"(width: {current_gripper_width*1000:.1f}mm)"
            )

        if prev_gripper_cmd_before is None:
            initial_gripper_command_deferred = True
        else:
            prev_gripper_close = self._interpret_gripper_close(prev_gripper_cmd_before)
            gripper_open_release_ready = (
                not gripper_close
                and self.consecutive_gripper_open_cmds >= self.gripper_release_confirm_steps
                and self.steps_since_last_gripper_cmd >= self.gripper_cooldown_steps
            )
            gripper_debounce_satisfied = (
                force_gripper_close or force_gripper_open or gripper_close or gripper_open_release_ready
            )
            if gripper_debounce_satisfied and self.current_gripper_state != gripper_close:
                self.current_gripper_state = gripper_close
                self.robot.control_gripper(self.current_gripper_state)
                self.last_gripper_state = gripper_close
                self.steps_since_last_gripper_cmd = 0
                gripper_command_sent = True
                if gripper_close:
                    self.pending_gripper_close_step = None
                    self.pending_gripper_close_source_step = None
                    self.pending_gripper_close_offset = None
                    self.pending_gripper_release_step = None
                    self.pending_gripper_release_source_step = None
                    self.pending_gripper_release_offset = None
                else:
                    self.pending_gripper_close_step = None
                    self.pending_gripper_close_source_step = None
                    self.pending_gripper_close_offset = None
                    self.pending_gripper_release_step = None
                    self.pending_gripper_release_source_step = None
                    self.pending_gripper_release_offset = None

        self.prev_gripper_cmd = float(gripper_cmd)

        self._last_execution_debug = {
            "action_mode": action_mode,
            "delta_action_base": "target_openvla",
            "requested_delta_action_base": delta_action_base,
            "action_base_source": action_base_source,
            "policy_action": pred_action.copy(),
            "policy_state_before": policy_state_before.copy(),
            "measured_state_before": execution_state.copy(),
            "action_base_state": action_base_state.copy(),
            "action_base_pose": action_base_pose.copy(),
            "measured_pos": measured_pos.copy(),
            "measured_euler": measured_euler.copy(),
            "measured_quat": measured_quat.copy(),
            "current_pos": current_pos.copy(),
            "current_euler": current_euler.copy(),
            "current_quat": current_quat.copy(),
            "model_delta_pos": model_delta_pos,
            "control_delta_pos_after_axis_mapping": control_delta_pos,
            "control_delta_pos_clipped": delta_pos_clipped,
            "position_clipped": position_clipped,
            "target_lead_before_clip": target_lead_before_clip,
            "target_lead_after_clip": target_lead_after_clip,
            "target_lead_clipped": target_lead_clipped,
            "target_pos_abs_from_policy": target_pos_abs,
            "model_delta_euler": model_delta_euler,
            "control_delta_euler": control_delta_euler,
            "rpy_action_locked": rpy_action_locked,
            "rpy_action_lock_mask": rpy_action_lock_mask.copy(),
            "rpy_action_lock_axes": self._rpy_action_lock_axes_label(rpy_action_lock_mask),
            "rotation_clipped": rotation_clipped,
            "target_quat_abs_before_clip": target_quat_abs,
            "target_pos_before_z_floor": target_pos_before_z_floor.copy(),
            "target_z_floor": target_z_floor,
            "target_z_floor_reason": target_z_floor_reason,
            "target_z_floor_clipped": target_z_floor_clipped,
            "target_pos_sent": target_pos.copy(),
            "target_quat_sent": target_quat.copy(),
            "target_pose_sent": target_ee_pose.copy(),
            "target_delta_pos_sent": target_pos - measured_pos,
            "target_delta_pos_from_action_base": target_pos - current_pos,
            "openvla_min_y_guard": OPENVLA_MIN_Y,
            "gripper_cmd": float(gripper_cmd),
            "raw_gripper_cmd": raw_gripper_cmd,
            "force_gripper_close_from_chunk_lookahead": force_gripper_close,
            "force_gripper_open_from_chunk_lookahead": force_gripper_open,
            "gripper_open_interpreted": gripper_open,
            "gripper_close_interpreted": gripper_close,
            "gripper_state_before": gripper_state_before,
            "gripper_state_after": self.last_gripper_state,
            "previous_gripper_cmd": prev_gripper_cmd_before,
            "previous_gripper_close_interpreted": prev_gripper_close,
            "consecutive_gripper_open_cmds": self.consecutive_gripper_open_cmds,
            "consecutive_gripper_close_cmds": self.consecutive_gripper_close_cmds,
            "gripper_release_confirm_steps": self.gripper_release_confirm_steps,
            "pending_gripper_close_step_before": pending_gripper_close_step_before,
            "pending_gripper_close_source_step_before": pending_gripper_close_source_step_before,
            "pending_gripper_close_offset_before": pending_gripper_close_offset_before,
            "pending_gripper_close_step_after": self.pending_gripper_close_step,
            "pending_gripper_close_source_step_after": self.pending_gripper_close_source_step,
            "pending_gripper_close_offset_after": self.pending_gripper_close_offset,
            "pending_gripper_release_step_before": pending_gripper_release_step_before,
            "pending_gripper_release_source_step_before": pending_gripper_release_source_step_before,
            "pending_gripper_release_offset_before": pending_gripper_release_offset_before,
            "pending_gripper_release_step_after": self.pending_gripper_release_step,
            "pending_gripper_release_source_step_after": self.pending_gripper_release_source_step,
            "pending_gripper_release_offset_after": self.pending_gripper_release_offset,
            "gripper_open_release_ready": gripper_open_release_ready,
            "gripper_debounce_satisfied": gripper_debounce_satisfied,
            "gripper_command_sent": gripper_command_sent,
            "initial_gripper_command_deferred": initial_gripper_command_deferred,
            "steps_since_last_gripper_cmd": self.steps_since_last_gripper_cmd,
        }

        return target_ee_pose

    def run_rollout(self, instruction: str, max_timesteps: Optional[int] = None):
        """Execute rollout with remote inference."""
        if max_timesteps is None:
            max_timesteps = self.config.max_timesteps

        self.gripper_release_confirm_steps = self._get_gripper_release_confirm_steps()
        self._configure_task_z_floors(instruction)

        print(f"\n{'=' * 70}")
        print(f"TASK: {instruction}")
        print(f"Max timesteps: {max_timesteps}")
        print(f"Server: {self.server_url}")
        print(f"Policy proprio format: {self.config.policy.proprio_format}")
        print(f"Policy action mode: {self.config.policy.action_mode}")
        print(f"Open-loop horizon: {self.config.policy.open_loop_horizon}")
        print(
            "Gripper control: "
            f"positive={self.config.policy.gripper_positive_action}, "
            f"release_confirm_steps={self.gripper_release_confirm_steps}"
        )
        print(
            "Gripper chunk lookahead: "
            f"close={self.config.policy.gripper_chunk_lookahead_close}, "
            f"release={self.config.policy.gripper_chunk_lookahead_release}, "
            f"confirm_steps={self.config.policy.gripper_chunk_lookahead_confirm_steps}, "
            f"min_offset={self._gripper_lookahead_min_offset()}"
        )
        print(f"Lock RPY actions: {self.config.policy.lock_rpy_actions}")
        print(f"Lock RPY action axes: {self._rpy_action_lock_axes_label()}")
        print("Delta control: OpenVLA target-pose accumulator")
        print(f"Requested delta action base option: {self.config.policy.delta_action_base}")
        print(
            "Target lead limits: "
            f"xyz={self.config.policy.max_target_position_lead:.3f}m, "
            f"z={self.config.policy.max_target_vertical_lead:.3f}m"
        )
        print(
            "Target z floors: "
            f"active_task={self.active_z_floor_task}, "
            f"active_global={self.active_min_target_z}, "
            f"active_closed_gripper={self.active_closed_gripper_min_target_z}, "
            f"build=({self.config.policy.build_task_min_target_z}, "
            f"{self.config.policy.build_task_closed_gripper_min_target_z})"
        )
        print(f"{'=' * 70}")

        # Reset gripper state tracking
        self.last_gripper_state = None
        self.current_gripper_state = None
        self.prev_gripper_cmd = None
        self.gripper_release_confirm_steps = self._get_gripper_release_confirm_steps()
        self.consecutive_gripper_open_cmds = 0
        self.consecutive_gripper_close_cmds = 0
        self.pending_gripper_close_step = None
        self.pending_gripper_close_source_step = None
        self.pending_gripper_close_offset = None
        self.pending_gripper_release_step = None
        self.pending_gripper_release_source_step = None
        self.pending_gripper_release_offset = None
        self.steps_since_last_gripper_cmd = 0
        self._debug_dir = None
        self._debug_query_index = 0
        self._debug_stems_by_step = {}
        self._debug_predictions_by_step = {}
        self._last_target_pose = None
        self._warned_openvla_delta_base = False
        self._early_stop_requested = False
        self._set_start_gripper_width_before_rollout()
        self._verify_start_gripper_width_before_rollout()
        self._close_gripper_before_rollout()

        actions_from_chunk_completed = 0
        pred_action_chunk = None
        current_chunk_start_step = None

        video_frames = []
        dt = 1.0 / self.config.robot.control_frequency
        inference_times = []

        print("Running rollout... (Press Ctrl+C to stop)")

        target_pose = None
        try:
            for t_step in range(max_timesteps):
                step_start_time = time.time()

                # 1. Get robot state for control + proprio for the policy.
                raw_state = self._get_robot_state(target_pose)
                policy_proprio = self._get_policy_proprio(raw_state)

                # 2. Capture images
                ret_ext, external_img, _ = self.external_cam.read()
                ret_wrist, wrist_img, _ = self.wrist_cam.read()
                ret_left, left_img, _ = self.left_cam.read()

                if not ret_ext or not ret_wrist or not ret_left:
                    print(f"  [Step {t_step}] Failed to capture images!")
                    break

                external_img_rgb = cv2.cvtColor(external_img, cv2.COLOR_BGR2RGB)
                wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

                if self.config.save_video:
                    video_frames.append(external_img_rgb.copy())

                if self.config.show_cameras:
                    self._update_display(external_img_rgb, wrist_img_rgb, left_img_rgb, t_step)
                    if self._check_user_quit():
                        print("\n  User requested quit via display window")
                        break

                # 3. Need new action chunk?
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.config.policy.open_loop_horizon:
                    # Reset for new chunk. Note: do NOT reset _last_target_pose here —
                    # OpenVLA control keeps a single target-pose accumulator across the entire rollout.
                    actions_from_chunk_completed = 0
                    current_chunk_start_step = t_step
                    self.chunk_predicted_actions = []
                    self.chunk_actual_states = []
                    self.chunk_initial_state = raw_state.copy()

                    # Build observation for server.
                    # Align camera slots with Real-PnP-Tube training data:
                    #   primary_image     <- camera_front_color
                    #   left_wrist_image  <- camera_left_color
                    #   right_wrist_image <- camera_wrist_color
                    # Match training preprocessing: direct resize to square, no aspect-ratio padding.
                    # policy.proprio_format selects either:
                    #   eef_8d:   [x,y,z, roll,pitch,yaw, grip1,grip2]
                    #   joint_9d: [joint_1..joint_7, gripper, gripper]
                    obs = {
                        "primary_image": resize_policy_image(external_img_rgb, IMAGE_RESOLUTION),
                        "left_wrist_image": resize_policy_image(left_img_rgb, IMAGE_RESOLUTION),
                        "right_wrist_image": resize_policy_image(wrist_img_rgb, IMAGE_RESOLUTION),
                        "proprio": policy_proprio.copy(),
                        "task_description": instruction,
                    }
                    if self.config.debug_predictions:
                        self._save_debug_inputs(obs, t_step, instruction)

                    # Run inference via server
                    inference_start = time.time()

                    result = self._send_to_server(obs)
                    inference_time = (time.time() - inference_start) * 1000
                    inference_times.append(inference_time)

                    pred_action_chunk = result["actions"]
                    pred_action_chunk = self.decode_numpy_dict(pred_action_chunk)
                    server_inference_ms = result.get("inference_time_ms", 0)

                    self.chunk_predicted_actions = pred_action_chunk.copy()
                    self._update_gripper_chunk_lookahead(pred_action_chunk, t_step)
                    if self.config.debug_predictions:
                        debug_result = {
                            "future_image_predictions": self._decode_json_numpy(
                                result.get("future_image_predictions")
                            ),
                            "current_proprio_normalized": self._decode_json_numpy(
                                result.get("current_proprio_normalized")
                            ),
                            "future_proprio": self._decode_json_numpy(result.get("future_proprio")),
                            "future_proprio_normalized": self._decode_json_numpy(
                                result.get("future_proprio_normalized")
                            ),
                            "value_prediction": result.get("value_prediction"),
                            "num_queries_best_of_n": result.get("num_queries_best_of_n"),
                            "actions": pred_action_chunk,
                        }
                        self._save_debug_predictions(debug_result, t_step, instruction)

                    num_best_of_n = result.get("num_queries_best_of_n")
                    value_pred = result.get("value_prediction")
                    print(f"\n  [Step {t_step:3d}] NEW ACTION CHUNK")
                    print(f"    Server inference: {server_inference_ms:.1f}ms")
                    print(f"    Round-trip: {inference_time:.1f}ms")
                    print(f"    Chunk size: {len(pred_action_chunk)} actions")
                    print(f"    First action: {pred_action_chunk[0]}")
                    if num_best_of_n is not None and num_best_of_n > 1:
                        print(f"    Best-of-N: selected highest-value chunk from {num_best_of_n} candidates")
                    if value_pred is not None:
                        print(f"    Value prediction: {value_pred}")

                # 4. Execute action
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if t_step % 10 == 0 or actions_from_chunk_completed == 1:
                    print(f"  [Step {t_step:3d}] Action {actions_from_chunk_completed}/{len(pred_action_chunk)}")

                force_gripper_close = self._should_force_gripper_close(t_step)
                force_gripper_open = self._should_force_gripper_open(t_step)
                target_pose = self._execute_action(
                    action,
                    raw_state,
                    target_pose,
                    dt,
                    force_gripper_close=force_gripper_close,
                    force_gripper_open=force_gripper_open,
                )

                # 5. Regulate timing
                elapsed_before_sleep = time.time() - step_start_time
                slept = 0.0
                step_overran = elapsed_before_sleep >= dt
                if not step_overran:
                    slept = dt - elapsed_before_sleep
                    time.sleep(slept)
                elif t_step % 10 == 0:
                    print(
                        f"  [WARNING] Step {t_step} overran: "
                        f"{elapsed_before_sleep*1000:.1f}ms (target: {dt*1000:.1f}ms)"
                    )
                step_elapsed_total = time.time() - step_start_time

                # Store actual state after action execution
                actual_state_after = self._get_robot_state(target_pose)
                self.chunk_actual_states.append(actual_state_after.copy())
                if self._on_frame is not None:
                    try:
                        self._on_frame({
                            "t_step": t_step,
                            "step_start_time": step_start_time,
                            "instruction": instruction,
                            "external_img_rgb": external_img_rgb,
                            "wrist_img_rgb": wrist_img_rgb,
                            "left_img_rgb": left_img_rgb,
                            "raw_state": raw_state,
                            "policy_action": action,
                            "target_pose": target_pose,
                            "actual_state_after": actual_state_after,
                        })
                    except Exception as cb_err:
                        print(f"  [WARNING] _on_frame callback failed at step {t_step}: {cb_err}")
                if self._early_stop_requested:
                    print(f"\n  [Step {t_step}] Early stop requested; ending rollout.")
                    break
                if self.config.debug_predictions:
                    self._save_debug_execution(
                        instruction=instruction,
                        t_step=t_step,
                        chunk_start_step=current_chunk_start_step,
                        action_index_in_chunk=actions_from_chunk_completed - 1,
                        policy_action=action,
                        measured_state_before=raw_state,
                        target_pose_sent=target_pose,
                        measured_state_after=actual_state_after,
                        step_timing={
                            "dt_ms": dt * 1000.0,
                            "elapsed_before_sleep_ms": elapsed_before_sleep * 1000.0,
                            "slept_ms": slept * 1000.0,
                            "elapsed_total_ms": step_elapsed_total * 1000.0,
                            "overran": step_overran,
                        },
                    )

        except KeyboardInterrupt:
            print("\n\n  Rollout interrupted by user")

        finally:
            print(f"\n{'=' * 70}")
            print("ROLLOUT COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Steps: {len(video_frames)}")
            if inference_times:
                print(f"  Inference round-trip: {np.mean(inference_times):.1f}ms +/- {np.std(inference_times):.1f}ms")
                print(f"    Min: {np.min(inference_times):.1f}ms, Max: {np.max(inference_times):.1f}ms")

            if self.config.save_video and video_frames:
                self._save_video(video_frames, instruction)

            if self.config.show_cameras:
                self._stop_display_thread()

    def _save_video(self, frames: list, instruction: str):
        """Save rollout video."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instruction = "".join(c if c.isalnum() else "_" for c in instruction[:30])
        filename = f"rollout_{timestamp}_{safe_instruction}.mp4"

        print(f"\nSaving video to {filename}...")
        if ImageSequenceClip is None:
            print("  ✗ Failed to save video: moviepy is not installed")
            print("    Install moviepy or rerun with --save-video False")
            return
        try:
            clip = ImageSequenceClip(frames, fps=self.config.video_fps)
            clip.write_videofile(filename, codec="libx264", logger=None)
            print(f"  ✓ Video saved: {filename}")
        except Exception as e:
            print(f"  ✗ Failed to save video: {e}")

    def _safe_instruction_name(self, instruction: str) -> str:
        safe_instruction = "".join(c if c.isalnum() else "_" for c in instruction[:40])
        return safe_instruction.strip("_") or "instruction"

    def _ensure_debug_dir(self, instruction: str) -> str:
        if self._debug_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_dirname = f"{timestamp}_{self._safe_instruction_name(instruction)}"
            debug_dir = os.path.abspath(os.path.join(self.config.debug_log_dir, base_dirname))
            suffix = 1
            while os.path.exists(debug_dir):
                debug_dir = os.path.abspath(os.path.join(self.config.debug_log_dir, f"{base_dirname}_{suffix:02d}"))
                suffix += 1
            self._debug_dir = debug_dir
            os.makedirs(self._debug_dir, exist_ok=False)
            print(
                f"  [DEBUG] Saving debug images to: {self._debug_dir} "
                f"(layout={self.config.debug_image_layout}, "
                f"current_model_aug={self.config.debug_apply_model_image_aug})"
            )
        return self._debug_dir

    def _coerce_rgb_image(self, image) -> Optional[np.ndarray]:
        if image is None:
            return None
        image = self._decode_json_numpy(image)
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _prepare_debug_model_image(self, image: np.ndarray, *, apply_model_image_aug: bool) -> np.ndarray:
        """Prepare a debug RGB image in the same square pixel space as the policy model."""
        if apply_model_image_aug:
            return apply_policy_image_aug_transform(image, IMAGE_RESOLUTION)
        return resize_policy_image(image, IMAGE_RESOLUTION)

    def _write_rgb_png(self, path: str, image: np.ndarray) -> None:
        cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    def _save_debug_image_slots(
        self,
        debug_dir: str,
        stem: str,
        group: str,
        slots: list[tuple[str, np.ndarray]],
    ) -> Optional[str]:
        slots = [(name, image) for name, image in slots if image is not None]
        if not slots:
            return None

        layout = self.config.debug_image_layout
        stitched_path = None
        if layout in ("stitched", "both"):
            stitched_path = os.path.join(debug_dir, f"{stem}_{group}_slots.png")
            self._write_rgb_png(stitched_path, np.hstack([image for _, image in slots]))

        if layout in ("separate", "both"):
            for slot_name, image in slots:
                image_path = os.path.join(debug_dir, f"{stem}_{group}_{slot_name}.png")
                self._write_rgb_png(image_path, image)

        return stitched_path

    def _make_future_image_slots(self, future_image_predictions) -> list[tuple[str, np.ndarray]]:
        if not future_image_predictions:
            return []
        preferred_order = [
            ("future_image", "primary_image"),
            ("future_image2", "secondary_image"),
            ("future_wrist_image", "wrist_image"),
            ("future_wrist_image2", "wrist_image2"),
        ]
        seen = set()
        tiles = []
        for key, label in preferred_order:
            if key in future_image_predictions:
                seen.add(key)
                image = self._coerce_rgb_image(future_image_predictions[key])
                if image is not None:
                    tiles.append((label, self._prepare_debug_model_image(image, apply_model_image_aug=False)))
        for key in sorted(set(future_image_predictions.keys()) - seen):
            image = self._coerce_rgb_image(future_image_predictions[key])
            if image is not None:
                tiles.append((key, self._prepare_debug_model_image(image, apply_model_image_aug=False)))
        return tiles

    def _to_jsonable(self, data):
        data = self._decode_json_numpy(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, np.generic):
            return data.item()
        if isinstance(data, dict):
            return {k: self._to_jsonable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_jsonable(v) for v in data]
        return data

    def _save_debug_inputs(self, obs: dict, t_step: int, instruction: str) -> None:
        try:
            debug_dir = self._ensure_debug_dir(instruction)
            stem = f"query_{self._debug_query_index:04d}_step_{t_step:04d}"
            self._debug_stems_by_step[t_step] = stem
            self._debug_query_index += 1

            input_slots = []
            for key, filename_slot in [
                ("primary_image", "primary_image"),
                ("left_wrist_image", "left_wrist_image"),
                ("right_wrist_image", "right_wrist_image"),
            ]:
                image = self._coerce_rgb_image(obs.get(key))
                if image is not None:
                    input_slots.append(
                        (
                            filename_slot,
                            self._prepare_debug_model_image(
                                image,
                                apply_model_image_aug=self.config.debug_apply_model_image_aug,
                            ),
                        )
                    )

            self._save_debug_image_slots(debug_dir, stem, "policy", input_slots)

            json_path = os.path.join(debug_dir, f"{stem}_policy_slots.json")
            payload = {
                "timestep": t_step,
                "task_description": instruction,
                "image_debug_layout": self.config.debug_image_layout,
                "image_debug_preprocessing": {
                    "resize": f"direct_square_{IMAGE_RESOLUTION}",
                    "padding": False,
                    "font_overlay": False,
                    "current_images_apply_model_image_aug": self.config.debug_apply_model_image_aug,
                    "model_image_aug": "90_percent_area_center_crop_then_resize"
                    if self.config.debug_apply_model_image_aug
                    else None,
                },
                "policy_slot_order": [
                    "primary_image",
                    "left_wrist_image",
                    "right_wrist_image",
                ],
                "camera_serials": {
                    "primary_image": self.config.camera.external_camera_serial,
                    "left_wrist_image": self.config.camera.left_camera_serial,
                    "right_wrist_image": self.config.camera.wrist_camera_serial,
                },
                "proprio": self._to_jsonable(obs.get("proprio")),
                "proprio_interpreted": self._to_jsonable(self._summarize_proprio_debug(obs.get("proprio"))),
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"  [WARNING] Failed to save debug inputs: {e}")

    def _save_debug_predictions(self, debug_result: dict, t_step: int, instruction: str) -> None:
        debug_dir = self._ensure_debug_dir(instruction)
        stem = self._debug_stems_by_step.get(t_step, f"step_{t_step:04d}")
        future_proprio = self._decode_json_numpy(debug_result.get("future_proprio"))
        future_proprio_normalized = self._decode_json_numpy(debug_result.get("future_proprio_normalized"))
        current_proprio_normalized = self._decode_json_numpy(debug_result.get("current_proprio_normalized"))
        self._debug_predictions_by_step[t_step] = {
            "chunk_stem": stem,
            "current_proprio_normalized": current_proprio_normalized,
            "future_proprio": future_proprio,
            "future_proprio_normalized": future_proprio_normalized,
            "actions": self._decode_json_numpy(debug_result.get("actions")),
        }
        json_path = os.path.join(debug_dir, f"{stem}_predictions.json")
        payload = {
            "timestep": t_step,
            "value_prediction": debug_result.get("value_prediction"),
            "num_queries_best_of_n": debug_result.get("num_queries_best_of_n"),
            "action_mode": self.config.policy.action_mode,
            "image_debug_layout": self.config.debug_image_layout,
            "future_image_debug_preprocessing": {
                "resize": f"direct_square_{IMAGE_RESOLUTION}",
                "padding": False,
                "font_overlay": False,
                "current_image_model_aug_already_applied_to_policy_debug": self.config.debug_apply_model_image_aug,
            },
            "actions": self._to_jsonable(debug_result.get("actions")),
            "current_proprio_normalized": self._to_jsonable(current_proprio_normalized),
            "current_proprio_normalized_interpreted": self._to_jsonable(
                self._summarize_proprio_debug(current_proprio_normalized)
            ),
            "future_proprio": self._to_jsonable(future_proprio),
            "future_proprio_interpreted": self._to_jsonable(self._summarize_proprio_debug(future_proprio)),
            "future_proprio_normalized": self._to_jsonable(future_proprio_normalized),
            "future_proprio_normalized_interpreted": self._to_jsonable(
                self._summarize_proprio_debug(future_proprio_normalized)
            ),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        future_slots = self._make_future_image_slots(debug_result.get("future_image_predictions"))
        if future_slots:
            self._save_debug_image_slots(debug_dir, stem, "future", future_slots)

    def _save_debug_execution(
        self,
        instruction: str,
        t_step: int,
        chunk_start_step: Optional[int],
        action_index_in_chunk: int,
        policy_action: np.ndarray,
        measured_state_before: np.ndarray,
        target_pose_sent: np.ndarray,
        measured_state_after: np.ndarray,
        step_timing: Optional[dict] = None,
    ) -> None:
        try:
            debug_dir = self._ensure_debug_dir(instruction)
            chunk_stem = None
            chunk_prediction = None
            if chunk_start_step is not None:
                chunk_stem = self._debug_stems_by_step.get(chunk_start_step)
                chunk_prediction = self._debug_predictions_by_step.get(chunk_start_step)

            measured_state_before = np.asarray(measured_state_before, dtype=np.float32)
            measured_state_after = np.asarray(measured_state_after, dtype=np.float32)
            target_pose_sent = np.asarray(target_pose_sent, dtype=np.float32)
            expected_state_after = self._pose_to_eef_state(target_pose_sent, measured_state_before[6:8])
            observed_state_delta = measured_state_after - measured_state_before
            measured_state_before_policy = canonicalize_lerobot_eef_state(measured_state_before)
            measured_state_after_policy = canonicalize_lerobot_eef_state(measured_state_after)
            expected_state_after_policy = canonicalize_lerobot_eef_state(expected_state_after)
            observed_state_delta_policy = measured_state_after_policy - measured_state_before_policy
            target_position_error_after = measured_state_after[:3] - target_pose_sent[:3]

            payload = {
                "timestep": t_step,
                "chunk_start_timestep": chunk_start_step,
                "chunk_stem": chunk_stem,
                "action_index_in_chunk": int(action_index_in_chunk),
                "action_mode": self.config.policy.action_mode,
                "policy_action": np.asarray(policy_action, dtype=np.float32),
                "measured_state_before": measured_state_before,
                "measured_state_before_interpreted": self._summarize_proprio_debug(measured_state_before),
                "measured_state_before_policy_space": measured_state_before_policy,
                "measured_state_before_policy_space_interpreted": self._summarize_proprio_debug(
                    measured_state_before_policy
                ),
                "expected_proprio_after_command": expected_state_after,
                "expected_proprio_after_command_interpreted": self._summarize_proprio_debug(expected_state_after),
                "expected_proprio_after_command_policy_space": expected_state_after_policy,
                "expected_proprio_after_command_policy_space_interpreted": self._summarize_proprio_debug(
                    expected_state_after_policy
                ),
                "target_pose_sent": target_pose_sent,
                "measured_state_after": measured_state_after,
                "measured_state_after_interpreted": self._summarize_proprio_debug(measured_state_after),
                "measured_state_after_policy_space": measured_state_after_policy,
                "measured_state_after_policy_space_interpreted": self._summarize_proprio_debug(
                    measured_state_after_policy
                ),
                "observed_state_delta": observed_state_delta,
                "observed_position_delta": observed_state_delta[:3],
                "observed_euler_delta": observed_state_delta[3:6],
                "observed_state_delta_policy_space": observed_state_delta_policy,
                "observed_position_delta_policy_space": observed_state_delta_policy[:3],
                "observed_euler_delta_policy_space": observed_state_delta_policy[3:6],
                "target_position_delta_from_before": target_pose_sent[:3] - measured_state_before[:3],
                "target_position_error_after": target_position_error_after,
                "pose_error_against_expected_rpy": measured_state_after[:6] - expected_state_after[:6],
                "pose_error_against_expected_policy_space": measured_state_after_policy[:6] - expected_state_after_policy[:6],
                "step_timing": step_timing,
                "execution": self._last_execution_debug,
                "chunk_future_proprio_prediction": None
                if chunk_prediction is None
                else {
                    "chunk_stem": chunk_prediction.get("chunk_stem"),
                    "future_proprio": chunk_prediction.get("future_proprio"),
                    "current_proprio_normalized": chunk_prediction.get("current_proprio_normalized"),
                    "current_proprio_normalized_interpreted": self._summarize_proprio_debug(
                        chunk_prediction.get("current_proprio_normalized")
                    ),
                    "future_proprio_interpreted": self._summarize_proprio_debug(chunk_prediction.get("future_proprio")),
                    "future_proprio_normalized": chunk_prediction.get("future_proprio_normalized"),
                    "future_proprio_normalized_interpreted": self._summarize_proprio_debug(
                        chunk_prediction.get("future_proprio_normalized")
                    ),
                },
            }

            json_path = os.path.join(debug_dir, "execution_trace.jsonl")
            with open(json_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._to_jsonable(payload)) + "\n")
        except Exception as e:
            print(f"  [WARNING] Failed to save execution debug trace: {e}")

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        if self.robot is not None:
            try:
                self.robot.terminate_current_policy()
                self.robot.close()
                print("  ✓ Robot disconnected")
            except Exception as e:
                if self._is_no_controller_running_error(e):
                    try:
                        self.robot.close()
                    except Exception:
                        pass
                    print("  ✓ Robot disconnected (no controller was running)")
                else:
                    print(f"  ✗ Error disconnecting robot: {e}")

        for camera_name in ("external_cam", "wrist_cam", "left_cam"):
            camera = getattr(self, camera_name, None)
            if camera is None:
                continue
            try:
                camera.release()
            except Exception as e:
                print(f"  ✗ Error releasing {camera_name}: {e}")
            finally:
                setattr(self, camera_name, None)

        print("  ✓ Cleanup complete")

    def _decode_json_numpy(self, data):
        if isinstance(data, dict) and "__numpy__" in data:
            decoded_bytes = base64.b64decode(data["__numpy__"])
            return np.frombuffer(decoded_bytes, dtype=data["dtype"]).reshape(data["shape"])
        if isinstance(data, dict):
            return {key: self._decode_json_numpy(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._decode_json_numpy(value) for value in data]
        return data

    def decode_numpy_dict(self, data):
        return self._decode_json_numpy(data)


def main(
    # Camera settings
    external_camera: Optional[str] = None,
    wrist_camera: Optional[str] = None,
    left_camera: Optional[str] = None,
    use_mock_cameras: bool = False,
    # Robot settings
    nuc_ip: str = "127.0.0.1",
    nuc_port: int = 4242,
    control_mode: Literal["joint", "eef"] = "eef",
    use_mock_robot: bool = False,
    # Server settings
    server_ip: str = "127.0.0.1",
    server_port: int = 8777,
    # Rollout settings
    max_timesteps: int = 600,
    show_cameras: bool = True,
    save_video: bool = False,
    debug_predictions: bool = False,
    debug_log_dir: str = "franka_debug_predictions",
    debug_image_layout: Literal["stitched", "separate", "both"] = "stitched",
    debug_apply_model_image_aug: bool = True,
    policy_proprio_format: Literal["eef_8d", "joint_9d"] = "eef_8d",
    policy_gripper_proprio_mode: Literal["measured_width_signed", "zero"] = "measured_width_signed",
    policy_action_mode: Literal["delta_eef_7d", "delta_eef_7d_z_positive_down", "absolute_eef"] = "delta_eef_7d",
    policy_gripper_positive_action: Literal["close", "open"] = "open",
    gripper_release_confirm_steps: Optional[int] = None,
    gripper_chunk_lookahead_close: bool = True,
    gripper_chunk_lookahead_release: bool = True,
    gripper_chunk_lookahead_confirm_steps: int = 3,
    gripper_chunk_lookahead_min_offset: Optional[int] = None,
    open_loop_horizon: int = 8,
    delta_action_base: Literal["measured", "target"] = "measured",
    lock_rpy_actions: bool = False,
    lock_rpy_action_axes: str = "",
    max_target_position_lead: float = 0.05,
    max_target_vertical_lead: float = 0.02,
    min_target_z: Optional[float] = None,
    closed_gripper_min_target_z: Optional[float] = None,
    build_task_min_target_z: Optional[float] = None,
    build_task_closed_gripper_min_target_z: Optional[float] = None,
    default_prompt: Optional[str] = None,
    close_gripper_on_start: bool = False,
    start_gripper_width: Optional[float] = None,
    start_joint_low: Optional[str] = None,
    start_joint_high: Optional[str] = None,
    start_joint_move_time: float = 3.0,
    start_joint_settle_time: float = 0.5,
):
    """
    Run Cosmos-Policy on Franka robot via remote inference server.

    Args:
        external_camera: Serial number of external RealSense camera
        wrist_camera: Serial number of wrist RealSense camera
        left_camera: Serial number of left RealSense camera
        use_mock_cameras: Use mock cameras for testing
        nuc_ip: IP address of Franka NUC
        nuc_port: Port of ZeroRPC server on NUC
        control_mode: Control mode ('joint' or 'eef')
        use_mock_robot: Use mock robot for testing
        server_ip: IP address of inference server (Docker container)
        server_port: Port of inference server
        max_timesteps: Maximum steps per rollout
        show_cameras: Display camera feeds during rollout
        save_video: Save rollout video
        debug_predictions: Ask the server to return future proprio/images/value and save them locally
        debug_log_dir: Directory for per-query debug JSON and image PNGs.
        debug_image_layout: Save debug images as stitched mosaics, separate slot PNGs, or both.
        debug_apply_model_image_aug: Apply the deploy server's deterministic image augmentation to current input debug PNGs.
        policy_proprio_format: Proprio sent to policy server: 'eef_8d' or legacy 'joint_9d'
        policy_gripper_proprio_mode: How to fill the two EEF gripper proprio channels.
        policy_action_mode: Action semantics for 7D policy actions. Use 'delta_eef_7d' for LeRobot.
        policy_gripper_positive_action: Whether positive policy gripper actions mean close or open.
        gripper_release_confirm_steps: Consecutive open-like gripper actions required before releasing.
        gripper_chunk_lookahead_close: Use future chunk gripper actions to schedule close.
        gripper_chunk_lookahead_release: Use future chunk gripper actions to schedule release.
        gripper_chunk_lookahead_confirm_steps: Consecutive future open-like actions needed for lookahead release.
        gripper_chunk_lookahead_min_offset: First action index eligible for lookahead scheduling.
        open_loop_horizon: Number of predicted actions to execute before querying the policy again.
        delta_action_base: Accepted for wrapper compatibility; OpenVLA control always uses previous target pose.
        lock_rpy_actions: Zero out policy droll/dpitch/dyaw before sending target pose commands.
        lock_rpy_action_axes: Comma-separated axes to zero out, e.g. 'roll,yaw'. Full lock overrides this.
        max_target_position_lead: Per-axis max target lead from measured pose (safety clamp kept on top of OpenVLA).
        max_target_vertical_lead: Z-axis max target lead from measured pose (safety clamp kept on top of OpenVLA).
        min_target_z: Optional absolute lower z limit for all target poses, in meters.
        closed_gripper_min_target_z: Optional lower z limit after the gripper has closed, in meters.
        build_task_min_target_z: Build-block-only absolute lower z limit, in meters.
        build_task_closed_gripper_min_target_z: Build-block-only lower z limit after the gripper has closed.
        default_prompt: Prompt to use when Enter is pressed at the task instruction prompt.
        close_gripper_on_start: Close the gripper once before each rollout begins.
        start_gripper_width: Optional physical gripper width to command before each rollout begins.
        start_joint_low: Optional comma-separated 7D lower bound for randomized pre-rollout joint start.
        start_joint_high: Optional comma-separated 7D upper bound for randomized pre-rollout joint start.
        start_joint_move_time: Seconds to move to the sampled pre-rollout joint start.
        start_joint_settle_time: Seconds to wait after reaching the sampled pre-rollout joint start.
    """
    server_url = f"http://{server_ip}:{server_port}"
    start_joint_low_values = None
    start_joint_high_values = None
    start_joint_rng = np.random.default_rng()
    if start_joint_low or start_joint_high:
        if not start_joint_low or not start_joint_high:
            raise ValueError("Provide both start_joint_low and start_joint_high, or neither.")
        if start_joint_move_time <= 0:
            raise ValueError("start_joint_move_time must be > 0")
        if start_joint_settle_time < 0:
            raise ValueError("start_joint_settle_time must be >= 0")
        start_joint_low_values = _parse_csv_vector(start_joint_low, expected_len=7, name="start joint low")
        start_joint_high_values = _parse_csv_vector(start_joint_high, expected_len=7, name="start joint high")
        if np.any(start_joint_high_values < start_joint_low_values):
            raise ValueError("start_joint_high must be >= start_joint_low for every joint.")

    config = cfg.Config(
        camera=cfg.CameraConfig(
            external_camera_serial=external_camera,
            wrist_camera_serial=wrist_camera,
            left_camera_serial=left_camera,
            use_mock_cameras=use_mock_cameras,
        ),
        robot=cfg.RobotConfig(
            nuc_ip=nuc_ip,
            nuc_port=nuc_port,
            control_mode=control_mode,
            use_mock_robot=use_mock_robot,
        ),
        policy=cfg.PolicyConfig(
            open_loop_horizon=open_loop_horizon,
            proprio_format=policy_proprio_format,
            gripper_proprio_mode=policy_gripper_proprio_mode,
            action_mode=policy_action_mode,
            gripper_positive_action=policy_gripper_positive_action,
            gripper_release_confirm_steps=gripper_release_confirm_steps,
            gripper_chunk_lookahead_close=gripper_chunk_lookahead_close,
            gripper_chunk_lookahead_release=gripper_chunk_lookahead_release,
            gripper_chunk_lookahead_confirm_steps=gripper_chunk_lookahead_confirm_steps,
            gripper_chunk_lookahead_min_offset=gripper_chunk_lookahead_min_offset,
            delta_action_base=delta_action_base,
            lock_rpy_actions=lock_rpy_actions,
            lock_rpy_action_axes=lock_rpy_action_axes,
            max_target_position_lead=max_target_position_lead,
            max_target_vertical_lead=max_target_vertical_lead,
            min_target_z=min_target_z,
            closed_gripper_min_target_z=closed_gripper_min_target_z,
            build_task_min_target_z=build_task_min_target_z,
            build_task_closed_gripper_min_target_z=build_task_closed_gripper_min_target_z,
            default_prompt=default_prompt,
        ),
        max_timesteps=max_timesteps,
        show_cameras=show_cameras,
        save_video=save_video,
        debug_predictions=debug_predictions,
        debug_log_dir=debug_log_dir,
        debug_image_layout=debug_image_layout,
        debug_apply_model_image_aug=debug_apply_model_image_aug,
        close_gripper_on_rollout_start=close_gripper_on_start,
        start_gripper_width=start_gripper_width,
    )

    runner = FrankaPolicyRunner(config, server_url)

    try:
        runner.setup()
        if start_joint_low_values is not None and start_joint_high_values is not None:
            print("[Franka eval] Randomized pre-rollout joint start enabled:")
            print(f"  low:  {start_joint_low}")
            print(f"  high: {start_joint_high}")
            print(f"  move: {start_joint_move_time}s (+ settle {start_joint_settle_time}s)")

        while True:
            print("\n" + "=" * 70)
            prompt_suffix = f" [default: {config.policy.default_prompt}]" if config.policy.default_prompt else ""
            raw_instruction = input(f"Enter task instruction{prompt_suffix} (or 'quit' to exit): ").strip()

            if raw_instruction.lower() in ["quit", "exit", "q"]:
                break

            instruction = _resolve_instruction(raw_instruction, config.policy.default_prompt)

            if not instruction:
                print("Please enter a valid instruction.")
                continue

            if start_joint_low_values is not None and start_joint_high_values is not None:
                start_joints = _sample_uniform_range(
                    start_joint_low_values,
                    start_joint_high_values,
                    start_joint_rng,
                )
                _move_to_start_joints(
                    runner,
                    start_joints,
                    start_joint_move_time,
                    start_joint_settle_time,
                    label="randomized start",
                )

            runner.run_rollout(instruction)

            continue_prompt = input("\nRun another task? (y/n): ").strip().lower()
            if continue_prompt != "y":
                break

    except KeyboardInterrupt:
        print("\n\nShutdown requested (Ctrl+C)")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        runner.cleanup()
        print("\nGoodbye!")


if __name__ == "__main__":
    tyro.cli(main)
