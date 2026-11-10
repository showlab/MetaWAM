"""Configuration for running a Cosmos Policy / MetaWAM inference server client on a Franka robot."""

import dataclasses
from typing import Literal, Optional


@dataclasses.dataclass
class CameraConfig:
    """Configuration for cameras."""

    # Camera serial numbers - find these by running: python camera_utils.py
    external_camera_serial: Optional[str] = None
    wrist_camera_serial: Optional[str] = None
    left_camera_serial: Optional[str] = None

    # Camera settings
    width: int = 640
    height: int = 480
    fps: int = 30

    # Use mock cameras for testing without hardware
    use_mock_cameras: bool = False


@dataclasses.dataclass
class RobotConfig:
    """Configuration for robot connection."""

    # NUC connection (your Polymetis ZeroRPC server)
    nuc_ip: str = "127.0.0.1"  # Placeholder; set to the IP of your Franka NUC
    nuc_port: int = 4242

    # Control mode: 'joint' for joint control, 'eef' for end-effector control
    control_mode: str = "joint_vel"  # We uses end-effector control

    # Control frequency (Hz) - We uses 15 Hz to match training distribution
    control_frequency: int = 15

    # Safety limits
    max_joint_velocity: float = 1.0  # rad/s
    max_gripper_width: float = 0.085  # meters (Franka max)

    # Use mock robot for testing without hardware
    use_mock_robot: bool = False


@dataclasses.dataclass
class PolicyConfig:
    """Configuration for policy inference."""

    # Action execution
    action_horizon: int = 4
    open_loop_horizon: int = 4
    proprio_format: Literal["eef_8d", "joint_9d"] = "eef_8d"
    gripper_proprio_mode: Literal["measured_width_signed", "zero"] = "measured_width_signed"
    action_mode: Literal["delta_eef_7d", "delta_eef_7d_z_positive_down", "absolute_eef"] = "delta_eef_7d"
    gripper_positive_action: Literal["close", "open"] = "open"
    gripper_release_confirm_steps: Optional[int] = None
    gripper_chunk_lookahead_close: bool = True
    gripper_chunk_lookahead_release: bool = True
    gripper_chunk_lookahead_confirm_steps: int = 3
    gripper_chunk_lookahead_min_offset: Optional[int] = None
    delta_action_base: Literal["measured", "target"] = "measured"
    lock_rpy_actions: bool = False
    lock_rpy_action_axes: str = ""
    max_target_position_lead: float = 0.05
    max_target_vertical_lead: float = 0.02
    min_target_z: Optional[float] = None
    closed_gripper_min_target_z: Optional[float] = None
    build_task_min_target_z: Optional[float] = None
    build_task_closed_gripper_min_target_z: Optional[float] = None

    # Default prompt if none provided
    default_prompt: Optional[str] = None


@dataclasses.dataclass
class Config:
    """Main configuration."""

    camera: CameraConfig = dataclasses.field(default_factory=CameraConfig)
    robot: RobotConfig = dataclasses.field(default_factory=RobotConfig)
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)

    # Rollout settings
    max_timesteps: int = 600

    # Visualization
    show_cameras: bool = True
    save_video: bool = True
    video_fps: int = 10

    # Debug model-side future predictions returned by the inference server.
    debug_predictions: bool = False
    debug_log_dir: str = "franka_debug_predictions"
    debug_image_layout: Literal["stitched", "separate", "both"] = "stitched"
    debug_apply_model_image_aug: bool = True

    # Close the gripper once before each rollout starts (matches OpenVLA-Franka warmup).
    close_gripper_on_rollout_start: bool = False
    # Optional physical gripper width to command before each rollout starts.
    start_gripper_width: Optional[float] = None


# Example configurations


def get_default_config() -> Config:
    """Get default configuration - UPDATE WITH YOUR CAMERA SERIALS."""
    return Config(
        camera=CameraConfig(
            external_camera_serial=None,  # Set your camera serial here
            wrist_camera_serial=None,  # Set your camera serial here
            use_mock_cameras=False,  # Set to True for testing without cameras
        ),
        robot=RobotConfig(
            nuc_ip="127.0.0.1",  # Set your NUC IP here
            nuc_port=4242,
        ),
    )


def get_test_config() -> Config:
    """Get test configuration with mock hardware."""
    return Config(
        camera=CameraConfig(
            use_mock_cameras=True,
        ),
        robot=RobotConfig(
            nuc_ip="127.0.0.1",
            nuc_port=4242,
        ),
        max_timesteps=100,
    )
