"""Replay a LeRobot dataset through MuJoCo physics and stream torques to Rerun.

Loads the dataset directly via huggingface_hub + parquet (bypassing lerobot's
version-tag requirement) so it works with untagged community datasets.

Usage:
    conda activate lerobot
    python unitree_lerobot/eval_robot/replay_mujoco.py \
        --repo_id unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket \
        --episode 0 --frequency 30
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
import tyro
from huggingface_hub import hf_hub_download

from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_MJCF = ASSETS_DIR / "g1" / "g1_body29_inspire.xml"

# Dataset layout for robot_q_desired / robot_q_current (36 elements):
#   [0:3]   base position (xyz)
#   [3:7]   base quaternion (wxyz)
#   [7:13]  left leg (6 joints)
#   [13:19] right leg (6 joints)
#   [19:22] waist (yaw, roll, pitch)
#   [22:29] LEFT ARM (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
#   [29:36] RIGHT ARM (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
#
# hand_cmd (12): 6 left + 6 right fingers in Inspire ordering
#   pinky, ring, middle, index, thumb_bend(pitch), thumb_rotation(yaw)

ARM_Q_INDICES_LEFT = list(range(22, 29))   # 7 joints
ARM_Q_INDICES_RIGHT = list(range(29, 36))  # 7 joints
ARM_Q_INDICES = ARM_Q_INDICES_LEFT + ARM_Q_INDICES_RIGHT  # 14 total

# MuJoCo actuator names in the order we'll build the 26-DOF ctrl vector
ACTUATOR_NAMES_ARM: list[str] = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ACTUATOR_NAMES_HAND: list[str] = [
    "L_pinky_proximal_joint", "L_ring_proximal_joint", "L_middle_proximal_joint",
    "L_index_proximal_joint", "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_middle_proximal_joint",
    "R_index_proximal_joint", "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
]

ALL_ACTUATOR_NAMES = ACTUATOR_NAMES_ARM + ACTUATOR_NAMES_HAND

JOINT_LABELS: list[str] = [
    "L_ShoulderPitch", "L_ShoulderRoll", "L_ShoulderYaw", "L_Elbow",
    "L_WristRoll", "L_WristPitch", "L_WristYaw",
    "R_ShoulderPitch", "R_ShoulderRoll", "R_ShoulderYaw", "R_Elbow",
    "R_WristRoll", "R_WristPitch", "R_WristYaw",
    "L_Pinky", "L_Ring", "L_Middle", "L_Index", "L_ThumbBend", "L_ThumbRot",
    "R_Pinky", "R_Ring", "R_Middle", "R_Index", "R_ThumbBend", "R_ThumbRot",
]


@dataclass
class ReplayConfig:
    repo_id: str = "unitreerobotics/G1_WBT_Inspire_Put_Vegetables_Into_Basket"
    episode: int = 0
    frequency: float = 30.0
    mjcf_path: str = str(DEFAULT_MJCF)
    headless: bool = True
    playback_speed: float = 1.0
    substeps: int = 10
    extra_episodes: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loading (bypasses lerobot version-tag requirement)
# ---------------------------------------------------------------------------

def load_dataset_episode(repo_id: str, episode_idx: int) -> dict[str, np.ndarray]:
    """Download and load a single episode from a LeRobot v3.0 dataset on HF Hub."""
    import pyarrow as pa

    info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
    with open(info_path) as f:
        info = json.load(f)

    chunk_size = info.get("chunks_size", 1000)
    ep_chunk_idx = episode_idx // chunk_size

    ep_meta_path = hf_hub_download(
        repo_id,
        f"meta/episodes/chunk-{ep_chunk_idx:03d}/file-000.parquet",
        repo_type="dataset",
    )
    ep_meta = pq.read_table(ep_meta_path)
    ep_indices = ep_meta.column("episode_index").to_pylist()
    if episode_idx not in ep_indices:
        raise ValueError(f"Episode {episode_idx} not found in metadata")
    row_pos = ep_indices.index(episode_idx)
    data_chunk = ep_meta.column("data/chunk_index")[row_pos].as_py()
    data_file = ep_meta.column("data/file_index")[row_pos].as_py()

    data_parquet = hf_hub_download(
        repo_id,
        f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(data_parquet)

    ep_col = table.column("episode_index").to_pylist()
    mask = [e == episode_idx for e in ep_col]
    indices = [i for i, m in enumerate(mask) if m]
    if not indices:
        raise ValueError(f"Episode {episode_idx} data not found in parquet file")

    episode_table = table.take(indices)

    result: dict[str, np.ndarray] = {}
    for col_name in episode_table.column_names:
        col = episode_table.column(col_name)
        try:
            arr = col.to_numpy(zero_copy_only=False)
            if hasattr(arr[0], "__len__") and not isinstance(arr[0], str):
                arr = np.stack(arr)
            result[col_name] = arr
        except Exception:
            result[col_name] = col.to_pylist()

    n_steps = len(indices)
    print(f"  Episode {episode_idx}: {n_steps} steps loaded")
    print(f"  Columns: {list(result.keys())}")
    return result


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------

def build_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = np.zeros(len(ALL_ACTUATOR_NAMES), dtype=np.int32)
    for i, name in enumerate(ALL_ACTUATOR_NAMES):
        ids[i] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if ids[i] == -1:
            raise ValueError(f"Actuator '{name}' not found in MJCF model")
    return ids


def build_dof_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = np.zeros(len(ALL_ACTUATOR_NAMES), dtype=np.int32)
    for i, name in enumerate(ALL_ACTUATOR_NAMES):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ids[i] = model.jnt_dofadr[jnt_id]
    return ids


def build_qpos_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = np.zeros(len(ALL_ACTUATOR_NAMES), dtype=np.int32)
    for i, name in enumerate(ALL_ACTUATOR_NAMES):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ids[i] = model.jnt_qposadr[jnt_id]
    return ids


def _servo_to_rad_hand(servo_6: np.ndarray) -> np.ndarray:
    """Convert 6 Inspire servo values (0-1 normalized) to MuJoCo joint radians.

    Per-hand ordering: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
    Finger joints (0-3): servo 1.0=open(0 rad), 0.0=curled(1.7 rad)
    Thumb bend (4):      servo 1.0=open(0 rad), 0.0=bent(0.5 rad)
    Thumb rotation (5):  servo maps directly (0.0 -> 0 rad, 1.0 -> 1.3 rad)
    """
    out = np.zeros(6, dtype=np.float64)
    for i in range(4):
        out[i] = (1.0 - np.clip(servo_6[i], 0.0, 1.0)) * 1.7
    out[4] = (1.0 - np.clip(servo_6[4], 0.0, 1.0)) * 0.5
    out[5] = np.clip(servo_6[5], 0.0, 1.0) * 1.3
    return out


def extract_action_26(step: dict[str, np.ndarray], idx: int) -> np.ndarray:
    """Build a 26-element action vector from the dataset's separate fields.

    Arm actions are in radians already. Hand actions are converted from
    Inspire servo normalized (0-1) to MuJoCo joint radians.
    """
    robot_q = step["action.robot_q_desired"]
    hand_cmd = step["action.hand_cmd"]

    row_q = robot_q[idx] if robot_q.ndim > 1 else robot_q
    row_h = hand_cmd[idx] if hand_cmd.ndim > 1 else hand_cmd

    arm_actions = row_q[ARM_Q_INDICES]                          # 14
    left_hand_rad = _servo_to_rad_hand(row_h[0:6])              # 6
    right_hand_rad = _servo_to_rad_hand(row_h[6:12])            # 6
    return np.concatenate([arm_actions, left_hand_rad, right_hand_rad])


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def replay_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    episode_data: dict[str, np.ndarray],
    cfg: ReplayConfig,
    actuator_ids: np.ndarray,
    dof_ids: np.ndarray,
    qpos_ids: np.ndarray,
    logger: RerunLogger,
    viewer: mujoco.viewer.Handle | None,
):
    n_steps = len(episode_data["frame_index"])
    print(f"Replaying: {n_steps} steps at {cfg.frequency} Hz (speed {cfg.playback_speed}x)")

    dt_target = 1.0 / (cfg.frequency * cfg.playback_speed)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    action_26 = extract_action_26(episode_data, 0)
    for i, act_id in enumerate(actuator_ids):
        data.ctrl[act_id] = action_26[i]
    for _ in range(200):
        mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()

    for step in tqdm.tqdm(range(n_steps), desc=f"Episode {cfg.episode}"):
        t_start = time.perf_counter()

        action_26 = extract_action_26(episode_data, step)
        for i, act_id in enumerate(actuator_ids):
            data.ctrl[act_id] = action_26[i]

        for _ in range(cfg.substeps):
            mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()

        qpos_26 = data.qpos[qpos_ids]
        torque_26 = data.qfrc_actuator[dof_ids]

        step_data = {
            "index": torch.tensor(step),
            "observation.state": torch.from_numpy(qpos_26.copy()).float(),
            "action": torch.from_numpy(action_26.copy()).float(),
            "torque": torch.from_numpy(torque_26.copy()).float(),
        }
        logger.log_step(step_data)

        elapsed = time.perf_counter() - t_start
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    cfg = tyro.cli(ReplayConfig)

    print(f"Loading MJCF: {cfg.mjcf_path}")
    model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
    data = mujoco.MjData(model)

    model.opt.timestep = 1.0 / (cfg.frequency * cfg.substeps)

    actuator_ids = build_actuator_ids(model)
    dof_ids = build_dof_ids(model)
    qpos_ids = build_qpos_ids(model)

    print(f"Loading dataset: {cfg.repo_id}")
    episodes = [cfg.episode] + cfg.extra_episodes
    logger = RerunLogger(joint_names=JOINT_LABELS)

    viewer = None
    if not cfg.headless:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except RuntimeError as e:
            print(f"MuJoCo viewer unavailable ({e})")
            print("  On macOS, use `mjpython` instead of `python` for 3D viewer,")
            print("  or run with --headless (default). Continuing with Rerun only.")
            viewer = None

    try:
        for ep in episodes:
            episode_data = load_dataset_episode(cfg.repo_id, ep)
            replay_episode(model, data, episode_data, cfg, actuator_ids, dof_ids, qpos_ids, logger, viewer)
    finally:
        if viewer is not None:
            viewer.close()

    print("Replay complete.")


if __name__ == "__main__":
    main()
