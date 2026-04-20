"""Convert the WBT Inspire dataset to a training-compatible format.

The source dataset has split features (observation.state.{ee_state,hand_state,robot_q_current}
and action.{ee_action,hand_cmd,robot_q_desired}). LeRobot's ACT/Diffusion policies expect
single concatenated observation.state and action tensors.

This script:
  1. Downloads the source dataset
  2. Concatenates sub-features into observation.state (60-dim) and action (60-dim)
  3. Keeps video features as-is (symlinked/copied)
  4. Recomputes stats for the merged features
  5. Uploads to a new HF dataset repo with the v3.0 tag

Usage:
    python scripts/convert_dataset_for_training.py \
        --src Kon-prosus/G1_WBT_Inspire_Put_Vegetables_Into_Basket \
        --dst Kon-prosus/G1_Inspire_Vegetables_Train \
        --push
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

STATE_KEYS = ["observation.state.ee_state", "observation.state.hand_state", "observation.state.robot_q_current"]
ACTION_KEYS = ["action.ee_action", "action.hand_cmd", "action.robot_q_desired"]
MERGED_STATE = "observation.state"
MERGED_ACTION = "action"
PASS_THROUGH = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def concat_columns(table: pa.Table, src_keys: list[str], dst_key: str) -> pa.Table:
    """Concatenate multiple list-of-float columns into one."""
    arrays = []
    for key in src_keys:
        col = table.column(key)
        arrays.append(col)

    n = len(table)
    merged = []
    for i in range(n):
        row = []
        for col in arrays:
            val = col[i].as_py()
            row.extend(val)
        merged.append(row)

    new_col = pa.array(merged, type=pa.list_(pa.float32()))
    table = table.append_column(dst_key, new_col)
    for key in src_keys:
        idx = table.column_names.index(key)
        table = table.remove_column(idx)
    return table


def compute_stats_for_column(values: np.ndarray) -> dict:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])] * values.shape[1],
        "q01": np.percentile(values, 1, axis=0).tolist(),
        "q10": np.percentile(values, 10, axis=0).tolist(),
        "q50": np.percentile(values, 50, axis=0).tolist(),
        "q90": np.percentile(values, 90, axis=0).tolist(),
        "q99": np.percentile(values, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="Kon-prosus/G1_WBT_Inspire_Put_Vegetables_Into_Basket")
    parser.add_argument("--dst", default="Kon-prosus/G1_Inspire_Vegetables_Train")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--local-dir", default="./converted_dataset")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading source dataset: {args.src}")
    src_dir = Path(snapshot_download(args.src, repo_type="dataset"))
    print(f"Source at: {src_dir}")

    with open(src_dir / "meta" / "info.json") as f:
        info = json.load(f)

    state_dim = sum(info["features"][k]["shape"][0] for k in STATE_KEYS)
    action_dim = sum(info["features"][k]["shape"][0] for k in ACTION_KEYS)
    print(f"Merged state dim: {state_dim}, action dim: {action_dim}")

    state_names = []
    for k in STATE_KEYS:
        state_names.extend(info["features"][k]["names"])
    action_names = []
    for k in ACTION_KEYS:
        action_names.extend(info["features"][k]["names"])

    # Process data parquet files
    data_dir = local_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    all_states = []
    all_actions = []

    for parquet_file in sorted((src_dir / "data").rglob("*.parquet")):
        rel = parquet_file.relative_to(src_dir / "data")
        out_path = data_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Converting {rel}...")
        table = pq.read_table(parquet_file)

        table = concat_columns(table, STATE_KEYS, MERGED_STATE)
        table = concat_columns(table, ACTION_KEYS, MERGED_ACTION)

        pq.write_table(table, out_path)

        state_col = table.column(MERGED_STATE)
        action_col = table.column(MERGED_ACTION)
        all_states.append(np.array([r.as_py() for r in state_col], dtype=np.float32))
        all_actions.append(np.array([r.as_py() for r in action_col], dtype=np.float32))

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    # Build new info.json
    new_features = {}
    new_features[MERGED_STATE] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": state_names,
    }
    new_features[MERGED_ACTION] = {
        "dtype": "float32",
        "shape": [action_dim],
        "names": action_names,
    }
    for k, v in info["features"].items():
        if k in STATE_KEYS or k in ACTION_KEYS:
            continue
        new_features[k] = v

    new_info = {**info, "features": new_features}

    meta_dir = local_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    # Copy episode metadata and tasks
    for meta_file in (src_dir / "meta").rglob("*"):
        if meta_file.is_file() and meta_file.name != "info.json" and meta_file.name != "stats.json":
            rel = meta_file.relative_to(src_dir / "meta")
            dest = meta_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_file, dest)

    # Compute new stats
    with open(src_dir / "meta" / "stats.json") as f:
        old_stats = json.load(f)

    new_stats = {}
    new_stats[MERGED_STATE] = compute_stats_for_column(all_states)
    new_stats[MERGED_ACTION] = compute_stats_for_column(all_actions)
    for k, v in old_stats.items():
        if k not in STATE_KEYS and k not in ACTION_KEYS:
            new_stats[k] = v

    with open(meta_dir / "stats.json", "w") as f:
        json.dump(new_stats, f, indent=2)

    # Symlink videos
    src_videos = src_dir / "videos"
    dst_videos = local_dir / "videos"
    if src_videos.exists():
        if dst_videos.exists():
            shutil.rmtree(dst_videos)
        shutil.copytree(src_videos, dst_videos, symlinks=True, dirs_exist_ok=True)
        print(f"Copied video directory")

    print(f"\nConverted dataset at: {local_dir}")
    print(f"  State: {MERGED_STATE} ({state_dim})")
    print(f"  Action: {MERGED_ACTION} ({action_dim})")

    if args.push:
        api = HfApi()
        print(f"\nCreating repo {args.dst}...")
        api.create_repo(args.dst, repo_type="dataset", exist_ok=True)

        print("Uploading metadata + data...")
        api.upload_folder(
            folder_path=str(meta_dir),
            path_in_repo="meta",
            repo_id=args.dst,
            repo_type="dataset",
        )
        api.upload_folder(
            folder_path=str(data_dir),
            path_in_repo="data",
            repo_id=args.dst,
            repo_type="dataset",
        )

        if dst_videos.exists():
            print("Uploading videos (this may take a while)...")
            api.upload_folder(
                folder_path=str(dst_videos),
                path_in_repo="videos",
                repo_id=args.dst,
                repo_type="dataset",
            )

        print("Creating v3.0 tag...")
        try:
            api.delete_tag(args.dst, tag="v3.0", repo_type="dataset")
        except Exception:
            pass
        api.create_tag(args.dst, tag="v3.0", repo_type="dataset")

        print(f"\nDone! Dataset at: https://huggingface.co/datasets/{args.dst}")


if __name__ == "__main__":
    main()
