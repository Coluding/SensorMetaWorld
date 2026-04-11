"""Collect and visualize one Meta-World AVID-style episode.

This mirrors the modality configuration used by `collect_metaworld_avid.py`,
but only runs a single episode for a single environment and writes:

- `overview.mp4`: RGB, depth, left tactile, and right tactile panels
- `proprioception.csv`: the 7D `qpos[:7]` vector for each step
- `metadata.json`: task/noise/episode details for the run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import cv2
import metaworld
import numpy as np

from collect_metaworld_avid import (
    DEFAULT_EXPERT_NOISE_MAX,
    DEFAULT_EXPERT_NOISE_MIN,
    DEPTH_CAMERA_NAME,
    DEPTH_IMAGE_HEIGHT,
    DEPTH_IMAGE_WIDTH,
    MAX_STEPS,
    RGB_CAMERA_NAME,
    RGB_IMAGE_HEIGHT,
    RGB_IMAGE_WIDTH,
    TACTILE_RESOLUTION,
    collect_episode,
    get_policy_for_task,
)
from metaworld_policies import NoisyExpertPolicy
from metaworld.sensors.tactile_digit_sensor import TactileDigitSensor
from metaworld.sensors.visual import DepthCameraSensor

DEFAULT_ENV_NAME = "basketball-v3"
DEFAULT_OUTPUT_DIR = Path("data/metaworld/episode_visualization")
DEFAULT_FPS = 20
FOURCC_ALIAS_MAP = {
    "libx264": "mp4v",
    "h264": "mp4v",
    "mpeg4": "mp4v",
    "x264": "mp4v",
}
PANEL_TITLE_HEIGHT = 28
PANEL_GAP = 12
FRAME_MARGIN = 12
TACTILE_PRESSURE_UPPER = float(TactileDigitSensor._PRESSURE_CLIP)
TACTILE_COLORMAP = getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_VIRIDIS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one Meta-World episode with the same RGB/depth/tactile/"
            "proprioception settings as collect_metaworld_avid.py and write "
            "visualization outputs."
        )
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=DEFAULT_ENV_NAME,
        help="MT50 environment name to visualize, e.g. basketball-v3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where overview.mp4, proprioception.csv, and metadata.json are saved.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Optional fixed task index within the selected environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for task selection, env reset, and expert noise sampling.",
    )
    parser.add_argument(
        "--expert-noise-min",
        type=float,
        default=DEFAULT_EXPERT_NOISE_MIN,
        help="Minimum Gaussian noise scale applied to the scripted expert.",
    )
    parser.add_argument(
        "--expert-noise-max",
        type=float,
        default=DEFAULT_EXPERT_NOISE_MAX,
        help="Maximum Gaussian noise scale applied to the scripted expert.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum number of environment steps to record.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="FPS for the output overview video.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="OpenCV fourcc or alias (e.g. mp4v, avc1, mpeg4, libx264).",
    )
    return parser.parse_args()


def _resolve_fourcc(codec: str) -> str:
    if len(codec) == 4:
        return codec
    return FOURCC_ALIAS_MAP.get(codec.lower(), "mp4v")


def _open_video_writer(
    output_path: Path,
    fps: int,
    codec: str,
    frame_size_wh: tuple[int, int],
) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        output_path.as_posix(),
        cv2.VideoWriter_fourcc(*_resolve_fourcc(codec)),
        float(fps),
        frame_size_wh,
    )
    if writer.isOpened():
        return writer

    fallback = cv2.VideoWriter(
        output_path.as_posix(),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        frame_size_wh,
    )
    if fallback.isOpened():
        return fallback

    raise RuntimeError(f"Could not open video writer for '{output_path}'.")


def _normalize_to_uint8(
    array: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), lower, upper)
    scale = upper - lower
    if scale <= 1e-8:
        return np.zeros(clipped.shape, dtype=np.uint8)
    normalized = (clipped - lower) / scale
    return np.round(normalized * 255.0).astype(np.uint8)


def _compute_depth_bounds(depth_frames: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(depth_frames, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    lower = float(np.percentile(finite, 1.0))
    upper = float(np.percentile(finite, 99.0))
    if upper <= lower:
        upper = lower + 1e-6
    return lower, upper


def _colorize_depth(depth_frame: np.ndarray, lower: float, upper: float) -> np.ndarray:
    depth_uint8 = _normalize_to_uint8(depth_frame, lower, upper)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)


def _colorize_tactile(tactile_frame: np.ndarray) -> np.ndarray:
    tactile_uint8 = _normalize_to_uint8(tactile_frame, 0.0, TACTILE_PRESSURE_UPPER)
    return cv2.applyColorMap(tactile_uint8, TACTILE_COLORMAP)


def _decorate_panel(panel_bgr: np.ndarray, title: str) -> np.ndarray:
    height, width = panel_bgr.shape[:2]
    canvas = np.full(
        (height + PANEL_TITLE_HEIGHT, width, 3),
        245,
        dtype=np.uint8,
    )
    canvas[PANEL_TITLE_HEIGHT:, :, :] = panel_bgr
    cv2.putText(
        canvas,
        title,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _make_overview_frame(
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    tactile_frame: np.ndarray,
    depth_bounds: tuple[float, float],
    frame_index: int,
    total_frames: int,
    env_name: str,
) -> np.ndarray:
    rgb_panel = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    depth_panel = _colorize_depth(depth_frame, *depth_bounds)

    left_tactile = cv2.resize(
        _colorize_tactile(tactile_frame[0]),
        (RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    right_tactile = cv2.resize(
        _colorize_tactile(tactile_frame[1]),
        (RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )

    panels = [
        _decorate_panel(rgb_panel, "RGB"),
        _decorate_panel(depth_panel, "Depth"),
        _decorate_panel(left_tactile, "Left tactile"),
        _decorate_panel(right_tactile, "Right tactile"),
    ]

    row_height = panels[0].shape[0]
    row_width = panels[0].shape[1]
    frame_height = 2 * row_height + PANEL_GAP + 2 * FRAME_MARGIN
    frame_width = 2 * row_width + PANEL_GAP + 2 * FRAME_MARGIN
    frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

    positions = [
        (FRAME_MARGIN, FRAME_MARGIN),
        (FRAME_MARGIN, FRAME_MARGIN + row_width + PANEL_GAP),
        (FRAME_MARGIN + row_height + PANEL_GAP, FRAME_MARGIN),
        (
            FRAME_MARGIN + row_height + PANEL_GAP,
            FRAME_MARGIN + row_width + PANEL_GAP,
        ),
    ]

    for panel, (top, left) in zip(panels, positions):
        frame[top : top + row_height, left : left + row_width] = panel

    overlay = f"{env_name} | frame {frame_index + 1}/{total_frames}"
    cv2.putText(
        frame,
        overlay,
        (FRAME_MARGIN, frame.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return frame


def save_overview_video(
    episode: dict[str, np.ndarray],
    output_path: Path,
    env_name: str,
    fps: int,
    codec: str,
) -> None:
    rgb_frames = np.asarray(episode["image"], dtype=np.uint8)
    depth_frames = np.asarray(episode["depth_image"], dtype=np.float32)
    tactile_frames = np.asarray(episode["tactile_image"], dtype=np.float32)

    depth_bounds = _compute_depth_bounds(depth_frames)
    example_frame = _make_overview_frame(
        rgb_frame=rgb_frames[0],
        depth_frame=depth_frames[0],
        tactile_frame=tactile_frames[0],
        depth_bounds=depth_bounds,
        frame_index=0,
        total_frames=len(rgb_frames),
        env_name=env_name,
    )
    writer = _open_video_writer(
        output_path=output_path,
        fps=fps,
        codec=codec,
        frame_size_wh=(example_frame.shape[1], example_frame.shape[0]),
    )

    try:
        writer.write(example_frame)
        for frame_index in range(1, len(rgb_frames)):
            writer.write(
                _make_overview_frame(
                    rgb_frame=rgb_frames[frame_index],
                    depth_frame=depth_frames[frame_index],
                    tactile_frame=tactile_frames[frame_index],
                    depth_bounds=depth_bounds,
                    frame_index=frame_index,
                    total_frames=len(rgb_frames),
                    env_name=env_name,
                )
            )
    finally:
        writer.release()


def save_proprioception_csv(episode: dict[str, np.ndarray], output_path: Path) -> None:
    proprioception = np.asarray(episode["proprioception"], dtype=np.float32)
    header = ["step"] + [f"qpos_{idx}" for idx in range(proprioception.shape[1])]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for step_index, row in enumerate(proprioception):
            writer.writerow([step_index, *row.tolist()])


def save_rgb_stats_csv(episode: dict[str, np.ndarray], output_path: Path) -> None:
    rgb_frames = np.asarray(episode["image"], dtype=np.uint8)
    header = [
        "frame",
        "max",
        "mean",
        "std",
        "r_max",
        "r_mean",
        "r_std",
        "g_max",
        "g_mean",
        "g_std",
        "b_max",
        "b_mean",
        "b_std",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for frame_index, frame in enumerate(rgb_frames):
            frame_f32 = frame.astype(np.float32)
            channel_stats = []
            for channel_index in range(frame_f32.shape[2]):
                channel = frame_f32[:, :, channel_index]
                channel_stats.extend(
                    [
                        float(channel.max()),
                        float(channel.mean()),
                        float(channel.std()),
                    ]
                )
            writer.writerow(
                [
                    frame_index,
                    float(frame_f32.max()),
                    float(frame_f32.mean()),
                    float(frame_f32.std()),
                    *channel_stats,
                ]
            )


def save_action_diagnostics_csv(
    episode: dict[str, np.ndarray], output_path: Path
) -> None:
    action = np.asarray(episode["action"], dtype=np.float32)
    expert_raw = np.asarray(episode.get("expert_action_raw", []), dtype=np.float32)
    expert_clipped = np.asarray(
        episode.get("expert_action_clipped", []), dtype=np.float32
    )
    action_noise = np.asarray(episode.get("action_noise", []), dtype=np.float32)

    if (
        len(expert_raw) != len(action)
        or len(expert_clipped) != len(action)
        or len(action_noise) != len(action)
    ):
        return

    header = [
        "step",
        "expert_raw_x",
        "expert_raw_y",
        "expert_raw_z",
        "expert_raw_gripper",
        "expert_clipped_x",
        "expert_clipped_y",
        "expert_clipped_z",
        "expert_clipped_gripper",
        "noise_x",
        "noise_y",
        "noise_z",
        "noise_gripper",
        "action_x",
        "action_y",
        "action_z",
        "action_gripper",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for step_index in range(len(action)):
            writer.writerow(
                [
                    step_index,
                    *expert_raw[step_index].tolist(),
                    *expert_clipped[step_index].tolist(),
                    *action_noise[step_index].tolist(),
                    *action[step_index].tolist(),
                ]
            )


def select_task(
    available_tasks: list,
    rng: random.Random,
    task_index: int | None,
) -> tuple[object, int]:
    if task_index is None:
        selected_index = rng.randrange(len(available_tasks))
    else:
        if task_index < 0 or task_index >= len(available_tasks):
            raise ValueError(
                f"task_index={task_index} is out of range for {len(available_tasks)} available tasks."
            )
        selected_index = task_index
    return available_tasks[selected_index], selected_index


def collect_single_episode(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, int | float | str]]:
    if args.expert_noise_min < 0.0 or args.expert_noise_max < 0.0:
        raise ValueError("Expert noise bounds must be non-negative.")
    if args.expert_noise_min > args.expert_noise_max:
        raise ValueError("expert_noise_min must be less than or equal to expert_noise_max.")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if args.fps <= 0:
        raise ValueError("fps must be positive.")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    mt50 = metaworld.MT50()
    if args.env_name not in mt50.train_classes:
        available_envs = ", ".join(sorted(mt50.train_classes.keys()))
        raise ValueError(
            f"Unknown environment '{args.env_name}'. Available MT50 envs: {available_envs}"
        )

    policy = get_policy_for_task(args.env_name)
    if policy is None:
        raise ValueError(f"No scripted policy available for '{args.env_name}'.")

    env_cls = mt50.train_classes[args.env_name]
    available_tasks = [task for task in mt50.train_tasks if task.env_name == args.env_name]
    if not available_tasks:
        raise ValueError(f"No tasks available for '{args.env_name}'.")

    selected_task, selected_task_index = select_task(
        available_tasks=available_tasks,
        rng=rng,
        task_index=args.task_index,
    )
    noise_scale = rng.uniform(args.expert_noise_min, args.expert_noise_max)
    noisy_policy = NoisyExpertPolicy(policy, noise_scale=noise_scale)

    env = env_cls(render_mode="rgb_array", camera_name=RGB_CAMERA_NAME)
    depth_sensor = DepthCameraSensor(
        camera_name=DEPTH_CAMERA_NAME,
        height=DEPTH_IMAGE_HEIGHT,
        width=DEPTH_IMAGE_WIDTH,
        normalize=False,
    )
    tactile_sensor = TactileDigitSensor(
        resolution=TACTILE_RESOLUTION,
        noise_std=0.0,
        base_texture=False,
        normalize=False,
        photometric_render=False,
    )

    try:
        env.set_task(selected_task)
        episode = collect_episode(
            env=env,
            policy=noisy_policy,
            task_name=args.env_name,
            depth_sensor=depth_sensor,
            tactile_sensor=tactile_sensor,
            max_steps=args.max_steps,
            reset_seed=args.seed,
        )
    finally:
        env.close()

    metadata: dict[str, int | float | str] = {
        "env_name": args.env_name,
        "seed": args.seed,
        "task_index": selected_task_index,
        "num_available_tasks": len(available_tasks),
        "noise_scale": float(noise_scale),
        "num_steps": int(len(episode["image"])),
        "rgb_camera_name": RGB_CAMERA_NAME,
        "depth_camera_name": DEPTH_CAMERA_NAME,
    }
    return episode, metadata


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve() / args.env_name
    output_dir.mkdir(parents=True, exist_ok=True)

    episode, metadata = collect_single_episode(args)

    video_path = output_dir / "overview.mp4"
    proprio_path = output_dir / "proprioception.csv"
    rgb_stats_path = output_dir / "rgb_stats.csv"
    action_diagnostics_path = output_dir / "action_diagnostics.csv"
    metadata_path = output_dir / "metadata.json"

    save_overview_video(
        episode=episode,
        output_path=video_path,
        env_name=args.env_name,
        fps=args.fps,
        codec=args.codec,
    )
    save_proprioception_csv(episode, proprio_path)
    save_rgb_stats_csv(episode, rgb_stats_path)
    save_action_diagnostics_csv(episode, action_diagnostics_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved overview video to {video_path}")
    print(f"Saved proprioception CSV to {proprio_path}")
    print(f"Saved RGB stats CSV to {rgb_stats_path}")
    if action_diagnostics_path.exists():
        print(f"Saved action diagnostics CSV to {action_diagnostics_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
