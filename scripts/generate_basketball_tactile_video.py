"""Generate a qualitative basketball run video with image-based tactile maps.

This script runs `basketball-v3` for a fixed number of steps and saves a video
with three synchronized panels:
1. Environment RGB rendering
2. Left tactile image (inside contacts in red channel, outside in green channel)
3. Right tactile image (inside contacts in red channel, outside in green channel)

Usage:
    python scripts/generate_basketball_tactile_video.py
    python scripts/generate_basketball_tactile_video.py --steps 200 --output tactile_demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import matplotlib
import numpy as np

from metaworld.policies.sawyer_basketball_v3_policy import SawyerBasketballV3Policy
from metaworld.sensors.tactile import ImageTactileSensor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a basketball-v3 episode with tactile image side panels."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of environment steps to record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("basketball_tactile_demo.mp4"),
        help="Output video path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="OpenCV fourcc or alias (e.g. mp4v, avc1, mpeg4, libx264).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reset seed for reproducibility.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="corner2",
        help="MuJoCo camera name used for env RGB rendering.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=32,
        help="Per-finger tactile image height.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=32,
        help="Per-finger tactile image width.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=1.0,
        help="Gaussian splat sigma in pixels for tactile rasterization.",
    )
    parser.add_argument(
        "--left-geoms",
        nargs="+",
        default=["leftpad_geom"],
        help="Left finger geom names to monitor.",
    )
    parser.add_argument(
        "--right-geoms",
        nargs="+",
        default=["rightpad_geom"],
        help="Right finger geom names to monitor.",
    )
    parser.add_argument(
        "--left-inside-sign",
        type=float,
        default=-1.0,
        help="Inside classification sign for left finger local normal axis.",
    )
    parser.add_argument(
        "--right-inside-sign",
        type=float,
        default=1.0,
        help="Inside classification sign for right finger local normal axis.",
    )
    return parser.parse_args()


def _resolve_fourcc(codec: str) -> str:
    """Resolve codec alias to a 4-char OpenCV fourcc code."""
    if len(codec) == 4:
        return codec

    alias = codec.lower()
    alias_map = {
        "libx264": "mp4v",
        "h264": "mp4v",
        "x264": "mp4v",
        "mpeg4": "mp4v",
    }
    return alias_map.get(alias, "mp4v")


def _open_video_writer(
    output_path: Path, fps: int, codec: str, frame_size_wh: tuple[int, int]
) -> cv2.VideoWriter:
    """Open an OpenCV video writer and verify it is usable."""
    fourcc_code = _resolve_fourcc(codec)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(
        output_path.as_posix(),
        fourcc,
        float(fps),
        frame_size_wh,
    )
    if writer.isOpened():
        return writer

    # Hard fallback for mp4 on many systems.
    fallback_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path.as_posix(),
        fallback_fourcc,
        float(fps),
        frame_size_wh,
    )
    if writer.isOpened():
        return writer

    raise RuntimeError(
        f"Could not open video writer for '{output_path}' with codec='{codec}'."
    )


def _create_panel_figure(
    rgb_frame: np.ndarray,
    tactile_size: tuple[int, int],
) -> tuple[plt.Figure, list[plt.Axes], list[matplotlib.image.AxesImage]]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
    for ax in axes:
        ax.axis("off")

    height, width = tactile_size
    rgb_artist = axes[0].imshow(rgb_frame)
    zeros_rgb = np.zeros((height, width, 3), dtype=np.float32)
    left_artist = axes[1].imshow(
        zeros_rgb,
    )
    right_artist = axes[2].imshow(
        zeros_rgb,
    )

    axes[0].set_title("RGB Render")
    axes[1].set_title("Left Tactile (R=inside, G=outside)")
    axes[2].set_title("Right Tactile (R=inside, G=outside)")
    fig.tight_layout()
    return fig, list(axes), [rgb_artist, left_artist, right_artist]


def _compose_signed_tactile_rgb(
    inside_map: np.ndarray, outside_map: np.ndarray
) -> np.ndarray:
    """Compose NN-friendly RGB encoding with decoupled channels.

    Channel semantics:
    - R: inside contact intensity
    - G: outside contact intensity
    - B: unused (0)
    """
    inside = np.asarray(inside_map, dtype=np.float32)
    outside = np.asarray(outside_map, dtype=np.float32)
    inside = np.clip(inside, 0.0, 1.0)
    outside = np.clip(outside, 0.0, 1.0)

    rgb = np.zeros((*inside.shape, 3), dtype=np.float32)
    rgb[..., 0] = inside
    rgb[..., 1] = outside
    return np.clip(rgb, 0.0, 1.0)


def _figure_to_rgb_frame(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure canvas into an `H x W x 3` uint8 frame."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = rgb.reshape((height, width, 3))
    return np.ascontiguousarray(frame)


def _flip_rgb_vertically(rgb_frame: np.ndarray) -> np.ndarray:
    """Flip an RGB frame top-to-bottom."""
    return np.ascontiguousarray(np.flipud(rgb_frame))


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "Meta-World/MT1",
        env_name="basketball-v3",
        render_mode="rgb_array",
        camera_name=args.camera_name,
        disable_env_checker=True,
    )
    policy = SawyerBasketballV3Policy()
    sensor = ImageTactileSensor(
        image_size=(args.image_height, args.image_width),
        left_geom_names=tuple(args.left_geoms),
        right_geom_names=tuple(args.right_geoms),
        # Match common MetaWorld gripper pad dimensions (x,z half-sizes in local frame).
        pad_half_extent=(0.045, 0.015),
        auto_pad_half_extent=True,
        gaussian_sigma=args.gaussian_sigma,
        inside_normal_sign_by_side=(args.left_inside_sign, args.right_inside_sign),
    )

    obs, _ = env.reset(seed=args.seed)
    sensor.validate(env)
    sensor.reset(env)
    metadata = sensor.get_metadata()
    print(
        "Tactile extents (effective): "
        f"left={metadata['left_pad_half_extent_effective']} "
        f"right={metadata['right_pad_half_extent_effective']}"
    )
    print(
        "Tactile surface dimensions (meters): "
        f"left={metadata['left_surface_dimensions_m']} "
        f"right={metadata['right_surface_dimensions_m']}"
    )
    print(f"Tactile gaussian_sigma: {args.gaussian_sigma}")

    initial_rgb = np.asarray(env.render(), dtype=np.uint8)
    initial_rgb = _flip_rgb_vertically(initial_rgb)
    fig, axes, artists = _create_panel_figure(
        rgb_frame=initial_rgb,
        tactile_size=sensor.image_size,
    )
    rgb_artist, left_artist, right_artist = artists

    first_frame = _figure_to_rgb_frame(fig)
    first_h, first_w = first_frame.shape[:2]
    writer = _open_video_writer(
        output_path,
        fps=args.fps,
        codec=args.codec,
        frame_size_wh=(first_w, first_h),
    )

    max_left_seen = 0.0
    max_right_seen = 0.0
    activated_steps = 0

    try:
        for step_idx in range(args.steps):
            action = np.clip(policy.get_action(obs), -1.0, 1.0)
            obs, _, terminated, truncated, _ = env.step(action)
            sensor.update(env)

            tactile = sensor.read().astype(np.float32).reshape(
                2, sensor.image_size[0], sensor.image_size[1]
            )
            left_tactile = tactile[0]
            right_tactile = tactile[1]
            left_inside = sensor.left_inside_image
            left_outside = sensor.left_outside_image
            right_inside = sensor.right_inside_image
            right_outside = sensor.right_outside_image
            left_color = _compose_signed_tactile_rgb(left_inside, left_outside)
            right_color = _compose_signed_tactile_rgb(right_inside, right_outside)
            left_peak = float(left_tactile.max())
            right_peak = float(right_tactile.max())
            max_left_seen = max(max_left_seen, left_peak)
            max_right_seen = max(max_right_seen, right_peak)
            if left_peak > 1e-6 or right_peak > 1e-6:
                activated_steps += 1
            rgb_frame = np.asarray(env.render(), dtype=np.uint8)
            rgb_frame = _flip_rgb_vertically(rgb_frame)

            rgb_artist.set_data(rgb_frame)
            left_artist.set_data(left_color)
            right_artist.set_data(right_color)
            axes[0].set_title(
                f"RGB Render ({args.camera_name}) step={step_idx + 1}"
            )
            axes[1].set_title(
                f"Left in={left_inside.max():.3f} out={left_outside.max():.3f}"
            )
            axes[2].set_title(
                f"Right in={right_inside.max():.3f} out={right_outside.max():.3f}"
            )

            panel_frame = _figure_to_rgb_frame(fig)
            writer.write(cv2.cvtColor(panel_frame, cv2.COLOR_RGB2BGR))

            if terminated or truncated:
                obs, _ = env.reset()
                sensor.reset(env)
    finally:
        writer.release()

    plt.close(fig)
    env.close()
    print(f"Saved tactile video to: {output_path}")
    print(
        "Tactile summary: "
        f"max_left={max_left_seen:.4f}, max_right={max_right_seen:.4f}, "
        f"activated_steps={activated_steps}/{args.steps}"
    )


if __name__ == "__main__":
    main()
