"""Interactively trim a collected MetaWorld HDF5 dataset.

Loads a dataset collected by ``collect_metaworld_avid.py`` (layout
``<env>/<camera>/episode_N`` + ``<env>/sensors/episode_N``), prints a summary of
its environments and cameras with their on-disk sizes, then asks which
environments and which cameras to drop. The kept data is written to a NEW file
(HDF5 cannot reclaim space in place), preserving the original's compression.

Usage:
    python scripts/trim_metaworld_dataset.py data/metaworld/big.hdf5
    python scripts/trim_metaworld_dataset.py big.hdf5 -o small.hdf5 --yes
"""

import argparse
import math
import random
from pathlib import Path

import h5py

SENSORS_GROUP = "sensors"


def _human(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _group_storage_size(group: h5py.Group) -> int:
    """Sum the actual (compressed) on-disk storage of every dataset in a group."""
    total = 0

    def _visit(_name, obj):
        nonlocal total
        if isinstance(obj, h5py.Dataset):
            total += obj.id.get_storage_size()

    group.visititems(_visit)
    return total


def scan(f: h5py.File):
    """Return (env_info, camera_sizes) summarising the file.

    env_info: dict env_name -> {"episodes": int, "size": int,
                                "cameras": {cam: size}}
    camera_sizes: dict camera_name -> total size across all envs
    """
    env_info = {}
    camera_sizes = {}

    for env_name in f:
        env_group = f[env_name]
        if not isinstance(env_group, h5py.Group):
            continue

        cameras = {}
        episodes = 0
        env_size = 0
        for sub_name in env_group:
            sub = env_group[sub_name]
            if not isinstance(sub, h5py.Group):
                continue
            size = _group_storage_size(sub)
            env_size += size
            if sub_name == SENSORS_GROUP:
                episodes = len(sub.keys())
            else:
                cameras[sub_name] = size
                camera_sizes[sub_name] = camera_sizes.get(sub_name, 0) + size

        if episodes == 0 and cameras:
            # Fall back to a camera group's episode count if no sensors group.
            any_cam = next(iter(cameras))
            episodes = len(env_group[any_cam].keys())

        env_info[env_name] = {
            "episodes": episodes,
            "size": env_size,
            "cameras": cameras,
        }

    return env_info, camera_sizes


def print_summary(env_info, camera_sizes):
    total = sum(info["size"] for info in env_info.values())
    print("\n=== Environments ===")
    print(f"{'#':>3}  {'environment':<32} {'episodes':>9} {'size':>12}")
    print("-" * 60)
    for idx, (env_name, info) in enumerate(sorted(env_info.items())):
        print(
            f"{idx:>3}  {env_name:<32} {info['episodes']:>9} "
            f"{_human(info['size']):>12}"
        )

    print("\n=== Cameras (totalled across all envs) ===")
    print(f"{'#':>3}  {'camera':<20} {'size':>12} {'% of file':>10}")
    print("-" * 50)
    for idx, (cam, size) in enumerate(sorted(camera_sizes.items())):
        pct = (100.0 * size / total) if total else 0.0
        print(f"{idx:>3}  {cam:<20} {_human(size):>12} {pct:>9.1f}%")

    print(f"\nTotal dataset size (datasets only): {_human(total)}")


def preview_cameras(f: h5py.File, env_info) -> None:
    """Show a grid (one frame per camera) from a random environment in a window."""
    candidates = [e for e, info in env_info.items() if info["cameras"]]
    if not candidates:
        print("No cameras available to preview.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; cannot show preview "
              "(pip install matplotlib).")
        return

    env_name = random.choice(candidates)
    cameras = sorted(env_info[env_name]["cameras"].keys())
    episodes = sorted(f[env_name][cameras[0]].keys())
    if not episodes:
        print(f"{env_name} has no episodes to preview.")
        return
    episode = random.choice(episodes)

    n = len(cameras)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx, cam in enumerate(cameras):
        ax = axes[idx // cols][idx % cols]
        ax.axis("off")
        try:
            pixels = f[env_name][cam][episode]["pixels"]
            frame = pixels[len(pixels) // 2]  # middle frame of the trajectory
            ax.imshow(frame)
            ax.set_title(cam)
        except Exception as exc:  # noqa: BLE001 - preview should never crash trim
            ax.set_title(f"{cam} (error)")
            print(f"  [warn] could not read {env_name}/{cam}/{episode}: {exc}")
    for idx in range(n, rows * cols):  # hide unused cells
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(f"{env_name}  /  {episode}  (middle frame)")
    fig.tight_layout()
    print(f"Showing {env_name} ({n} cameras). Close the window to continue.")
    plt.show()


def _prompt_selection(prompt: str, names: list[str]) -> list[str]:
    """Ask the user to pick items by index or name. Returns the chosen names."""
    if not names:
        return []
    raw = input(
        f"\n{prompt}\n  (comma/space separated indices or names; "
        f"'all' or blank for none): "
    ).strip()
    if not raw:
        return []
    if raw.lower() == "all":
        return list(names)

    chosen = []
    for token in raw.replace(",", " ").split():
        if token.isdigit():
            i = int(token)
            if 0 <= i < len(names):
                chosen.append(names[i])
            else:
                print(f"  [ignored] index {i} out of range")
        elif token in names:
            chosen.append(token)
        else:
            print(f"  [ignored] unknown selection '{token}'")
    # De-duplicate, preserve order.
    seen = set()
    return [c for c in chosen if not (c in seen or seen.add(c))]


def write_trimmed(src_path, dst_path, remove_envs, remove_cameras):
    """Copy kept envs/cameras (and all sensors) into a new HDF5 file."""
    remove_envs = set(remove_envs)
    remove_cameras = set(remove_cameras)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Preserve any root-level attributes.
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        for env_name in src:
            if env_name in remove_envs:
                continue
            src_env = src[env_name]
            if not isinstance(src_env, h5py.Group):
                continue

            dst_env = dst.create_group(env_name)
            for key, value in src_env.attrs.items():
                dst_env.attrs[key] = value

            kept_cameras = []
            for sub_name in src_env:
                if sub_name != SENSORS_GROUP and sub_name in remove_cameras:
                    continue
                print(f"  copying {env_name}/{sub_name} ...", flush=True)
                src.copy(src_env[sub_name], dst_env, name=sub_name)
                if sub_name != SENSORS_GROUP:
                    kept_cameras.append(sub_name)

            # Keep the camera_names attr consistent with what remains.
            if "camera_names" in dst_env.attrs:
                dst_env.attrs["camera_names"] = kept_cameras


def main():
    parser = argparse.ArgumentParser(
        description="Interactively trim cameras/envs from a MetaWorld HDF5 dataset"
    )
    parser.add_argument("input", type=str, help="Path to the collected HDF5 file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the trimmed file (default: <input>_trimmed.hdf5)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the final confirmation prompt.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_trimmed" + input_path.suffix)
    )
    if output_path.resolve() == input_path.resolve():
        parser.error("output path must differ from input path")

    with h5py.File(input_path, "r") as f:
        print(f"Scanning {input_path} ({_human(input_path.stat().st_size)} on disk)...")
        env_info, camera_sizes = scan(f)

    if not env_info:
        print("No environment groups found; nothing to do.")
        return

    print_summary(env_info, camera_sizes)

    # Optional visual preview of the camera angles from a random environment.
    with h5py.File(input_path, "r") as f:
        while True:
            ans = input(
                "\nPreview camera angles from a random environment in a window? "
                "[y/N]: "
            ).strip().lower()
            if ans in ("y", "yes"):
                preview_cameras(f, env_info)
            else:
                break

    env_names = sorted(env_info.keys())
    camera_names = sorted(camera_sizes.keys())

    remove_envs = _prompt_selection("Which ENVIRONMENTS to remove?", env_names)
    remove_cameras = _prompt_selection("Which CAMERAS to remove (from all kept envs)?", camera_names)

    # Estimate resulting size.
    removed_size = sum(env_info[e]["size"] for e in remove_envs)
    for env_name, info in env_info.items():
        if env_name in remove_envs:
            continue
        for cam in remove_cameras:
            removed_size += info["cameras"].get(cam, 0)
    total = sum(info["size"] for info in env_info.values())

    print("\n=== Plan ===")
    print(f"  Remove envs    : {remove_envs or '(none)'}")
    print(f"  Remove cameras : {remove_cameras or '(none)'}")
    print(f"  Estimated freed: {_human(removed_size)} "
          f"(datasets {_human(total)} -> {_human(total - removed_size)})")
    print(f"  Output file    : {output_path}")

    if not remove_envs and not remove_cameras:
        print("\nNothing selected for removal. Exiting without writing.")
        return

    if not args.yes:
        confirm = input("\nWrite trimmed copy? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return

    print(f"\nWriting {output_path} ...")
    write_trimmed(input_path, output_path, remove_envs, remove_cameras)
    print(
        f"\nDone. Trimmed file: {output_path} "
        f"({_human(output_path.stat().st_size)} on disk)."
    )
    print("Verify it, then delete the original if you no longer need it.")


if __name__ == "__main__":
    main()
