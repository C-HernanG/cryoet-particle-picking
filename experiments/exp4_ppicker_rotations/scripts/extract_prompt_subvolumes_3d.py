#!/usr/bin/env python3
"""
Extract and visualize the 10 selected EXP4 prompt subvolumes with label masks.

The script reads the selected prompt metadata, converts prompt coordinates from
Angstroms to voxel indices using each tomogram's MRC voxel size, re-extracts
each prompt subvolume as a fixed 37x37x37 patch, extracts the matching
37x37x37 binary label patch, and renders a grid of 3D voxel views where only
label-positive particle voxels are shown. The raw image patches are still saved
in full for reuse.

Usage:
    python experiments/exp4_ppicker_rotations/scripts/extract_prompt_subvolumes_3d.py

    python experiments/exp4_ppicker_rotations/scripts/extract_prompt_subvolumes_3d.py \
        --prompts-json <results_dir>/prompts/all_rotation_prompts.json \
        --output-dir <results_dir>/prompt_subvolumes_3d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from paths import EXP4_RESULTS_DIR, POLNET_SYNTH_LABELS_DIR, POLNET_SYNTH_TOMOS_DIR
from experiments.config import EXP4_NUM_PROMPTS, PROMPT_SIZE


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return unique paths while preserving their input order."""
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def candidate_exp4_roots() -> list[Path]:
    """Return candidate EXP4 result roots that may contain prompt artifacts."""
    return unique_paths(
        [
            Path(EXP4_RESULTS_DIR),
            PROJECT_ROOT / "results" / "exp4_ppicker_rotations",
            PROJECT_ROOT / "results" / "exp4",
        ]
    )


def resolve_default_prompts_source() -> Path | None:
    """Resolve the default prompt metadata source from known EXP4 locations."""
    for root in candidate_exp4_roots():
        combined = root / "prompts" / "all_rotation_prompts.json"
        if combined.exists():
            return combined
        prompt_dir = root / "prompts"
        if prompt_dir.exists() and any(prompt_dir.glob("prompt_*.json")):
            return prompt_dir
    return None


def resolve_default_output_dir() -> Path:
    """Choose a default output directory under the first existing EXP4 results root."""
    for root in candidate_exp4_roots():
        if root.exists():
            return root / "prompt_subvolumes_3d"
    return Path(EXP4_RESULTS_DIR) / "prompt_subvolumes_3d"


def load_mrc(path: Path) -> np.ndarray:
    """Load an MRC volume as a float32 NumPy array."""
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32)


def read_voxel_size_angstrom(path: Path) -> float:
    """Read the voxel size in Angstroms from an MRC header."""
    with mrcfile.open(path, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"Invalid voxel size in MRC header: {path} -> {voxel_size}")
    return voxel_size


def save_mrc(data: np.ndarray, path: Path) -> None:
    """Write a NumPy array to an MRC file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))


def parse_prompt_payload(payload: dict, default_index: int) -> dict:
    """Normalize one prompt JSON payload into the common manifest schema."""
    if "index" in payload and "metadata" in payload:
        metadata = payload.get("metadata", {})
        coords = metadata.get("coords")
        quat = metadata.get("quaternion")
        if coords is None:
            coords = [metadata["x"], metadata["y"], metadata["z"]]
        if quat is None:
            quat = [metadata["q1"], metadata["q2"], metadata["q3"], metadata["q4"]]
        entry = {
            "prompt_idx": int(payload.get("index", metadata.get("prompt_index", default_index))),
            "tomo_name": str(metadata["tomo_name"]),
            "q1": float(quat[0]),
            "q2": float(quat[1]),
            "q3": float(quat[2]),
            "q4": float(quat[3]),
        }
        attach_coordinate_fields(entry, metadata, coords)
        return entry

    metadata = payload.get("metadata", payload)
    coords = metadata.get("coords")
    quat = metadata.get("quaternion")
    if coords is None:
        coords = [metadata["x"], metadata["y"], metadata["z"]]
    if quat is None:
        quat = [metadata["q1"], metadata["q2"], metadata["q3"], metadata["q4"]]
    entry = {
        "prompt_idx": int(metadata.get("prompt_index", payload.get("index", default_index))),
        "tomo_name": str(metadata["tomo_name"]),
        "q1": float(quat[0]),
        "q2": float(quat[1]),
        "q3": float(quat[2]),
        "q4": float(quat[3]),
    }
    attach_coordinate_fields(entry, metadata, coords)
    return entry


def attach_coordinate_fields(entry: dict, metadata: dict, generic_coords: list[float]) -> None:
    """Attach voxel and Angstrom coordinate variants to a normalized prompt entry."""
    entry["x"] = float(generic_coords[0])
    entry["y"] = float(generic_coords[1])
    entry["z"] = float(generic_coords[2])

    coords_angstrom = metadata.get("coords_angstrom")
    if coords_angstrom is None and all(key in metadata for key in ["x_angstrom", "y_angstrom", "z_angstrom"]):
        coords_angstrom = [metadata["x_angstrom"], metadata["y_angstrom"], metadata["z_angstrom"]]
    if coords_angstrom is not None:
        entry["x_angstrom"] = float(coords_angstrom[0])
        entry["y_angstrom"] = float(coords_angstrom[1])
        entry["z_angstrom"] = float(coords_angstrom[2])

    coords_voxel = metadata.get("coords_voxel")
    if coords_voxel is None and all(key in metadata for key in ["x_voxel", "y_voxel", "z_voxel"]):
        coords_voxel = [metadata["x_voxel"], metadata["y_voxel"], metadata["z_voxel"]]
    if coords_voxel is not None:
        entry["x_voxel"] = float(coords_voxel[0])
        entry["y_voxel"] = float(coords_voxel[1])
        entry["z_voxel"] = float(coords_voxel[2])


def resolve_prompt_coordinates(entry: dict, voxel_size_angstrom: float, coords_unit: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return prompt coordinates in both Angstrom and voxel units."""
    if coords_unit == "angstrom":
        if all(key in entry for key in ["x_angstrom", "y_angstrom", "z_angstrom"]):
            coords_angstrom = np.array(
                [entry["x_angstrom"], entry["y_angstrom"], entry["z_angstrom"]],
                dtype=np.float32,
            )
        else:
            coords_angstrom = np.array([entry["x"], entry["y"], entry["z"]], dtype=np.float32)
        coords_voxel = coords_angstrom / voxel_size_angstrom
        return coords_angstrom, coords_voxel, "angstrom"

    if coords_unit == "voxel":
        if all(key in entry for key in ["x_voxel", "y_voxel", "z_voxel"]):
            coords_voxel = np.array(
                [entry["x_voxel"], entry["y_voxel"], entry["z_voxel"]],
                dtype=np.float32,
            )
        else:
            coords_voxel = np.array([entry["x"], entry["y"], entry["z"]], dtype=np.float32)
        coords_angstrom = coords_voxel * voxel_size_angstrom
        return coords_angstrom, coords_voxel, "voxel"

    raise ValueError(f"Unsupported coords unit: {coords_unit}")


def load_prompt_entries(source: Path) -> list[dict]:
    """Load prompt entries from a combined JSON file or a prompt directory."""
    if source.is_dir():
        combined = source / "all_rotation_prompts.json"
        if combined.exists():
            source = combined
        else:
            entries = []
            for idx, prompt_file in enumerate(sorted(source.glob("prompt_*.json"))):
                with open(prompt_file, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                inferred_index = idx
                stem_parts = prompt_file.stem.split("_")
                if stem_parts and stem_parts[-1].isdigit():
                    inferred_index = int(stem_parts[-1])
                entries.append(parse_prompt_payload(payload, inferred_index))
            if not entries:
                raise FileNotFoundError(f"No prompt JSON files found in {source}")
            return sorted(entries, key=lambda item: item["prompt_idx"])

    if not source.exists():
        raise FileNotFoundError(f"Prompt metadata source not found: {source}")

    with open(source, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if "prompts" in payload:
        entries = [
            parse_prompt_payload(prompt_payload, idx)
            for idx, prompt_payload in enumerate(payload["prompts"])
        ]
    else:
        entries = [parse_prompt_payload(payload, 0)]

    return sorted(entries, key=lambda item: item["prompt_idx"])


def extract_subvolume(volume: np.ndarray, x: float, y: float, z: float, size: int) -> np.ndarray:
    """Extract a cubic subvolume around the requested coordinate with zero padding."""
    if size % 2 == 0:
        raise ValueError("Prompt size must be odd.")

    half = size // 2
    center_xyz = np.round(np.array([x, y, z], dtype=np.float32)).astype(int)
    center_zyx = center_xyz[::-1]

    subtomo = np.zeros((size, size, size), dtype=volume.dtype)
    start = center_zyx - half
    stop = center_zyx + half + 1

    src_slices = []
    dst_slices = []
    for axis in range(3):
        src_start = max(int(start[axis]), 0)
        src_stop = min(int(stop[axis]), int(volume.shape[axis]))
        if src_stop <= src_start:
            return subtomo
        dst_start = src_start - int(start[axis])
        dst_stop = dst_start + (src_stop - src_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))

    subtomo[tuple(dst_slices)] = volume[tuple(src_slices)]
    return subtomo


def get_tomo_id_from_name(tomo_name: str) -> int:
    """Extract the numeric tomogram identifier from a PolNet tomogram name."""
    parts = tomo_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse tomogram id from name: {tomo_name}")
    return int(parts[2])


def get_label_filename(tomo_name: str) -> str:
    """Return the label-volume filename associated with a tomogram name."""
    return f"tomo_lbls_{get_tomo_id_from_name(tomo_name)}.mrc"


def build_label_mask(label_subvolume: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build a boolean render mask and basic diagnostics from a label subvolume."""
    mask = np.asarray(label_subvolume) > 0
    diagnostics = {
        "visible_voxels": int(mask.sum()),
    }
    return mask, diagnostics


def prompt_to_euler(prompt: dict) -> np.ndarray:
    """Convert prompt quaternions to XYZ Euler angles in degrees."""
    with np.errstate(invalid="ignore"):
        return Rotation.from_quat(
            [prompt["q1"], prompt["q2"], prompt["q3"], prompt["q4"]]
        ).as_euler("xyz", degrees=True)


def facecolors_from_volume(volume_xyz: np.ndarray, mask_xyz: np.ndarray) -> np.ndarray:
    """Map voxel intensities to RGBA colors for masked 3D rendering."""
    colors = np.zeros(volume_xyz.shape + (4,), dtype=np.float32)
    if not np.any(mask_xyz):
        return colors

    values = np.abs(volume_xyz[mask_xyz])
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low:
        high = low + 1e-6

    normalized = np.clip((np.abs(volume_xyz) - low) / (high - low), 0.0, 1.0)
    colors = plt.cm.inferno(normalized)
    colors[..., -1] = mask_xyz.astype(np.float32) * 0.92
    return colors


def add_xyz_arrows(ax, mask_xyz: np.ndarray) -> None:
    """Draw XYZ axis arrows in an empty corner of the 3D prompt view."""
    shape_xyz = tuple(int(v) for v in mask_xyz.shape)
    size = float(min(shape_xyz))
    length = max(size * 0.12, 2.2)

    max_probe = max(2, int(round(size * 0.16)))
    max_probe = min(max_probe, shape_xyz[0], shape_xyz[1], shape_xyz[2])

    best_corner = None
    for probe in range(max_probe, 0, -1):
        corner_starts = []
        for x0 in [0, max(shape_xyz[0] - probe, 0)]:
            for y0 in [0, max(shape_xyz[1] - probe, 0)]:
                for z0 in [0, max(shape_xyz[2] - probe, 0)]:
                    corner_starts.append((x0, y0, z0))

        empty_corners = [
            c
            for c in corner_starts
            if not np.any(mask_xyz[c[0] : c[0] + probe, c[1] : c[1] + probe, c[2] : c[2] + probe])
        ]
        if empty_corners:
            best_corner = empty_corners[0]
            break

    if best_corner is None:
        return

    origin = np.array(
        [
            1.2 if best_corner[0] == 0 else shape_xyz[0] - 2.2,
            1.2 if best_corner[1] == 0 else shape_xyz[1] - 2.2,
            1.2 if best_corner[2] == 0 else shape_xyz[2] - 2.2,
        ],
        dtype=np.float32,
    )
    direction_sign = np.array(
        [
            1.0 if best_corner[0] == 0 else -1.0,
            1.0 if best_corner[1] == 0 else -1.0,
            1.0 if best_corner[2] == 0 else -1.0,
        ],
        dtype=np.float32,
    )

    axes = [
        ("X", np.array([1.0, 0.0, 0.0], dtype=np.float32), "#ff4d4d"),
        ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float32), "#2ecc71"),
        ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float32), "#4da3ff"),
    ]

    for label, direction, color in axes:
        vec = direction * direction_sign * length
        ax.quiver(
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
            float(vec[0]),
            float(vec[1]),
            float(vec[2]),
            color=color,
            linewidth=1.0,
            arrow_length_ratio=0.2,
        )
        tip = origin + vec * 1.08
        ax.text(float(tip[0]), float(tip[1]), float(tip[2]), label, color=color, fontsize=6, weight="bold")


def render_grid(
    entries: list[dict],
    output_path: Path,
    profile_elev: float,
    profile_azim: float,
    plan_elev: float,
    plan_azim: float,
    elevation_elev: float,
    elevation_azim: float,
) -> None:
    """Render the multi-view 3D prompt grid and save it as a figure."""
    view_specs = [
        ("Profile", profile_elev, profile_azim),
        ("Plan", plan_elev, plan_azim),
        ("Elevation", elevation_elev, elevation_azim),
    ]
    rows = len(entries)
    cols = len(view_specs)
    fig = plt.figure(figsize=(cols * 4.2, max(rows * 3.8, 4.0)))

    for row_idx, entry in enumerate(entries):
        volume_xyz = np.transpose(entry["raw_subvolume"], (2, 1, 0))
        mask_xyz = np.transpose(entry["render_mask"], (2, 1, 0))
        colors = facecolors_from_volume(volume_xyz, mask_xyz)
        euler = entry["euler_xyz_deg"]

        for col_idx, (view_name, elev, azim) in enumerate(view_specs):
            plot_idx = row_idx * cols + col_idx + 1
            ax = fig.add_subplot(rows, cols, plot_idx, projection="3d")
            ax.voxels(mask_xyz, facecolors=colors, edgecolor=None, shade=True)
            add_xyz_arrows(ax, mask_xyz)
            ax.set_box_aspect(mask_xyz.shape)
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()

            title = (
                f"P{entry['prompt_idx']:02d} {view_name} | {entry['tomo_name']}\n"
                f"xyz=({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg"
            )
            ax.set_title(title, fontsize=8, pad=8)

    fig.tight_layout(pad=0.4, w_pad=0.1, h_pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for prompt subvolume extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and visualize the selected EXP4 prompt subvolumes."
    )
    parser.add_argument(
        "--prompts-json",
        type=Path,
        default=None,
        help="Path to all_rotation_prompts.json or to the prompt JSON directory.",
    )
    parser.add_argument(
        "--tomos-dir",
        type=Path,
        default=Path(POLNET_SYNTH_TOMOS_DIR),
        help="Directory containing the source PolNet-generated synthetic tomograms.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path(POLNET_SYNTH_LABELS_DIR),
        help="Directory containing the binary PolNet-generated synthetic label tomograms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the extracted subvolumes and the 3D grid will be written.",
    )
    parser.add_argument(
        "--prompt-size",
        type=int,
        default=PROMPT_SIZE,
        help="Edge size of the extracted cubic prompt volume.",
    )
    parser.add_argument(
        "--coords-unit",
        choices=["angstrom", "voxel"],
        default="voxel",
        help=(
            "Interpret generic prompt coords as Angstroms or voxels. "
            "Default is 'angstrom' because source PolNet coordinates come from the CSV in Angstroms."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=EXP4_NUM_PROMPTS,
        help="Number of prompts to extract from the metadata file.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=25.0,
        help="Matplotlib elevation for the profile view.",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=35.0,
        help="Matplotlib azimuth for the profile view.",
    )
    parser.add_argument(
        "--plan-elev",
        type=float,
        default=0.0,
        help="Matplotlib elevation angle for the plan view (Y direction by default).",
    )
    parser.add_argument(
        "--plan-azim",
        type=float,
        default=90.0,
        help="Matplotlib azimuth angle for the plan view (Y direction by default).",
    )
    parser.add_argument(
        "--elevation-elev",
        type=float,
        default=90.0,
        help="Matplotlib elevation angle for the elevation view (Z direction by default).",
    )
    parser.add_argument(
        "--elevation-azim",
        type=float,
        default=-90.0,
        help="Matplotlib azimuth angle for the elevation view (Z direction by default).",
    )
    return parser


def main() -> None:
    """Run prompt extraction, manifest generation, and 3D rendering."""
    parser = build_argument_parser()
    args = parser.parse_args()

    prompts_source = args.prompts_json or resolve_default_prompts_source()
    if prompts_source is None:
        searched = [
            root / "prompts" / "all_rotation_prompts.json" for root in candidate_exp4_roots()
        ]
        message = "\n".join(f"  - {path}" for path in searched)
        raise FileNotFoundError(
            "Could not find EXP4 prompt metadata automatically. "
            "Provide --prompts-json explicitly.\nSearched:\n"
            f"{message}"
        )

    output_dir = args.output_dir or resolve_default_output_dir()
    raw_dir = output_dir / "raw_subvolumes"
    label_dir = output_dir / "label_subvolumes"
    figure_path = output_dir / "prompt_subvolumes_3d_grid.png"

    entries = load_prompt_entries(prompts_source)
    if args.num_prompts > 0:
        entries = entries[: args.num_prompts]
    if not entries:
        raise ValueError("No prompt entries were loaded.")

    tomo_cache: dict[str, np.ndarray] = {}
    tomo_voxel_size_cache: dict[str, float] = {}
    label_cache: dict[str, np.ndarray] = {}
    rendered_entries: list[dict] = []
    manifest_rows: list[dict] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXP4 PROMPT SUBVOLUME EXTRACTION")
    print("=" * 70)
    print(f"Prompt metadata source: {prompts_source}")
    print(f"Tomograms dir:         {args.tomos_dir}")
    print(f"Labels dir:            {args.labels_dir}")
    print(f"Output dir:            {output_dir}")
    print(f"Prompts to process:    {len(entries)}")
    print()

    for entry in entries:
        tomo_name = entry["tomo_name"]
        tomo_path = args.tomos_dir / f"{tomo_name}.mrc"
        if not tomo_path.exists():
            raise FileNotFoundError(f"Tomogram not found for prompt {entry['prompt_idx']}: {tomo_path}")

        if tomo_name not in tomo_cache:
            print(f"Loading {tomo_name} ...")
            tomo_cache[tomo_name] = load_mrc(tomo_path)
            tomo_voxel_size_cache[tomo_name] = read_voxel_size_angstrom(tomo_path)
        if tomo_name not in label_cache:
            label_path = args.labels_dir / get_label_filename(tomo_name)
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Label tomogram not found for prompt {entry['prompt_idx']}: {label_path}"
                )
            label_cache[tomo_name] = load_mrc(label_path)

        voxel_size_angstrom = tomo_voxel_size_cache[tomo_name]
        coords_angstrom, coords_voxel, coords_unit_used = resolve_prompt_coordinates(
            entry,
            voxel_size_angstrom=voxel_size_angstrom,
            coords_unit=args.coords_unit,
        )

        raw_subvolume = extract_subvolume(
            tomo_cache[tomo_name],
            x=float(coords_voxel[0]),
            y=float(coords_voxel[1]),
            z=float(coords_voxel[2]),
            size=args.prompt_size,
        )
        label_subvolume = extract_subvolume(
            label_cache[tomo_name],
            x=float(coords_voxel[0]),
            y=float(coords_voxel[1]),
            z=float(coords_voxel[2]),
            size=args.prompt_size,
        )
        render_mask, diagnostics = build_label_mask(label_subvolume)
        euler_xyz_deg = prompt_to_euler(entry)

        prompt_stem = f"prompt_{entry['prompt_idx']:02d}"
        raw_path = raw_dir / f"{prompt_stem}_raw.mrc"
        label_path = label_dir / f"{prompt_stem}_labels.mrc"

        save_mrc(raw_subvolume, raw_path)
        save_mrc(label_subvolume.astype(np.float32), label_path)

        if not np.any(render_mask):
            raise ValueError(
                f"Empty label mask for prompt {entry['prompt_idx']:02d} from {tomo_name}. "
                "Check label alignment or coordinate convention."
            )

        rendered_entries.append(
            {
                **entry,
                "raw_subvolume": raw_subvolume,
                "render_mask": render_mask,
                "euler_xyz_deg": euler_xyz_deg,
                "coords_voxel": coords_voxel,
            }
        )
        manifest_rows.append(
            {
                **entry,
                "coords_unit_used": coords_unit_used,
                "voxel_size_angstrom": float(voxel_size_angstrom),
                "x_used_angstrom": float(coords_angstrom[0]),
                "y_used_angstrom": float(coords_angstrom[1]),
                "z_used_angstrom": float(coords_angstrom[2]),
                "x_used_voxel": float(coords_voxel[0]),
                "y_used_voxel": float(coords_voxel[1]),
                "z_used_voxel": float(coords_voxel[2]),
                "euler_x_deg": float(euler_xyz_deg[0]),
                "euler_y_deg": float(euler_xyz_deg[1]),
                "euler_z_deg": float(euler_xyz_deg[2]),
                "raw_shape_zyx": "x".join(str(v) for v in raw_subvolume.shape),
                "label_shape_zyx": "x".join(str(v) for v in label_subvolume.shape),
                "raw_subvolume_mrc": str(raw_path),
                "label_subvolume_mrc": str(label_path),
                **diagnostics,
            }
        )
        print(
            f"  Prompt {entry['prompt_idx']:02d}: "
            f"patch {tuple(raw_subvolume.shape)} "
            f"(coords {coords_angstrom.round(2).tolist()} A -> {coords_voxel.round(2).tolist()} vox, "
            f"label voxels={diagnostics['visible_voxels']})"
        )

    render_grid(
        rendered_entries,
        output_path=figure_path,
        profile_elev=args.view_elev,
        profile_azim=args.view_azim,
        plan_elev=args.plan_elev,
        plan_azim=args.plan_azim,
        elevation_elev=args.elevation_elev,
        elevation_azim=args.elevation_azim,
    )

    manifest_path = output_dir / "prompt_subvolumes_manifest.csv"
    pd.DataFrame(manifest_rows).sort_values("prompt_idx").to_csv(manifest_path, index=False)

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "prompts_source": str(prompts_source),
                "tomos_dir": str(args.tomos_dir),
                "labels_dir": str(args.labels_dir),
                "output_dir": str(output_dir),
                "prompt_size": args.prompt_size,
                "coords_unit": args.coords_unit,
                "num_prompts": len(rendered_entries),
                "grid_figure": str(figure_path),
                "manifest_csv": str(manifest_path),
            },
            fh,
            indent=2,
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Manifest: {manifest_path}")
    print(f"3D grid:  {figure_path}")


if __name__ == "__main__":
    main()
