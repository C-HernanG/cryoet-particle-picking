#!/usr/bin/env python3
"""
Create a 2x3 TP/FP/FN comparison grid for EXP2 results.

The script compares the saved baseline (`test`) and fine-tuned (`test_ft`)
inference outputs, matches predictions against ground truth using the same
distance threshold as the experiment notebook, and renders a grid with:

  - rows: base / fine-tuned
  - columns: TP (green), FP (red), FN (yellow)

The displayed z slice is shared across both rows and is selected from the
predicted coordinates themselves so all six panels use the same slice. By
default, each row renders its own saved `full_segmentation_output/*.pt`
volume directly, so the script can work straight from the EXP2 outputs.

Usage:
    conda activate propicker
    python experiments/exp2_umusynth_thy/scripts/create_tp_fp_fn_grid.py
    python experiments/exp2_umusynth_thy/scripts/create_tp_fp_fn_grid.py \
        --tomo tomo_rec_7_snr1.13
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cryoet_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import PROMPT_SIZE  # noqa: E402
from paths import EXP2_COORDS_DIR, EXP2_RESULTS_DIR, UMU_SYNTH_TOMOS_DIR  # noqa: E402


DEFAULT_BASE_DIR = EXP2_RESULTS_DIR / "fine_tuning_deepetpicker" / "test"
DEFAULT_FINETUNED_DIR = EXP2_RESULTS_DIR / "fine_tuning_deepetpicker" / "test_ft"
DEFAULT_OUTPUT_DIR = EXP2_RESULTS_DIR / "visualizations"

PANEL_SPECS = (
    ("TP", "#32CD32"),
    ("FP", "#FF3B30"),
    ("FN", "#FFD60A"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 2x3 base/fine-tuned TP/FP/FN grid for EXP2.",
    )
    parser.add_argument(
        "--tomo",
        type=str,
        default=None,
        help="Tomogram name to visualize. If omitted, a representative tomo is selected automatically.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing the baseline inference outputs.",
    )
    parser.add_argument(
        "--finetuned-dir",
        type=Path,
        default=DEFAULT_FINETUNED_DIR,
        help="Directory containing the fine-tuned inference outputs.",
    )
    parser.add_argument(
        "--coords-dir",
        type=Path,
        default=EXP2_COORDS_DIR,
        help="Directory containing ground-truth coordinate CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to results/exp2_umusynth_thy/visualizations/<tomo>_tp_fp_fn_grid.png",
    )
    parser.add_argument(
        "--projection",
        choices=("slice", "mean", "max"),
        default="slice",
        help=(
            "Background mode. 'slice' renders a shared XY slice selected from "
            "the predicted coordinates. 'mean'/'max' keep the whole-volume XY "
            "projection while still plotting the matched coordinates."
        ),
    )
    parser.add_argument(
        "--slice-half-width",
        type=int,
        default=0,
        help=(
            "Half-width in z around the shared slice used to decide which points "
            "are shown. Default 0 means only points whose rounded z matches the "
            "selected slice are displayed."
        ),
    )
    parser.add_argument(
        "--background-source",
        choices=("pt", "tomo"),
        default="pt",
        help=(
            "Background volume source. 'pt' uses each row's saved "
            "full_segmentation_output/<tomo>.pt directly. 'tomo' uses the "
            "original or preprocessed tomogram volume."
        ),
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=18.0,
        help="Scatter marker size for TP/FP/FN dots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def load_ground_truth_coords(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if table.size == 0:
        return np.empty((0, 3), dtype=float)

    table = np.atleast_1d(table)
    return np.column_stack((table["X"], table["Y"], table["Z"])).astype(float)


def load_predicted_coords(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 3), dtype=float)

    try:
        coords = np.loadtxt(path, delimiter="\t", dtype=float)
    except ValueError:
        return np.empty((0, 3), dtype=float)

    coords = np.atleast_2d(coords)
    if coords.shape[1] >= 4:
        return coords[:, 1:4]
    return coords[:, :3]


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    deltas = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def classify_predictions(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    distance_thresh: float,
) -> dict[str, Any]:
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    if len(gt_coords) and len(pred_coords):
        distances = pairwise_distances(pred_coords, gt_coords)
        for pred_idx in range(len(pred_coords)):
            gt_idx = int(np.argmin(distances[pred_idx]))
            if distances[pred_idx, gt_idx] < distance_thresh and gt_idx not in matched_gt:
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)

    matched_pred_indices = np.array(sorted(matched_pred), dtype=int)
    unmatched_pred_indices = np.array(
        [idx for idx in range(len(pred_coords)) if idx not in matched_pred],
        dtype=int,
    )
    unmatched_gt_indices = np.array(
        [idx for idx in range(len(gt_coords)) if idx not in matched_gt],
        dtype=int,
    )

    tp = len(matched_pred_indices)
    fp = len(unmatched_pred_indices)
    fn = len(unmatched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_coords": pred_coords,
        "gt_coords": gt_coords,
        "TP": pred_coords[matched_pred_indices] if tp else np.empty((0, 3), dtype=float),
        "FP": pred_coords[unmatched_pred_indices] if fp else np.empty((0, 3), dtype=float),
        "FN": gt_coords[unmatched_gt_indices] if fn else np.empty((0, 3), dtype=float),
    }


def available_tomos(run_dir: Path) -> list[str]:
    coords_dir = run_dir / "PredictedLabels" / "Coords_All"
    if not coords_dir.exists():
        return []
    return sorted(path.stem for path in coords_dir.glob("*.coords"))


def evaluate_run(run_dir: Path, coords_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pred_dir = run_dir / "PredictedLabels" / "Coords_All"

    for tomo_name in available_tomos(run_dir):
        gt_path = coords_dir / f"{tomo_name}_thyroglobulin_coords.csv"
        pred_path = pred_dir / f"{tomo_name}.coords"
        gt_coords = load_ground_truth_coords(gt_path)
        pred_coords = load_predicted_coords(pred_path)
        results[tomo_name] = classify_predictions(
            gt_coords=gt_coords,
            pred_coords=pred_coords,
            distance_thresh=PROMPT_SIZE / 2,
        )

    return results


def select_representative_tomo(
    base_results: dict[str, dict[str, Any]],
    finetuned_results: dict[str, dict[str, Any]],
) -> str:
    common_tomos = sorted(set(base_results) & set(finetuned_results))
    if not common_tomos:
        raise RuntimeError("No shared tomograms found between baseline and fine-tuned results.")

    def has_all_panels(result: dict[str, Any]) -> bool:
        return all(len(result[label]) > 0 for label, _ in PANEL_SPECS)

    complete_candidates = [
        tomo_name
        for tomo_name in common_tomos
        if has_all_panels(base_results[tomo_name]) and has_all_panels(finetuned_results[tomo_name])
    ]
    if complete_candidates:
        return max(
            complete_candidates,
            key=lambda tomo_name: (
                float(finetuned_results[tomo_name]["f1"]),
                -float(finetuned_results[tomo_name]["fp"]),
            ),
        )

    return max(
        common_tomos,
        key=lambda tomo_name: (
            float(finetuned_results[tomo_name]["f1"]) - float(base_results[tomo_name]["f1"]),
            float(finetuned_results[tomo_name]["f1"]),
        ),
    )


def find_volume_path(tomo_name: str, base_dir: Path, finetuned_dir: Path) -> tuple[Path | None, str]:
    candidates = (
        (UMU_SYNTH_TOMOS_DIR / f"{tomo_name}.mrc", "original"),
        (finetuned_dir / "raw_data" / f"{tomo_name}.mrc", "raw_data"),
        (base_dir / "raw_data" / f"{tomo_name}.mrc", "raw_data"),
        (finetuned_dir / "data_std" / f"{tomo_name}.mrc", "data_std"),
        (base_dir / "data_std" / f"{tomo_name}.mrc", "data_std"),
    )
    for candidate, source_kind in candidates:
        if candidate.exists():
            return candidate, source_kind
    return None, "blank"


def find_locmap_path(run_dir: Path, tomo_name: str) -> Path | None:
    locmap_path = run_dir / "full_segmentation_output" / f"{tomo_name}.pt"
    if locmap_path.exists():
        return locmap_path
    return None


def load_volume(volume_path: Path, source_kind: str) -> np.ndarray:
    import mrcfile

    with mrcfile.open(volume_path, permissive=True) as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32)

    # The inference helper stores raw_data after inverting the original tomo.
    # Revert that inversion so the fallback display matches the notebook view.
    if source_kind in {"raw_data", "data_std"}:
        volume = -volume
    return volume


def load_pt_volume(locmap_path: Path) -> np.ndarray:
    import torch

    tensor = torch.load(locmap_path, map_location="cpu")

    if isinstance(tensor, dict):
        for key in ("locmap", "prediction", "pred", "output", "tensor", "volume", "logits"):
            if key in tensor:
                tensor = tensor[key]
                break
        else:
            raise RuntimeError(
                f"Unsupported .pt payload in {locmap_path}: expected a tensor or a known tensor key."
            )

    if hasattr(tensor, "detach"):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)

    while array.ndim > 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 4:
        array = array[1] if array.shape[0] > 1 else array[0]

    if array.ndim != 3:
        raise RuntimeError(f"Unexpected .pt volume shape {array.shape} in {locmap_path}")

    return np.asarray(array, dtype=np.float32)


def resolve_background_volume(
    tomo_name: str,
    row_run_dir: Path,
    base_dir: Path,
    finetuned_dir: Path,
    background_source: str,
) -> tuple[np.ndarray | None, str]:
    if background_source == "pt":
        locmap_path = find_locmap_path(row_run_dir, tomo_name)
        if locmap_path is not None:
            return load_pt_volume(locmap_path), str(locmap_path)

    volume_path, volume_source = find_volume_path(tomo_name, base_dir, finetuned_dir)
    if volume_path is not None:
        return load_volume(volume_path, volume_source), str(volume_path)

    return None, "blank_background"


def infer_canvas_shape(*coord_sets: np.ndarray) -> tuple[int, int]:
    max_x = 0.0
    max_y = 0.0
    for coords in coord_sets:
        if len(coords) == 0:
            continue
        finite_coords = coords[np.isfinite(coords).all(axis=1)]
        if len(finite_coords) == 0:
            continue
        max_x = max(max_x, float(np.max(finite_coords[:, 0])))
        max_y = max(max_y, float(np.max(finite_coords[:, 1])))

    width = max(128, int(np.ceil(max_x)) + 24)
    height = max(128, int(np.ceil(max_y)) + 24)
    return height, width


def normalize_image(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise RuntimeError("No finite voxels found in rendered image.")

    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    if np.isclose(vmin, vmax):
        vmin, vmax = float(finite.min()), float(finite.max())

    fill_value = float(np.median(finite))
    image = np.nan_to_num(image, nan=fill_value, posinf=vmax, neginf=vmin)
    return np.clip((image - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def render_background(
    volume: np.ndarray | None,
    projection: str,
    z_index: int,
    blank_shape: tuple[int, int],
) -> np.ndarray:
    if volume is None:
        return np.zeros(blank_shape, dtype=np.float32)

    if projection == "max":
        return normalize_image(np.nanmax(volume, axis=0))
    if projection == "mean":
        with np.errstate(invalid="ignore"):
            return normalize_image(np.nanmean(volume, axis=0))

    z_index = int(np.clip(z_index, 0, volume.shape[0] - 1))
    candidate_z = [z_index]
    for offset in range(1, volume.shape[0]):
        left = z_index - offset
        right = z_index + offset
        if left >= 0:
            candidate_z.append(left)
        if right < volume.shape[0]:
            candidate_z.append(right)

    for current_z in candidate_z:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            image = volume[current_z]
        if np.isfinite(image).any():
            return normalize_image(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(invalid="ignore"):
            return normalize_image(np.nanmean(volume, axis=0))


def rounded_valid_z(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.empty(0, dtype=int)
    finite_mask = np.isfinite(coords[:, 2])
    if not np.any(finite_mask):
        return np.empty(0, dtype=int)
    return np.rint(coords[finite_mask, 2]).astype(int)


def select_shared_slice_from_predictions(
    base_pred_coords: np.ndarray,
    finetuned_pred_coords: np.ndarray,
    fallback_coords: np.ndarray,
    slice_half_width: int,
    volume: np.ndarray | None,
) -> int:
    all_pred_z = np.concatenate(
        [rounded_valid_z(base_pred_coords), rounded_valid_z(finetuned_pred_coords)]
    )

    if volume is not None and volume.shape[0] > 0:
        depth = int(volume.shape[0])
        valid_pred_z = all_pred_z[(all_pred_z >= 0) & (all_pred_z < depth)]
    else:
        valid_pred_z = all_pred_z[all_pred_z >= 0]
        depth = int(valid_pred_z.max()) + 1 if valid_pred_z.size else 0

    if valid_pred_z.size:
        hist = np.bincount(valid_pred_z, minlength=max(depth, int(valid_pred_z.max()) + 1))
        if slice_half_width > 0:
            window = np.ones(2 * slice_half_width + 1, dtype=int)
            scores = np.convolve(hist, window, mode="same")
        else:
            scores = hist

        best_candidates = np.flatnonzero(scores == scores.max())
        if best_candidates.size == 1:
            return int(best_candidates[0])

        exact_scores = hist[best_candidates]
        exact_best = best_candidates[exact_scores == exact_scores.max()]
        center = int(np.rint(valid_pred_z.mean()))
        return int(min(exact_best, key=lambda z_idx: (abs(z_idx - center), z_idx)))

    fallback_z = rounded_valid_z(fallback_coords)
    if volume is not None and volume.shape[0] > 0:
        if fallback_z.size:
            return int(np.clip(int(np.rint(fallback_z.mean())), 0, volume.shape[0] - 1))
        return int(volume.shape[0] // 2)
    if fallback_z.size:
        return int(max(0, int(np.rint(fallback_z.mean()))))
    return 0


def filter_coords_near_z(coords: np.ndarray, z_index: int, slice_half_width: int) -> np.ndarray:
    if len(coords) == 0:
        return coords
    rounded_z = rounded_valid_z(coords)
    if rounded_z.size == 0:
        return np.empty((0, 3), dtype=float)
    finite_mask = np.isfinite(coords[:, 2])
    finite_coords = coords[finite_mask]
    return finite_coords[np.abs(rounded_z - z_index) <= slice_half_width]


def panel_display_data(
    coords: np.ndarray,
    volume: np.ndarray | None,
    projection: str,
    z_index: int,
    slice_half_width: int,
    blank_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    background = render_background(
        volume=volume,
        projection=projection,
        z_index=z_index,
        blank_shape=blank_shape,
    )
    if projection == "slice":
        visible_coords = filter_coords_near_z(
            coords=coords,
            z_index=z_index,
            slice_half_width=slice_half_width,
        )
    else:
        visible_coords = coords
    return background, visible_coords, z_index


def draw_panel(
    ax: plt.Axes,
    background: np.ndarray,
    coords: np.ndarray,
    color: str,
    title: str,
    marker_size: float,
    total_count: int,
    z_index: int | None = None,
) -> None:
    ax.imshow(background, cmap="gray", interpolation="nearest")
    if len(coords):
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=marker_size,
            c=color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.9,
        )
    subtitle = f"shown={len(coords)}/{total_count}"
    if z_index is not None:
        subtitle = f"z={z_index}, {subtitle}"
    ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def build_output_path(tomo_name: str, output: Path | None) -> Path:
    if output is not None:
        return output
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{tomo_name}_tp_fp_fn_grid.png"


def main() -> None:
    args = parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {args.base_dir}")
    if not args.finetuned_dir.exists():
        raise FileNotFoundError(f"Fine-tuned directory not found: {args.finetuned_dir}")

    base_results = evaluate_run(args.base_dir, args.coords_dir)
    finetuned_results = evaluate_run(args.finetuned_dir, args.coords_dir)

    tomo_name = args.tomo or select_representative_tomo(base_results, finetuned_results)
    if tomo_name not in base_results:
        raise KeyError(f"{tomo_name} not found in baseline results.")
    if tomo_name not in finetuned_results:
        raise KeyError(f"{tomo_name} not found in fine-tuned results.")

    base_result = base_results[tomo_name]
    finetuned_result = finetuned_results[tomo_name]

    base_volume, base_background_label = resolve_background_volume(
        tomo_name=tomo_name,
        row_run_dir=args.base_dir,
        base_dir=args.base_dir,
        finetuned_dir=args.finetuned_dir,
        background_source=args.background_source,
    )
    finetuned_volume, finetuned_background_label = resolve_background_volume(
        tomo_name=tomo_name,
        row_run_dir=args.finetuned_dir,
        base_dir=args.base_dir,
        finetuned_dir=args.finetuned_dir,
        background_source=args.background_source,
    )

    blank_shape = infer_canvas_shape(
        base_result["pred_coords"],
        finetuned_result["pred_coords"],
        base_result["gt_coords"],
    )
    depth_hint_volume = base_volume if base_volume is not None else finetuned_volume

    shared_z_index = select_shared_slice_from_predictions(
        base_pred_coords=base_result["pred_coords"],
        finetuned_pred_coords=finetuned_result["pred_coords"],
        fallback_coords=base_result["gt_coords"],
        slice_half_width=args.slice_half_width,
        volume=depth_hint_volume,
    )

    output_path = build_output_path(tomo_name, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.5))

    row_specs = (
        ("Base", base_result, base_volume),
        ("Fine-tuned", finetuned_result, finetuned_volume),
    )

    for row_idx, (row_label, row_result, row_volume) in enumerate(row_specs):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, fontweight="bold")
        for col_idx, (panel_label, color) in enumerate(PANEL_SPECS):
            panel_coords = row_result[panel_label]
            background, visible_coords, z_index = panel_display_data(
                coords=panel_coords,
                volume=row_volume,
                projection=args.projection,
                z_index=shared_z_index,
                slice_half_width=args.slice_half_width,
                blank_shape=blank_shape,
            )
            draw_panel(
                ax=axes[row_idx, col_idx],
                background=background,
                coords=visible_coords,
                color=color,
                title=panel_label,
                marker_size=args.marker_size,
                total_count=len(panel_coords),
                z_index=z_index if args.projection == "slice" else None,
            )

    fig.suptitle(
        (
            f"EXP2 TP/FP/FN comparison for {tomo_name}\n"
            f"projection={args.projection}, shared_predicted_z={shared_z_index}, "
            f"match_radius={PROMPT_SIZE / 2:.1f}px, slice_half_width={args.slice_half_width}, "
            f"background_source={args.background_source}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected tomo: {tomo_name}")
    print(f"Shared predicted slice z: {shared_z_index}")
    for label, result, _row_volume in row_specs:
        print(
            f"{label:11s} "
            f"TP={int(result['tp']):3d} "
            f"FP={int(result['fp']):3d} "
            f"FN={int(result['fn']):3d} "
            f"F1={float(result['f1']):.3f}"
        )
    print(f"Base background: {base_background_label}")
    print(f"Fine-tuned background: {finetuned_background_label}")
    print(f"Saved grid: {output_path}")


if __name__ == "__main__":
    main()
