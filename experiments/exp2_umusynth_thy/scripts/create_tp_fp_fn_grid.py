#!/usr/bin/env python3
"""
Create a 2x3 TP/FP/FN comparison grid for EXP2 results.

The script compares the saved baseline (`test`) and fine-tuned (`test_ft`)
inference outputs, matches predictions against ground truth using the same
distance threshold as the experiment notebook, and renders a grid with:

  - rows: base / fine-tuned
  - columns: TP (green), FP (red), FN (yellow)

By default, the script auto-selects a representative tomogram:
  1. prefer tomograms where both rows have non-empty TP/FP/FN sets
  2. among those, choose the one with the highest fine-tuned F1
  3. otherwise, choose the tomogram with the largest F1 improvement

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
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cryoet_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mrcfile
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
        choices=("mean", "max"),
        default="mean",
        help="XY tomogram projection to display behind the markers.",
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

    coords = np.loadtxt(path, delimiter="\t", dtype=float)
    coords = np.atleast_2d(coords)
    if coords.shape[1] >= 4:
        return coords[:, 1:4]
    return coords[:, :3]


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    deltas = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def classify_predictions(gt_coords: np.ndarray, pred_coords: np.ndarray, distance_thresh: float) -> dict[str, np.ndarray | float | int]:
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
        "TP": pred_coords[matched_pred_indices] if tp else np.empty((0, 3), dtype=float),
        "FP": pred_coords[unmatched_pred_indices] if fp else np.empty((0, 3), dtype=float),
        "FN": gt_coords[unmatched_gt_indices] if fn else np.empty((0, 3), dtype=float),
    }


def available_tomos(run_dir: Path) -> list[str]:
    coords_dir = run_dir / "PredictedLabels" / "Coords_All"
    if not coords_dir.exists():
        return []
    return sorted(path.stem for path in coords_dir.glob("*.coords"))


def evaluate_run(run_dir: Path, coords_dir: Path) -> dict[str, dict[str, np.ndarray | float | int]]:
    results: dict[str, dict[str, np.ndarray | float | int]] = {}
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
    base_results: dict[str, dict[str, np.ndarray | float | int]],
    finetuned_results: dict[str, dict[str, np.ndarray | float | int]],
) -> str:
    common_tomos = sorted(set(base_results) & set(finetuned_results))
    if not common_tomos:
        raise RuntimeError("No shared tomograms found between baseline and fine-tuned results.")

    def has_all_panels(result: dict[str, np.ndarray | float | int]) -> bool:
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


def find_volume_path(tomo_name: str, base_dir: Path, finetuned_dir: Path) -> Path:
    candidates = (
        finetuned_dir / "data_std" / f"{tomo_name}.mrc",
        finetuned_dir / "raw_data" / f"{tomo_name}.mrc",
        base_dir / "data_std" / f"{tomo_name}.mrc",
        base_dir / "raw_data" / f"{tomo_name}.mrc",
        UMU_SYNTH_TOMOS_DIR / f"{tomo_name}.mrc",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate a tomogram volume for "
        f"{tomo_name}. Checked: {', '.join(str(path) for path in candidates)}"
    )


def load_projection(volume_path: Path, projection: str) -> np.ndarray:
    with mrcfile.open(volume_path, permissive=True) as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32)

    if projection == "max":
        image = np.nanmax(volume, axis=0)
    else:
        with np.errstate(invalid="ignore"):
            image = np.nanmean(volume, axis=0)

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise RuntimeError(f"No finite voxels found in {volume_path}")

    vmin, vmax = np.percentile(finite, [1.0, 99.0])
    if np.isclose(vmin, vmax):
        vmin, vmax = float(finite.min()), float(finite.max())

    image = np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin)
    return np.clip((image - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)


def draw_panel(ax: plt.Axes, background: np.ndarray, coords: np.ndarray, color: str, title: str, marker_size: float) -> None:
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
    ax.set_title(f"{title}\n(n={len(coords)})", fontsize=11, fontweight="bold")
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

    volume_path = find_volume_path(tomo_name, args.base_dir, args.finetuned_dir)
    background = load_projection(volume_path, projection=args.projection)

    output_path = build_output_path(tomo_name, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.5))

    row_specs = (
        ("Base", base_results[tomo_name]),
        ("Fine-tuned", finetuned_results[tomo_name]),
    )

    for row_idx, (row_label, row_result) in enumerate(row_specs):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, fontweight="bold")
        for col_idx, (panel_label, color) in enumerate(PANEL_SPECS):
            draw_panel(
                ax=axes[row_idx, col_idx],
                background=background,
                coords=row_result[panel_label],
                color=color,
                title=panel_label,
                marker_size=args.marker_size,
            )

    fig.suptitle(
        (
            f"EXP2 TP/FP/FN comparison for {tomo_name}\n"
            f"projection={args.projection}, match_radius={PROMPT_SIZE / 2:.1f}px, "
            f"background={volume_path.name}"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected tomo: {tomo_name}")
    for label, result in row_specs:
        print(
            f"{label:11s} "
            f"TP={int(result['tp']):3d} "
            f"FP={int(result['fp']):3d} "
            f"FN={int(result['fn']):3d} "
            f"F1={float(result['f1']):.3f}"
        )
    print(f"Volume used: {volume_path}")
    print(f"Saved grid: {output_path}")


if __name__ == "__main__":
    main()
