from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from scipy.fft import fftn, fftshift
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

GLOBAL_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
THYROGLOBULIN_C2_AXIS_LOCAL = np.array([0.0, 0.0, 1.0], dtype=float)
THYROGLOBULIN_C2_ROTATION = Rotation.from_rotvec(np.pi * THYROGLOBULIN_C2_AXIS_LOCAL)

CAUSAL_ROTATION_ACQUISITION_FEATURES = [
    "c2_axis_to_global_z_deg",
    "c2_rotation_nn_deg",
    "symmetry_alias_gap_deg",
    "missing_wedge_anisotropy",
]
CAUSAL_CONFOUNDER_FEATURES = [
    "quality_score",
    "dist_to_center_norm",
    "edge_distance_norm",
    "mass_center_shift",
    "inertia_anisotropy",
]
CAUSAL_PERMUTATION_ITERATIONS = 1000
CAUSAL_RANDOM_STATE = 42
C2_NEAR_EQUIVALENCE_DEG = 20.0
C2_ALIAS_RAW_DEG = 150.0


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def parse_snr_from_tomo_name(tomo_name: str) -> float:
    match = re.search(r"snr([0-9]+(?:\.[0-9]+)?)", str(tomo_name))
    return float(match.group(1)) if match else np.nan


def parse_tomo_id(tomo_name: str) -> float:
    match = re.search(r"tomo_rec_(\d+)", str(tomo_name))
    return float(match.group(1)) if match else np.nan


def safe_pearsonr(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    return stats.pearsonr(x, y)


def safe_spearmanr(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    return stats.spearmanr(x[mask], y[mask])


def safe_mannwhitneyu(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue


def pooled_effect_size(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def _normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Quaternion has invalid norm.")
    return q / norm


def _canonicalize_quaternion_array(quat_array):
    quat_array = np.asarray(quat_array, dtype=float)
    if quat_array.ndim != 2 or quat_array.shape[1] != 4:
        raise ValueError("Expected quaternion array with shape (N, 4).")

    quat_array = quat_array / np.clip(np.linalg.norm(quat_array, axis=1, keepdims=True), 1e-8, None)
    signs = np.ones((len(quat_array), 1), dtype=float)
    for idx, row in enumerate(quat_array):
        non_zero = np.flatnonzero(np.abs(row) > 1e-12)
        if len(non_zero) > 0 and row[non_zero[0]] < 0:
            signs[idx, 0] = -1.0
    return quat_array * signs


def rotation_geodesic_distance_deg(rot_a, rot_b):
    return float(np.degrees((rot_a.inv() * rot_b).magnitude()))


def quaternion_angular_distance_deg(q1, q2):
    rot_a = Rotation.from_quat(_normalize_quaternion(q1))
    rot_b = Rotation.from_quat(_normalize_quaternion(q2))
    return rotation_geodesic_distance_deg(rot_a, rot_b)


def quaternion_angular_distance_c2_deg(q1, q2, symmetry_rotation=THYROGLOBULIN_C2_ROTATION):
    rot_a = Rotation.from_quat(_normalize_quaternion(q1))
    rot_b = Rotation.from_quat(_normalize_quaternion(q2))
    return min(
        rotation_geodesic_distance_deg(rot_a, rot_b),
        rotation_geodesic_distance_deg(rot_a, rot_b * symmetry_rotation),
    )


def quaternion_to_rotation_features(q1, q2, q3, q4):
    q = _normalize_quaternion([q1, q2, q3, q4])

    rot = Rotation.from_quat(q)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        euler = rot.as_euler("xyz", degrees=True)

    mat = rot.as_matrix()
    z_alignments = np.abs(mat[2, :])
    z_angles = np.degrees(np.arccos(np.clip(z_alignments, 0.0, 1.0)))
    c2_axis_global = mat[:, 2]
    c2_axis_alignments = np.abs(c2_axis_global)
    c2_axis_angles = np.degrees(np.arccos(np.clip(c2_axis_alignments, 0.0, 1.0)))

    return {
        "euler_x": float(euler[0]),
        "euler_y": float(euler[1]),
        "euler_z": float(euler[2]),
        "axis_x_to_z_abs_cos": float(z_alignments[0]),
        "axis_y_to_z_abs_cos": float(z_alignments[1]),
        "axis_z_to_z_abs_cos": float(z_alignments[2]),
        "axis_x_to_z_deg": float(z_angles[0]),
        "axis_y_to_z_deg": float(z_angles[1]),
        "axis_z_to_z_deg": float(z_angles[2]),
        "closest_axis_to_z_deg": float(z_angles.min()),
        "farthest_axis_to_z_deg": float(z_angles.max()),
        "beam_axis_alignment_max": float(z_alignments.max()),
        "beam_axis_alignment_mean": float(z_alignments.mean()),
        "beam_axis_alignment_std": float(z_alignments.std()),
        "local_z_to_global_z_deg": float(z_angles[2]),
        "local_z_in_plane_abs": float(np.sqrt(mat[0, 2] ** 2 + mat[1, 2] ** 2)),
        "c2_axis_alignment_abs_cos": float(c2_axis_alignments[2]),
        "c2_axis_to_global_x_deg": float(c2_axis_angles[0]),
        "c2_axis_to_global_y_deg": float(c2_axis_angles[1]),
        "c2_axis_to_global_z_deg": float(c2_axis_angles[2]),
        "c2_axis_in_plane_abs": float(np.sqrt(c2_axis_global[0] ** 2 + c2_axis_global[1] ** 2)),
    }


def compute_exact_distance_to_center(x, y, z, tomo_shape):
    center_xyz = np.array(
        [
            (tomo_shape[2] - 1) / 2.0,
            (tomo_shape[1] - 1) / 2.0,
            (tomo_shape[0] - 1) / 2.0,
        ],
        dtype=np.float32,
    )
    coords_xyz = np.array([x, y, z], dtype=np.float32)
    dist = float(np.linalg.norm(coords_xyz - center_xyz))
    max_dist = float(np.linalg.norm(center_xyz)) if np.linalg.norm(center_xyz) > 0 else 1.0
    return dist, dist / max_dist


def _robust_norm(vol, eps=1e-6):
    v = vol.astype(np.float32, copy=False)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return (v - med) / (mad + eps)


def _center_border_ratio(vol, border=4, centre_frac=0.45, eps=1e-6):
    m = vol.shape[0]
    a = np.abs(vol)
    b = border

    border_mask = np.zeros((m, m, m), dtype=bool)
    border_mask[:b, :, :] = True
    border_mask[-b:, :, :] = True
    border_mask[:, :b, :] = True
    border_mask[:, -b:, :] = True
    border_mask[:, :, :b] = True
    border_mask[:, :, -b:] = True

    c = int(round(m * centre_frac))
    c = max(3, min(c, m))
    s = (m - c) // 2
    centre_mask = np.zeros((m, m, m), dtype=bool)
    centre_mask[s : s + c, s : s + c, s : s + c] = True

    centre_energy = a[centre_mask].mean()
    border_energy = a[border_mask].mean()
    return (centre_energy + eps) / (border_energy + eps)


def _freq_structure_ratio(vol, k_low=0.08, k_high=0.30, eps=1e-8):
    m = vol.shape[0]
    F = np.fft.fftn(vol)
    P = (np.abs(F) ** 2).astype(np.float64)
    f = np.fft.fftfreq(m)
    kx, ky, kz = np.meshgrid(f, f, f, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k = k / (k.max() + eps)
    mid = (k >= k_low) & (k < k_high)
    high = k >= k_high
    E_mid = P[mid].sum()
    E_high = P[high].sum()
    return (E_mid + eps) / (E_high + eps)


def _dc_penalty(vol, eps=1e-6):
    return np.abs(vol.mean()) / (vol.std() + eps)


def compute_quality_score(
    subtomo,
    border=4,
    centre_frac=0.45,
    k_low=0.08,
    k_high=0.30,
    w_freq=1.0,
    w_centre=0.9,
    w_dc=0.6,
):
    vol = _to_numpy(subtomo)
    vol = vol if vol.ndim == 3 else np.squeeze(vol)
    vol = _robust_norm(vol)

    fr = _freq_structure_ratio(vol, k_low=k_low, k_high=k_high)
    cr = _center_border_ratio(vol, border=border, centre_frac=centre_frac)
    dc = _dc_penalty(vol)
    score = (w_freq * np.log(fr) + w_centre * np.log(cr)) - (w_dc * dc)

    features = {
        "freq_ratio": float(fr),
        "centre_ratio": float(cr),
        "dc_penalty": float(dc),
    }
    return float(score), features


def compute_subtomo_features(subtomo):
    vol = _to_numpy(subtomo)
    vol = vol if vol.ndim == 3 else np.squeeze(vol)
    vol = vol.astype(np.float32)
    vol_norm = _robust_norm(vol)
    abs_vol = np.abs(vol_norm) + 1e-8

    features = {
        "mean": float(vol.mean()),
        "std": float(vol.std()),
        "min": float(vol.min()),
        "max": float(vol.max()),
        "range": float(vol.max() - vol.min()),
        "skewness": float(stats.skew(vol.flatten())),
        "kurtosis": float(stats.kurtosis(vol.flatten())),
    }

    quality_score, quality_parts = compute_quality_score(vol)
    features["quality_score"] = float(quality_score)
    features.update(quality_parts)

    features["energy"] = float(np.sum(vol_norm**2))
    features["contrast"] = float(vol.std() / (np.abs(vol.mean()) + 1e-8))

    F = fftn(vol_norm)
    P = np.abs(F) ** 2
    P_shifted = fftshift(P)

    m = vol.shape[0]
    center = m // 2
    z_idx, y_idx, x_idx = np.ogrid[:m, :m, :m]
    r = np.sqrt((z_idx - center) ** 2 + (y_idx - center) ** 2 + (x_idx - center) ** 2)
    r_norm = r / (m / 2)

    low_band = r_norm <= 0.15
    mid_band = (r_norm > 0.15) & (r_norm <= 0.35)
    high_band = r_norm > 0.35

    total_power = P_shifted.sum() + 1e-8
    features["power_low"] = float(P_shifted[low_band].sum() / total_power)
    features["power_mid"] = float(P_shifted[mid_band].sum() / total_power)
    features["power_high"] = float(P_shifted[high_band].sum() / total_power)
    features["freq_ratio_mid_high"] = float(features["power_mid"] / (features["power_high"] + 1e-8))

    central_xy = P_shifted[center, :, :].sum()
    central_xz = P_shifted[:, center, :].sum()
    central_yz = P_shifted[:, :, center].sum()
    features["anisotropy_xy_xz"] = float(central_xy / (central_xz + 1e-8))
    features["anisotropy_xy_yz"] = float(central_xy / (central_yz + 1e-8))
    features["missing_wedge_anisotropy"] = float(
        0.5
        * (
            abs(np.log(features["anisotropy_xy_xz"] + 1e-8))
            + abs(np.log(features["anisotropy_xy_yz"] + 1e-8))
        )
    )

    border = 4
    center_cube = vol_norm[border:-border, border:-border, border:-border]
    features["center_mean"] = float(center_cube.mean())
    features["center_std"] = float(center_cube.std())
    features["center_energy_ratio"] = float(np.sum(center_cube**2) / (features["energy"] + 1e-8))

    weights = abs_vol
    coords = np.indices(vol.shape).astype(np.float32)
    weight_sum = weights.sum()
    com = np.array(
        [float((coords[i] * weights).sum() / weight_sum) for i in range(3)],
        dtype=np.float32,
    )
    geom_center = np.array([(m - 1) / 2.0] * 3, dtype=np.float32)
    features["mass_center_shift"] = float(np.linalg.norm(com - geom_center))

    inner_shell = r_norm <= 0.25
    outer_shell = r_norm > 0.5
    features["radial_inner_energy"] = float(abs_vol[inner_shell].mean())
    features["radial_outer_energy"] = float(abs_vol[outer_shell].mean())
    features["radial_inner_outer_ratio"] = float(
        features["radial_inner_energy"] / (features["radial_outer_energy"] + 1e-8)
    )

    gz, gy, gx = np.gradient(vol_norm)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    features["gradient_energy"] = float((grad_mag**2).mean())

    flat_coords = np.stack(
        [coords[0].ravel(), coords[1].ravel(), coords[2].ravel()],
        axis=1,
    )
    centered = flat_coords - com
    weighted = centered * weights.ravel()[:, None]
    cov = weighted.T @ centered / weight_sum
    eigvals = np.sort(np.linalg.eigvalsh(cov))
    features["inertia_anisotropy"] = float(eigvals[-1] / (eigvals[0] + 1e-8))

    return features


def radial_power_spectrum(vol):
    vol = _to_numpy(vol)
    F = fftn(vol)
    P = np.abs(fftshift(F)) ** 2

    m = vol.shape[0]
    center = m // 2
    z_idx, y_idx, x_idx = np.ogrid[:m, :m, :m]
    r = np.sqrt((z_idx - center) ** 2 + (y_idx - center) ** 2 + (x_idx - center) ** 2).astype(int)

    max_r = center
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.any():
            radial_profile[i] = P[mask].mean()
    return radial_profile


def _select_focus_results(df_results, study_num_prompts, checkpoint_type, increment):
    df_focus = df_results[
        (df_results["checkpoint_type"] == checkpoint_type)
        & (df_results["increment"] == increment)
        & (df_results["prompt_idx"] < study_num_prompts)
    ].copy()

    if len(df_focus) > 0:
        return df_focus, checkpoint_type, increment

    group_sizes = (
        df_results[df_results["prompt_idx"] < study_num_prompts]
        .groupby(["checkpoint_type", "increment"])
        .size()
        .sort_values(ascending=False)
    )
    if group_sizes.empty:
        return pd.DataFrame(), checkpoint_type, increment

    checkpoint_type, increment = group_sizes.index[0]
    df_focus = df_results[
        (df_results["checkpoint_type"] == checkpoint_type)
        & (df_results["increment"] == increment)
        & (df_results["prompt_idx"] < study_num_prompts)
    ].copy()
    return df_focus, checkpoint_type, increment


def _prepare_prompt_dataframe(df_selected, prompt_info, study_num_prompts):
    n_prompts = min(study_num_prompts, len(prompt_info))
    rows = []
    for idx in range(n_prompts):
        info = prompt_info[idx]
        rows.append(
            {
                "prompt_idx": idx,
                "tomo_name": info["tomo_name"],
                "x": float(info["x"]),
                "y": float(info["y"]),
                "z": float(info["z"]),
                "q1": float(info["q1"]),
                "q2": float(info["q2"]),
                "q3": float(info["q3"]),
                "q4": float(info["q4"]),
                "source_snr": parse_snr_from_tomo_name(info["tomo_name"]),
                "source_tomo_id": parse_tomo_id(info["tomo_name"]),
            }
        )

    df_prompts = pd.DataFrame(rows)

    if df_selected is not None and len(df_selected) >= n_prompts:
        df_selected_reset = df_selected.reset_index(drop=True).copy()
        df_selected_reset["prompt_idx"] = np.arange(len(df_selected_reset))
        merge_cols = ["prompt_idx"]
        if "quality_score" in df_selected_reset.columns:
            merge_cols.append("quality_score")
        if "tomo_name" in df_selected_reset.columns:
            merge_cols.append("tomo_name")
        df_prompts = df_prompts.merge(
            df_selected_reset[merge_cols].rename(
                columns={"quality_score": "selection_quality_score", "tomo_name": "selection_tomo_name"}
            ),
            on="prompt_idx",
            how="left",
        )

    return df_prompts


def _load_tomo_shapes(tomo_dir, tomo_names):
    shape_cache = {}
    tomo_dir = Path(tomo_dir)
    for tomo_name in sorted(set(tomo_names)):
        tomo_path = tomo_dir / f"{tomo_name}.mrc"
        if not tomo_path.exists():
            continue
        with mrcfile.open(str(tomo_path), mode="r", permissive=True) as mrc:
            shape_cache[tomo_name] = tuple(int(v) for v in mrc.data.shape)
    return shape_cache


def _build_prompt_performance(df_focus):
    df_focus = df_focus.copy()
    df_focus["val_tomo_snr"] = df_focus["tomo_name"].map(parse_snr_from_tomo_name)

    df_prompt_perf = (
        df_focus.groupby("prompt_idx")
        .agg(
            validation_tomo_count=("tomo_name", "nunique"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            precision_min=("precision", "min"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            recall_min=("recall", "min"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            f1_min=("f1", "min"),
            tp_sum=("tp", "sum"),
            fp_sum=("fp", "sum"),
            fn_sum=("fn", "sum"),
        )
        .reset_index()
    )

    df_prompt_perf["pred_count_sum"] = df_prompt_perf["tp_sum"] + df_prompt_perf["fp_sum"]
    df_prompt_perf["gt_count_sum"] = df_prompt_perf["tp_sum"] + df_prompt_perf["fn_sum"]
    df_prompt_perf["pred_gt_ratio_mean"] = df_prompt_perf["pred_count_sum"] / (
        df_prompt_perf["gt_count_sum"] + 1e-8
    )
    df_prompt_perf["fp_per_tp_mean"] = df_prompt_perf["fp_sum"] / (df_prompt_perf["tp_sum"] + 1e-8)

    snr_rows = []
    for prompt_idx, group in df_focus.groupby("prompt_idx"):
        recall_corr, recall_p_value = safe_pearsonr(group["val_tomo_snr"], group["recall"])
        f1_corr, f1_p_value = safe_pearsonr(group["val_tomo_snr"], group["f1"])
        if group["val_tomo_snr"].nunique() >= 2:
            recall_slope = stats.linregress(group["val_tomo_snr"], group["recall"]).slope
            f1_slope = stats.linregress(group["val_tomo_snr"], group["f1"]).slope
        else:
            recall_slope = np.nan
            f1_slope = np.nan
        snr_rows.append(
            {
                "prompt_idx": prompt_idx,
                "validation_snr_f1_r": f1_corr,
                "validation_snr_f1_p": f1_p_value,
                "validation_snr_f1_slope": f1_slope,
                "validation_snr_recall_r": recall_corr,
                "validation_snr_recall_p": recall_p_value,
                "validation_snr_recall_slope": recall_slope,
            }
        )

    df_prompt_perf = df_prompt_perf.merge(pd.DataFrame(snr_rows), on="prompt_idx", how="left")
    return df_prompt_perf


def _build_prompt_orientation_context(df_prompts, proximity_degrees=(30.0, 45.0)):
    if len(df_prompts) == 0:
        return pd.DataFrame(columns=["prompt_idx"])

    prompt_rows = df_prompts.sort_values("prompt_idx").reset_index(drop=True)
    quats = prompt_rows[["q1", "q2", "q3", "q4"]].to_numpy(dtype=float)
    n_prompts = len(prompt_rows)

    raw_dist = np.full((n_prompts, n_prompts), np.inf, dtype=float)
    c2_dist = np.full((n_prompts, n_prompts), np.inf, dtype=float)

    for i in range(n_prompts):
        for j in range(i + 1, n_prompts):
            raw_ij = quaternion_angular_distance_deg(quats[i], quats[j])
            c2_ij = quaternion_angular_distance_c2_deg(quats[i], quats[j])
            raw_dist[i, j] = raw_dist[j, i] = raw_ij
            c2_dist[i, j] = c2_dist[j, i] = c2_ij

    rows = []
    for i, prompt_idx in enumerate(prompt_rows["prompt_idx"].tolist()):
        raw_nn = np.nan if n_prompts == 1 else float(np.min(raw_dist[i]))
        c2_nn = np.nan if n_prompts == 1 else float(np.min(c2_dist[i]))
        row = {
            "prompt_idx": prompt_idx,
            "raw_rotation_nn_deg": raw_nn,
            "c2_rotation_nn_deg": c2_nn,
            "symmetry_alias_gap_deg": (
                raw_nn - c2_nn if np.isfinite(raw_nn) and np.isfinite(c2_nn) else np.nan
            ),
        }
        for deg in proximity_degrees:
            suffix = str(int(deg))
            row[f"raw_neighbour_count_{suffix}deg"] = int(np.sum(raw_dist[i] < deg))
            row[f"c2_neighbour_count_{suffix}deg"] = int(np.sum(c2_dist[i] < deg))
        rows.append(row)

    return pd.DataFrame(rows)


def _build_subtomo_feature_table(df_prompts, subtomos, embeddings, tomo_shapes, study_num_prompts):
    n_prompts = min(study_num_prompts, len(subtomos), len(embeddings))
    embedding_array = _to_numpy(embeddings)[:n_prompts]
    emb_centroid = embedding_array.mean(axis=0)
    emb_dist = cdist(embedding_array, embedding_array)
    np.fill_diagonal(emb_dist, np.inf)

    emb_scaled = StandardScaler().fit_transform(embedding_array)
    emb_pca = PCA(n_components=2).fit_transform(emb_scaled)

    rows = []
    spectra = {}
    for idx in range(n_prompts):
        prompt_row = df_prompts.loc[df_prompts["prompt_idx"] == idx].iloc[0]
        subtomo = subtomos[idx]
        features = compute_subtomo_features(subtomo)
        rot_features = quaternion_to_rotation_features(
            prompt_row["q1"],
            prompt_row["q2"],
            prompt_row["q3"],
            prompt_row["q4"],
        )

        tomo_shape = tomo_shapes.get(prompt_row["tomo_name"])
        if tomo_shape is not None:
            dist_exact, dist_norm = compute_exact_distance_to_center(
                prompt_row["x"],
                prompt_row["y"],
                prompt_row["z"],
                tomo_shape,
            )
            x_norm = float(prompt_row["x"] / max(tomo_shape[2] - 1, 1))
            y_norm = float(prompt_row["y"] / max(tomo_shape[1] - 1, 1))
            z_norm = float(prompt_row["z"] / max(tomo_shape[0] - 1, 1))
            edge_distance_norm = float(
                min(x_norm, 1 - x_norm, y_norm, 1 - y_norm, z_norm, 1 - z_norm)
            )
            z_center_offset_abs = float(abs(z_norm - 0.5))
        else:
            dist_exact = np.nan
            dist_norm = np.nan
            x_norm = np.nan
            y_norm = np.nan
            z_norm = np.nan
            edge_distance_norm = np.nan
            z_center_offset_abs = np.nan

        emb = embedding_array[idx]
        rows.append(
            {
                "prompt_idx": idx,
                **features,
                **rot_features,
                "dist_to_center_exact": dist_exact,
                "dist_to_center_norm": dist_norm,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "z_norm": z_norm,
                "edge_distance_norm": edge_distance_norm,
                "z_center_offset_abs": z_center_offset_abs,
                "emb_norm": float(np.linalg.norm(emb)),
                "emb_mean": float(np.mean(emb)),
                "emb_std": float(np.std(emb)),
                "emb_dist_to_centroid": float(np.linalg.norm(emb - emb_centroid)),
                "emb_nn_dist": float(np.min(emb_dist[idx])),
                "emb_pc1": float(emb_pca[idx, 0]),
                "emb_pc2": float(emb_pca[idx, 1]),
            }
        )
        spectra[idx] = radial_power_spectrum(subtomo)

    df_features = pd.DataFrame(rows)
    df_context = _build_prompt_orientation_context(df_prompts.head(n_prompts).copy())
    df_features = df_features.merge(df_context, on="prompt_idx", how="left")
    return df_features, spectra


def _build_correlation_table(df_prompt_analysis):
    feature_groups = {
        "symmetry": [
            "c2_axis_to_global_z_deg",
            "c2_axis_alignment_abs_cos",
            "c2_axis_in_plane_abs",
            "c2_rotation_nn_deg",
            "symmetry_alias_gap_deg",
            "c2_neighbour_count_30deg",
            "c2_neighbour_count_45deg",
        ],
        "acquisition": [
            "closest_axis_to_z_deg",
            "farthest_axis_to_z_deg",
            "beam_axis_alignment_max",
            "beam_axis_alignment_mean",
            "local_z_to_global_z_deg",
            "axis_x_to_z_deg",
            "axis_y_to_z_deg",
            "axis_z_to_z_deg",
            "euler_x",
            "euler_y",
            "euler_z",
            "missing_wedge_anisotropy",
            "freq_ratio_mid_high",
            "gradient_energy",
        ],
        "position": [
            "z_norm",
            "z_center_offset_abs",
            "dist_to_center_norm",
            "edge_distance_norm",
            "x_norm",
            "y_norm",
        ],
        "quality": [
            "source_snr",
            "quality_score",
            "selection_quality_score",
            "freq_ratio",
            "centre_ratio",
            "dc_penalty",
            "contrast",
            "mass_center_shift",
            "inertia_anisotropy",
            "radial_inner_outer_ratio",
            "center_energy_ratio",
            "power_low",
            "power_mid",
            "power_high",
        ],
        "embedding": [
            "emb_dist_to_centroid",
            "emb_nn_dist",
            "emb_std",
            "emb_norm",
            "emb_pc1",
            "emb_pc2",
        ],
        "response": [
            "validation_snr_f1_r",
            "validation_snr_f1_slope",
            "validation_snr_recall_r",
            "validation_snr_recall_slope",
            "pred_gt_ratio_mean",
            "fp_per_tp_mean",
        ],
    }

    rows = []
    for group_name, feats in feature_groups.items():
        for feat in feats:
            if feat not in df_prompt_analysis.columns:
                continue
            pearson_f1, pearson_f1_p = safe_pearsonr(df_prompt_analysis[feat], df_prompt_analysis["f1_mean"])
            pearson_recall, pearson_recall_p = safe_pearsonr(df_prompt_analysis[feat], df_prompt_analysis["recall_mean"])
            spearman_f1, spearman_f1_p = safe_spearmanr(df_prompt_analysis[feat], df_prompt_analysis["f1_mean"])
            spearman_recall, spearman_recall_p = safe_spearmanr(
                df_prompt_analysis[feat], df_prompt_analysis["recall_mean"]
            )
            rows.append(
                {
                    "feature_group": group_name,
                    "feature": feat,
                    "pearson_f1": pearson_f1,
                    "pearson_f1_p": pearson_f1_p,
                    "spearman_f1": spearman_f1,
                    "spearman_f1_p": spearman_f1_p,
                    "pearson_recall": pearson_recall,
                    "pearson_recall_p": pearson_recall_p,
                    "spearman_recall": spearman_recall,
                    "spearman_recall_p": spearman_recall_p,
                }
            )

    df_corr = pd.DataFrame(rows)
    if len(df_corr) > 0:
        df_corr["abs_pearson_f1"] = df_corr["pearson_f1"].abs()
        df_corr["abs_pearson_recall"] = df_corr["pearson_recall"].abs()
        df_corr = df_corr.sort_values(["abs_pearson_f1", "abs_pearson_recall", "feature_group"], ascending=[False, False, True])
    return df_corr


def _build_effect_table(df_prompt_analysis, feature_list):
    q25 = df_prompt_analysis["f1_mean"].quantile(0.25)
    q75 = df_prompt_analysis["f1_mean"].quantile(0.75)

    df_prompt_analysis = df_prompt_analysis.copy()
    df_prompt_analysis["performance_group"] = np.where(
        df_prompt_analysis["f1_mean"] <= q25,
        "worst_quartile",
        np.where(df_prompt_analysis["f1_mean"] >= q75, "best_quartile", "middle"),
    )

    worst = df_prompt_analysis[df_prompt_analysis["performance_group"] == "worst_quartile"]
    best = df_prompt_analysis[df_prompt_analysis["performance_group"] == "best_quartile"]

    rows = []
    for feat in feature_list:
        if feat not in df_prompt_analysis.columns:
            continue
        worst_vals = worst[feat]
        best_vals = best[feat]
        rows.append(
            {
                "feature": feat,
                "worst_mean": pd.to_numeric(worst_vals, errors="coerce").mean(),
                "best_mean": pd.to_numeric(best_vals, errors="coerce").mean(),
                "diff_worst_minus_best": pd.to_numeric(worst_vals, errors="coerce").mean()
                - pd.to_numeric(best_vals, errors="coerce").mean(),
                "cohens_d": pooled_effect_size(worst_vals, best_vals),
                "mannwhitney_p": safe_mannwhitneyu(worst_vals, best_vals),
            }
        )

    df_effects = pd.DataFrame(rows)
    if len(df_effects) > 0:
        df_effects["abs_cohens_d"] = df_effects["cohens_d"].abs()
        df_effects = df_effects.sort_values("abs_cohens_d", ascending=False)
    return df_prompt_analysis, df_effects


def _build_standardized_feature_frame(df, feature_names):
    frames = []
    active = []
    for feat in feature_names:
        if feat not in df.columns:
            continue
        values = pd.to_numeric(df[feat], errors="coerce")
        if values.nunique(dropna=True) <= 1:
            continue
        std = values.std(ddof=0)
        if not np.isfinite(std) or std < 1e-8:
            continue
        frames.append(((values - values.mean()) / std).rename(feat))
        active.append(feat)

    if not frames:
        return pd.DataFrame(index=df.index), []
    return pd.concat(frames, axis=1), active


def _fit_ols_matrix(X, y, column_names):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim == 1:
        X = X[:, None]
    if len(X) != len(y):
        raise ValueError("Design matrix and target have incompatible lengths.")

    X_aug = np.column_stack([np.ones(len(y), dtype=float), X])
    columns = ["intercept"] + list(column_names)

    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    fitted = X_aug @ beta
    resid = y - fitted

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    rank = int(np.linalg.matrix_rank(X_aug))
    df_resid = max(len(y) - rank, 1)
    mse = sse / df_resid

    xtx_inv = np.linalg.pinv(X_aug.T @ X_aug)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * mse, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stat), df=df_resid)

    term_df = pd.DataFrame(
        {
            "term": columns,
            "coef": beta,
            "std_error": se,
            "t_stat": t_stat,
            "p_value": p_values,
        }
    )

    return {
        "n_obs": int(len(y)),
        "rank": rank,
        "df_resid": int(df_resid),
        "sse": sse,
        "sst": sst,
        "r2": float(1.0 - sse / (sst + 1e-12)),
        "term_table": term_df,
    }


def _compare_nested_models(base_fit, full_fit):
    delta_sse = max(base_fit["sse"] - full_fit["sse"], 0.0)
    df_num = max(full_fit["rank"] - base_fit["rank"], 0)
    df_den = max(full_fit["df_resid"], 1)

    if df_num == 0:
        f_stat = np.nan
        p_value = np.nan
    else:
        numerator = delta_sse / df_num
        denominator = full_fit["sse"] / df_den if full_fit["sse"] > 0 else np.nan
        if not np.isfinite(denominator) or denominator <= 0:
            f_stat = np.nan
            p_value = np.nan
        else:
            f_stat = numerator / denominator
            p_value = stats.f.sf(f_stat, df_num, df_den)

    return {
        "delta_r2": max(full_fit["r2"] - base_fit["r2"], 0.0),
        "partial_r2": delta_sse / (base_fit["sse"] + 1e-12),
        "joint_f": f_stat,
        "joint_p": p_value,
        "df_num": df_num,
        "df_den": df_den,
    }


def _run_stratified_block_permutation(
    base_X,
    block_X,
    y,
    strata,
    n_permutations,
    random_state,
    base_column_names,
    block_column_names,
):
    base_fit = _fit_ols_matrix(base_X, y, base_column_names)
    full_fit = _fit_ols_matrix(
        np.column_stack([base_X, block_X]),
        y,
        list(base_column_names) + list(block_column_names),
    )
    observed = max(full_fit["r2"] - base_fit["r2"], 0.0)

    strata = pd.Series(strata).astype(str).to_numpy()
    unique_strata = np.unique(strata)
    block_X = np.asarray(block_X, dtype=float)
    exchangeable_counts = {stratum: int(np.sum(strata == stratum)) for stratum in unique_strata}
    exchangeable_prompts = int(sum(size for size in exchangeable_counts.values() if size > 1))
    exchangeable_strata = int(sum(size > 1 for size in exchangeable_counts.values()))

    rng = np.random.default_rng(random_state)
    perm_stats = np.zeros(n_permutations, dtype=float)

    if block_X.size == 0 or exchangeable_prompts == 0:
        perm_stats.fill(np.nan)
    else:
        for perm_idx in range(n_permutations):
            permuted_block = block_X.copy()
            for stratum in unique_strata:
                indices = np.where(strata == stratum)[0]
                if len(indices) <= 1:
                    continue
                shuffled = indices.copy()
                rng.shuffle(shuffled)
                permuted_block[indices] = block_X[shuffled]

            perm_fit = _fit_ols_matrix(
                np.column_stack([base_X, permuted_block]),
                y,
                list(base_column_names) + list(block_column_names),
            )
            perm_stats[perm_idx] = max(perm_fit["r2"] - base_fit["r2"], 0.0)

    valid_perm = perm_stats[np.isfinite(perm_stats)]
    if len(valid_perm) == 0:
        p_value = np.nan
        null_mean = np.nan
        null_std = np.nan
    else:
        p_value = (1.0 + np.sum(valid_perm >= observed)) / (len(valid_perm) + 1.0)
        null_mean = float(np.mean(valid_perm))
        null_std = float(np.std(valid_perm, ddof=0))

    summary = {
        "observed_delta_r2": observed,
        "permutation_p": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "n_permutations": int(len(valid_perm)),
        "exchangeable_prompts": exchangeable_prompts,
        "exchangeable_strata": exchangeable_strata,
    }
    return summary, perm_stats


def _run_adjusted_rotation_checks(df_prompt_analysis):
    required_cols = (
        ["prompt_idx", "tomo_name", "f1_mean", "recall_mean"]
        + CAUSAL_ROTATION_ACQUISITION_FEATURES
        + CAUSAL_CONFOUNDER_FEATURES
    )
    available_cols = [col for col in required_cols if col in df_prompt_analysis.columns]
    df_model = df_prompt_analysis[available_cols].copy()
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df_model) < 8:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    rotation_df, active_rotation = _build_standardized_feature_frame(
        df_model, CAUSAL_ROTATION_ACQUISITION_FEATURES
    )
    confounder_df, active_confounders = _build_standardized_feature_frame(
        df_model, CAUSAL_CONFOUNDER_FEATURES
    )
    tomo_dummies = pd.get_dummies(
        df_model["tomo_name"].astype(str),
        prefix="source_tomo",
        drop_first=True,
    ).astype(float)

    base_design = pd.concat([confounder_df, tomo_dummies], axis=1)
    full_design = pd.concat([base_design, rotation_df], axis=1)

    if len(active_rotation) == 0 or full_design.shape[1] == 0:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    summary_rows = []
    term_rows = []
    permutation_rows = []
    permutation_samples = []

    for metric_name, seed_offset in [("f1_mean", 0), ("recall_mean", 1)]:
        y = pd.to_numeric(df_model[metric_name], errors="coerce").to_numpy(dtype=float)
        base_fit = _fit_ols_matrix(base_design.to_numpy(dtype=float), y, base_design.columns.tolist())
        full_fit = _fit_ols_matrix(full_design.to_numpy(dtype=float), y, full_design.columns.tolist())
        nested = _compare_nested_models(base_fit, full_fit)

        perm_summary, perm_stats = _run_stratified_block_permutation(
            base_X=base_design.to_numpy(dtype=float),
            block_X=rotation_df.to_numpy(dtype=float),
            y=y,
            strata=df_model["tomo_name"],
            n_permutations=CAUSAL_PERMUTATION_ITERATIONS,
            random_state=CAUSAL_RANDOM_STATE + seed_offset,
            base_column_names=base_design.columns.tolist(),
            block_column_names=rotation_df.columns.tolist(),
        )

        summary_rows.append(
            {
                "metric": metric_name,
                "n_obs": int(len(df_model)),
                "rotation_feature_count": len(active_rotation),
                "confounder_feature_count": len(active_confounders),
                "source_tomo_dummy_count": int(tomo_dummies.shape[1]),
                "r2_base": base_fit["r2"],
                "r2_full": full_fit["r2"],
                **nested,
            }
        )
        permutation_rows.append({"metric": metric_name, **perm_summary})

        term_df = full_fit["term_table"].copy()
        term_df["metric"] = metric_name
        term_df["term_group"] = "source_tomogram_fe"
        term_df.loc[term_df["term"].isin(active_confounders), "term_group"] = "confounder"
        term_df.loc[term_df["term"].isin(active_rotation), "term_group"] = "rotation_acquisition"
        term_rows.append(term_df)

        permutation_samples.append(
            pd.DataFrame(
                {
                    "metric": metric_name,
                    "perm_idx": np.arange(len(perm_stats), dtype=int),
                    "delta_r2": perm_stats,
                }
            )
        )

    df_model_summary = pd.DataFrame(summary_rows)
    df_term_table = pd.concat(term_rows, ignore_index=True)
    df_permutation_summary = pd.DataFrame(permutation_rows)
    df_permutation_samples = pd.concat(permutation_samples, ignore_index=True)

    keep_groups = {"rotation_acquisition", "confounder"}
    df_display_terms = df_term_table[
        df_term_table["term_group"].isin(keep_groups) & (df_term_table["term"] != "intercept")
    ].copy()
    df_display_terms["abs_coef"] = df_display_terms["coef"].abs()
    df_display_terms = df_display_terms.sort_values(["metric", "term_group", "abs_coef"], ascending=[True, True, False])

    return df_model, df_model_summary, df_display_terms, df_permutation_summary, df_permutation_samples


def _build_c2_consistency_tables(df_prompt_analysis):
    required_cols = ["prompt_idx", "tomo_name", "q1", "q2", "q3", "q4", "f1_mean", "recall_mean"]
    if any(col not in df_prompt_analysis.columns for col in required_cols):
        empty = pd.DataFrame()
        return empty, empty, empty

    df_pairs_source = (
        df_prompt_analysis[required_cols]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("prompt_idx")
        .reset_index(drop=True)
    )
    if len(df_pairs_source) < 2:
        empty = pd.DataFrame()
        return empty, empty, empty

    quats = df_pairs_source[["q1", "q2", "q3", "q4"]].to_numpy(dtype=float)
    rows = []
    for i in range(len(df_pairs_source)):
        for j in range(i + 1, len(df_pairs_source)):
            raw_deg = quaternion_angular_distance_deg(quats[i], quats[j])
            c2_deg = quaternion_angular_distance_c2_deg(quats[i], quats[j])
            rows.append(
                {
                    "prompt_idx_a": int(df_pairs_source.loc[i, "prompt_idx"]),
                    "prompt_idx_b": int(df_pairs_source.loc[j, "prompt_idx"]),
                    "tomo_name_a": df_pairs_source.loc[i, "tomo_name"],
                    "tomo_name_b": df_pairs_source.loc[j, "tomo_name"],
                    "raw_rotation_deg": raw_deg,
                    "c2_rotation_deg": c2_deg,
                    "symmetry_alias_gap_deg": raw_deg - c2_deg,
                    "f1_abs_diff": abs(df_pairs_source.loc[i, "f1_mean"] - df_pairs_source.loc[j, "f1_mean"]),
                    "recall_abs_diff": abs(
                        df_pairs_source.loc[i, "recall_mean"] - df_pairs_source.loc[j, "recall_mean"]
                    ),
                    "same_source_tomo": bool(
                        df_pairs_source.loc[i, "tomo_name"] == df_pairs_source.loc[j, "tomo_name"]
                    ),
                }
            )

    df_pairs = pd.DataFrame(rows)
    df_pairs["near_c2_pair"] = df_pairs["c2_rotation_deg"] <= C2_NEAR_EQUIVALENCE_DEG
    df_pairs["alias_pair"] = df_pairs["near_c2_pair"] & (
        df_pairs["raw_rotation_deg"] >= C2_ALIAS_RAW_DEG
    )

    groups = {
        "all_pairs": df_pairs,
        "near_c2_pairs": df_pairs[df_pairs["near_c2_pair"]],
        "near_c2_alias_pairs": df_pairs[df_pairs["alias_pair"]],
        "near_c2_non_alias_pairs": df_pairs[df_pairs["near_c2_pair"] & ~df_pairs["alias_pair"]],
    }

    summary_rows = []
    for group_name, group_df in groups.items():
        summary_rows.append(
            {
                "pair_group": group_name,
                "pair_count": int(len(group_df)),
                "mean_f1_abs_diff": pd.to_numeric(group_df["f1_abs_diff"], errors="coerce").mean(),
                "median_f1_abs_diff": pd.to_numeric(group_df["f1_abs_diff"], errors="coerce").median(),
                "mean_recall_abs_diff": pd.to_numeric(group_df["recall_abs_diff"], errors="coerce").mean(),
                "median_recall_abs_diff": pd.to_numeric(group_df["recall_abs_diff"], errors="coerce").median(),
                "mean_alias_gap_deg": pd.to_numeric(
                    group_df["symmetry_alias_gap_deg"], errors="coerce"
                ).mean(),
                "same_tomo_fraction": pd.to_numeric(group_df["same_source_tomo"], errors="coerce").mean(),
            }
        )

    near_pairs = groups["near_c2_pairs"]
    alias_pairs = groups["near_c2_alias_pairs"]
    non_alias_pairs = groups["near_c2_non_alias_pairs"]
    diagnostics = pd.DataFrame(
        [
            {
                "near_c2_f1_gap_spearman": safe_spearmanr(
                    near_pairs["c2_rotation_deg"], near_pairs["f1_abs_diff"]
                )[0]
                if len(near_pairs) >= 3
                else np.nan,
                "near_c2_f1_gap_spearman_p": safe_spearmanr(
                    near_pairs["c2_rotation_deg"], near_pairs["f1_abs_diff"]
                )[1]
                if len(near_pairs) >= 3
                else np.nan,
                "alias_vs_nonalias_f1_gap_p": safe_mannwhitneyu(
                    alias_pairs["f1_abs_diff"], non_alias_pairs["f1_abs_diff"]
                ),
                "alias_vs_nonalias_recall_gap_p": safe_mannwhitneyu(
                    alias_pairs["recall_abs_diff"], non_alias_pairs["recall_abs_diff"]
                ),
            }
        ]
    )

    df_top_pairs = df_pairs[df_pairs["near_c2_pair"]].copy()
    df_top_pairs["sort_key"] = (
        df_top_pairs["f1_abs_diff"].fillna(-np.inf)
        + 0.25 * df_top_pairs["symmetry_alias_gap_deg"].fillna(0.0) / 180.0
    )
    df_top_pairs = df_top_pairs.sort_values("sort_key", ascending=False).drop(columns="sort_key")

    return pd.DataFrame(summary_rows), diagnostics, df_top_pairs


def _plot_causal_rotation_checks(df_permutation_samples, df_permutation_summary, df_c2_pairs, analysis_dir):
    has_perm = len(df_permutation_samples) > 0 and df_permutation_samples["delta_r2"].notna().any()
    has_pairs = len(df_c2_pairs) > 0
    if not has_perm and not has_pairs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, metric_name in zip(axes[:2], ["f1_mean", "recall_mean"]):
        metric_df = df_permutation_samples[
            (df_permutation_samples["metric"] == metric_name) & df_permutation_samples["delta_r2"].notna()
        ].copy()
        if len(metric_df) == 0:
            axis.set_visible(False)
            continue
        sns.histplot(metric_df["delta_r2"], bins=30, color="steelblue", ax=axis)
        observed_row = df_permutation_summary[df_permutation_summary["metric"] == metric_name]
        if len(observed_row) > 0:
            observed = observed_row.iloc[0]["observed_delta_r2"]
            axis.axvline(observed, color="crimson", linewidth=2, linestyle="--", label="Observed")
            axis.legend()
        axis.set_title(f"Permutation null for {metric_name}")
        axis.set_xlabel("Rotation-block delta R^2")
        axis.set_ylabel("Count")

    if has_pairs:
        scatter_df = df_c2_pairs[df_c2_pairs["near_c2_pair"]].copy()
        if len(scatter_df) == 0:
            axes[2].set_visible(False)
        else:
            scatter = axes[2].scatter(
                scatter_df["c2_rotation_deg"],
                scatter_df["f1_abs_diff"],
                c=scatter_df["symmetry_alias_gap_deg"],
                cmap="viridis",
                s=60,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.3,
            )
            axes[2].set_title("Near-C2 pairs: F1 gap vs C2 distance")
            axes[2].set_xlabel("SO(3)/C2 distance (deg)")
            axes[2].set_ylabel("|Delta F1|")
            plt.colorbar(scatter, ax=axes[2], label="Raw-vs-C2 alias gap (deg)")
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    fig.savefig(analysis_dir / "rotational_issues_causal_checks.png", dpi=150, bbox_inches="tight")
    plt.show()


def _summarize_causal_findings(df_model_summary, df_permutation_summary, df_c2_summary, df_c2_diagnostics):
    if len(df_model_summary) == 0:
        return

    print("\nAdjusted rotation/acquisition block:")
    for metric_name in ["f1_mean", "recall_mean"]:
        model_row = df_model_summary[df_model_summary["metric"] == metric_name]
        perm_row = df_permutation_summary[df_permutation_summary["metric"] == metric_name]
        if len(model_row) == 0 or len(perm_row) == 0:
            continue
        model_row = model_row.iloc[0]
        perm_row = perm_row.iloc[0]
        print(
            f"  - {metric_name}: delta_R2={model_row['delta_r2']:.4f}, "
            f"partial_R2={model_row['partial_r2']:.4f}, "
            f"joint_p={model_row['joint_p']:.4f}, "
            f"perm_p={perm_row['permutation_p']:.4f}"
        )
        if np.isfinite(model_row["joint_p"]) and np.isfinite(perm_row["permutation_p"]):
            if model_row["joint_p"] < 0.05 and perm_row["permutation_p"] < 0.05:
                print(
                    "    The rotation + acquisition block survives confounder adjustment and tomogram-stratified permutation."
                )
            else:
                print(
                    "    The rotation + acquisition block does not remain strong after adjustment or permutation control."
                )

    if len(df_c2_summary) == 0:
        return

    near_row = df_c2_summary[df_c2_summary["pair_group"] == "near_c2_pairs"]
    alias_row = df_c2_summary[df_c2_summary["pair_group"] == "near_c2_alias_pairs"]
    non_alias_row = df_c2_summary[df_c2_summary["pair_group"] == "near_c2_non_alias_pairs"]

    print("\nC2 consistency check:")
    if len(near_row) > 0:
        near_row = near_row.iloc[0]
        print(
            f"  - Near-C2 pairs (<= {C2_NEAR_EQUIVALENCE_DEG:.0f} deg): "
            f"n={int(near_row['pair_count'])}, mean |Delta F1|={near_row['mean_f1_abs_diff']:.4f}, "
            f"mean |Delta recall|={near_row['mean_recall_abs_diff']:.4f}"
        )
    if len(alias_row) > 0 and len(non_alias_row) > 0:
        alias_row = alias_row.iloc[0]
        non_alias_row = non_alias_row.iloc[0]
        print(
            f"  - Alias-like near-C2 pairs (raw >= {C2_ALIAS_RAW_DEG:.0f} deg): "
            f"mean |Delta F1|={alias_row['mean_f1_abs_diff']:.4f} vs "
            f"{non_alias_row['mean_f1_abs_diff']:.4f} for non-alias near-C2 pairs"
        )
    if len(df_c2_diagnostics) > 0:
        diag_row = df_c2_diagnostics.iloc[0]
        print(
            "    Diagnostic p-values: "
            f"near-C2 Spearman p={diag_row['near_c2_f1_gap_spearman_p']:.4f}, "
            f"alias-vs-nonalias F1 p={diag_row['alias_vs_nonalias_f1_gap_p']:.4f}"
        )


def _run_pca_and_clustering(df_prompt_analysis):
    cluster_features = [
        "c2_axis_to_global_z_deg",
        "c2_rotation_nn_deg",
        "symmetry_alias_gap_deg",
        "source_snr",
        "z_center_offset_abs",
        "dist_to_center_norm",
        "quality_score",
        "missing_wedge_anisotropy",
        "freq_ratio_mid_high",
        "gradient_energy",
        "inertia_anisotropy",
        "mass_center_shift",
        "emb_dist_to_centroid",
    ]
    cluster_features = [feat for feat in cluster_features if feat in df_prompt_analysis.columns]

    df_model = df_prompt_analysis[["prompt_idx", "recall_mean", "f1_mean"] + cluster_features].copy()
    valid_mask = df_model[cluster_features].notna().all(axis=1)
    df_model_valid = df_model[valid_mask].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_model_valid[cluster_features])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df_model_valid["feature_pc1"] = coords[:, 0]
    df_model_valid["feature_pc2"] = coords[:, 1]

    if len(df_model_valid) >= 6:
        candidate_ks = list(range(2, min(6, len(df_model_valid) - 1) + 1))
        silhouette_by_k = {}
        for k in candidate_ks:
            labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
            silhouette_by_k[k] = silhouette_score(X, labels)
        best_k = max(silhouette_by_k, key=silhouette_by_k.get)
    else:
        silhouette_by_k = {}
        best_k = 2 if len(df_model_valid) >= 2 else 1

    if best_k > 1:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        df_model_valid["cluster_id"] = kmeans.fit_predict(X)
    else:
        df_model_valid["cluster_id"] = 0

    cluster_summary_cols = [
        "f1_mean",
        "recall_mean",
        "c2_axis_to_global_z_deg",
        "c2_rotation_nn_deg",
        "symmetry_alias_gap_deg",
        "source_snr",
        "quality_score",
        "missing_wedge_anisotropy",
        "gradient_energy",
        "mass_center_shift",
        "inertia_anisotropy",
        "z_center_offset_abs",
        "emb_dist_to_centroid",
    ]
    cluster_summary_cols = [c for c in cluster_summary_cols if c in df_model_valid.columns]
    cluster_summary = (
        df_model_valid.groupby("cluster_id")[cluster_summary_cols]
        .mean()
        .assign(prompt_count=df_model_valid.groupby("cluster_id").size())
        .reset_index()
    )

    return df_model_valid, cluster_summary, cluster_features, pca, silhouette_by_k


def _plot_overview(df_prompt_analysis, df_corr, analysis_dir):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    scatter = axes[0, 0].scatter(
        df_prompt_analysis["quality_score"],
        df_prompt_analysis["f1_mean"],
        c=df_prompt_analysis["source_snr"],
        cmap="plasma",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[0, 0].set_xlabel("Prompt quality score")
    axes[0, 0].set_ylabel("Mean F1")
    axes[0, 0].set_title("Prompt quality vs F1")
    plt.colorbar(scatter, ax=axes[0, 0], label="Prompt source SNR")

    # Keep only the last four entries from the displayed top-F1 correlation ranking.
    top_corr = df_corr.head(10).tail(4).copy()
    sns.barplot(
        data=top_corr,
        x="pearson_f1",
        y="feature",
        hue="feature_group",
        dodge=False,
        ax=axes[0, 1],
    )
    axes[0, 1].axvline(0.0, color="black", linewidth=1)
    axes[0, 1].set_title("Top F1 correlations (last 4 ranked variables)")
    axes[0, 1].set_xlabel("Pearson r with mean F1")
    axes[0, 1].set_ylabel("")

    structure_candidates = ["freq_ratio", "freq_ratio_mid_high"]
    centering_candidates = ["mass_center_shift", "centre_ratio"]
    structure_col = next((col for col in structure_candidates if col in df_prompt_analysis.columns), None)
    centering_col = next((col for col in centering_candidates if col in df_prompt_analysis.columns), None)

    pca_required_cols = ["prompt_idx", "f1_mean", "source_snr", "contrast", "q1", "q2", "q3", "q4"]
    if structure_col is not None:
        pca_required_cols.append(structure_col)
    if centering_col is not None:
        pca_required_cols.append(centering_col)

    pca_df = (
        df_prompt_analysis[pca_required_cols]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(pca_df) >= 2 and structure_col is not None and centering_col is not None:
        quat_array = _canonicalize_quaternion_array(pca_df[["q1", "q2", "q3", "q4"]].to_numpy(dtype=float))
        pca_features = pd.DataFrame(
            {
                "source_snr": pd.to_numeric(pca_df["source_snr"], errors="coerce").to_numpy(dtype=float),
                "contrast": pd.to_numeric(pca_df["contrast"], errors="coerce").to_numpy(dtype=float),
                "structure_proxy": pd.to_numeric(pca_df[structure_col], errors="coerce").to_numpy(dtype=float),
                "centering_proxy": pd.to_numeric(pca_df[centering_col], errors="coerce").to_numpy(dtype=float),
                "quat_q1": quat_array[:, 0],
                "quat_q2": quat_array[:, 1],
                "quat_q3": quat_array[:, 2],
                "quat_q4": quat_array[:, 3],
            },
            index=pca_df.index,
        )

        scaled_features = StandardScaler().fit_transform(pca_features)
        full_pca = PCA()
        full_pca.fit(scaled_features)

        targeted_pca = PCA(n_components=2)
        pca_coords = targeted_pca.fit_transform(scaled_features)

        pca_scatter = axes[1, 0].scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=pca_df["f1_mean"],
            cmap="viridis",
            s=80,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.3,
        )
        explained_var = targeted_pca.explained_variance_ratio_ * 100.0
        axes[1, 0].set_xlabel(f"PC1 ({explained_var[0]:.1f}% var)")
        axes[1, 0].set_ylabel(f"PC2 ({explained_var[1]:.1f}% var)")
        axes[1, 0].set_title("PCA: SNR, contrast, structure, centering, quaternions")
        axes[1, 0].text(
            0.02,
            0.02,
            f"structure={structure_col}, centering={centering_col}",
            transform=axes[1, 0].transAxes,
            fontsize=9,
            va="bottom",
        )
        plt.colorbar(pca_scatter, ax=axes[1, 0], label="Mean F1")

        full_explained = full_pca.explained_variance_ratio_ * 100.0
        component_idx = np.arange(1, len(full_explained) + 1)
        cumulative_explained = np.cumsum(full_explained)
        axes[1, 1].bar(component_idx, full_explained, color="steelblue", alpha=0.85)
        axes[1, 1].plot(component_idx, cumulative_explained, color="crimson", marker="o", linewidth=2)
        axes[1, 1].set_xticks(component_idx)
        axes[1, 1].set_xlabel("Principal component")
        axes[1, 1].set_ylabel("Explained variance (%)")
        axes[1, 1].set_title("PCA scree plot")
        axes[1, 1].set_ylim(0, max(100, cumulative_explained.max() * 1.05))
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    fig.tight_layout()
    fig.savefig(analysis_dir / "rotational_issues_overview.png", dpi=150, bbox_inches="tight")
    plt.show()


def _plot_pca_clusters(df_clustered, cluster_summary, cluster_features, spectra_by_prompt, df_prompt_analysis, analysis_dir):
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    scatter = axes[0, 0].scatter(
        df_clustered["feature_pc1"],
        df_clustered["feature_pc2"],
        c=df_clustered["f1_mean"],
        cmap="viridis",
        s=90,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    for _, row in df_clustered.nsmallest(8, "f1_mean").iterrows():
        axes[0, 0].annotate(
            f"P{int(row['prompt_idx'])}",
            (row["feature_pc1"], row["feature_pc2"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
        )
    axes[0, 0].set_title("PCA of theory-guided prompt descriptors")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    plt.colorbar(scatter, ax=axes[0, 0], label="Mean F1")

    sns.scatterplot(
        data=df_clustered,
        x="feature_pc1",
        y="feature_pc2",
        hue="cluster_id",
        palette="tab10",
        s=90,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("KMeans clusters on interpretable prompt features")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

    heat_cols = [
        "c2_axis_to_global_z_deg",
        "c2_rotation_nn_deg",
        "symmetry_alias_gap_deg",
        "f1_mean",
        "recall_mean",
        "source_snr",
        "quality_score",
        "missing_wedge_anisotropy",
        "gradient_energy",
        "mass_center_shift",
    ]
    heat_cols = [c for c in heat_cols if c in cluster_summary.columns]
    heat_df = cluster_summary.set_index("cluster_id")[heat_cols].copy()
    heat_df = (heat_df - heat_df.mean()) / heat_df.std(ddof=0).replace(0, np.nan)
    sns.heatmap(heat_df, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", ax=axes[1, 0])
    axes[1, 0].set_title("Cluster profile z-scores")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Cluster")

    df_prompt_analysis = df_prompt_analysis.copy()
    q25 = df_prompt_analysis["f1_mean"].quantile(0.25)
    q75 = df_prompt_analysis["f1_mean"].quantile(0.75)
    worst_ids = df_prompt_analysis[df_prompt_analysis["f1_mean"] <= q25]["prompt_idx"].tolist()
    best_ids = df_prompt_analysis[df_prompt_analysis["f1_mean"] >= q75]["prompt_idx"].tolist()

    if worst_ids:
        worst_spectrum = np.vstack([spectra_by_prompt[idx] for idx in worst_ids]).mean(axis=0)
        axes[1, 1].semilogy(worst_spectrum, color="crimson", label="Worst quartile")
    if best_ids:
        best_spectrum = np.vstack([spectra_by_prompt[idx] for idx in best_ids]).mean(axis=0)
        axes[1, 1].semilogy(best_spectrum, color="darkgreen", label="Best quartile")
    axes[1, 1].set_title("Average radial power spectrum")
    axes[1, 1].set_xlabel("Radial frequency bin")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(analysis_dir / "rotational_issues_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.show()


def _summarize_findings(df_prompt_analysis, df_corr, cluster_summary, checkpoint_type, increment):
    print("=" * 90)
    print("CRYOET THEORY-GUIDED INTERPRETATION")
    print("=" * 90)
    print(f"Study subset: {len(df_prompt_analysis)} prompts (prompt_idx 0..{len(df_prompt_analysis) - 1})")
    print(f"Focus checkpoint: {checkpoint_type}_inc{increment}")
    print("Main failure signal: mean F1 across the 5 validation tomograms")
    print("Secondary safeguard metric: mean recall, so false-negative sensitivity remains visible")
    print("\nAssumptions used in this study:")
    print("  - Thyroglobulin is treated as a C2-symmetric particle.")
    print("  - The prompt local Z axis is used as a proxy for the thyroglobulin C2 axis.")
    print("  - Global Z is used as the acquisition anisotropy axis because tilt-axis metadata is not stored here.")
    print("  - missing_wedge_anisotropy is a post-reconstruction Fourier proxy, not the exact acquisition operator.")

    def _print_top(group_name):
        subset = df_corr[df_corr["feature_group"] == group_name]
        if len(subset) == 0:
            return
        row = subset.iloc[0]
        print(
            f"  - Strongest {group_name:<10} signal: {row['feature']:<28} "
            f"f1_r={row['pearson_f1']:+.3f} (p={row['pearson_f1_p']:.4f}), "
            f"recall_r={row['pearson_recall']:+.3f}"
        )

    print("\nTop signals by family:")
    for group_name in ["symmetry", "acquisition", "quality", "position", "embedding", "response"]:
        _print_top(group_name)

    symmetry_f1_corr, _ = safe_pearsonr(df_prompt_analysis["c2_axis_to_global_z_deg"], df_prompt_analysis["f1_mean"])
    symmetry_recall_corr, _ = safe_pearsonr(df_prompt_analysis["c2_axis_to_global_z_deg"], df_prompt_analysis["recall_mean"])
    alias_f1_corr, _ = safe_pearsonr(df_prompt_analysis["symmetry_alias_gap_deg"], df_prompt_analysis["f1_mean"])
    alias_recall_corr, _ = safe_pearsonr(df_prompt_analysis["symmetry_alias_gap_deg"], df_prompt_analysis["recall_mean"])
    snr_f1_corr, _ = safe_pearsonr(df_prompt_analysis["source_snr"], df_prompt_analysis["f1_mean"])
    snr_recall_corr, _ = safe_pearsonr(df_prompt_analysis["source_snr"], df_prompt_analysis["recall_mean"])
    z_f1_corr, _ = safe_pearsonr(df_prompt_analysis["z_center_offset_abs"], df_prompt_analysis["f1_mean"])
    z_recall_corr, _ = safe_pearsonr(df_prompt_analysis["z_center_offset_abs"], df_prompt_analysis["recall_mean"])
    wedge_f1_corr, _ = safe_pearsonr(df_prompt_analysis["missing_wedge_anisotropy"], df_prompt_analysis["f1_mean"])
    wedge_recall_corr, _ = safe_pearsonr(df_prompt_analysis["missing_wedge_anisotropy"], df_prompt_analysis["recall_mean"])

    print("\nHypothesis check:")
    if np.isfinite(symmetry_f1_corr):
        if symmetry_f1_corr > 0:
            print(
                "  - C2 axis vs Z: prompts whose thyroglobulin symmetry axis is farther from global Z tend "
                "to have higher F1, which is compatible with anisotropic acquisition around the poorly "
                "sampled direction."
            )
        else:
            print(
                "  - C2 axis vs Z: prompts whose symmetry axis is closer to global Z tend to have higher F1, "
                "so a simple symmetry-axis / missing-wedge story is not sufficient on its own."
            )
        print(
            f"    Secondary recall check: recall_r={symmetry_recall_corr:+.3f}"
        )
    if np.isfinite(alias_f1_corr):
        direction = "worse" if alias_f1_corr < 0 else "better"
        print(
            "  - Symmetry-collapse gap: "
            f"f1_r={alias_f1_corr:+.3f}, recall_r={alias_recall_corr:+.3f}. Large raw-vs-C2 nearest-neighbour "
            f"gaps indicate prompts that look diverse in SO(3) but become close after quotienting by C2. "
            f"Those prompts tend to perform {direction}."
        )
    if np.isfinite(snr_f1_corr):
        print(
            "  - Source SNR: "
            f"f1_r={snr_f1_corr:+.3f}, recall_r={snr_recall_corr:+.3f}. This tests whether prompt failures "
            "are partly explained by the quality of the tomogram used to extract the prompt."
        )
    if np.isfinite(wedge_f1_corr):
        print(
            "  - Fourier anisotropy proxy: "
            f"f1_r={wedge_f1_corr:+.3f}, recall_r={wedge_recall_corr:+.3f}. Large anisotropy is a practical "
            "proxy for wedge / tilt artefacts."
        )
    if np.isfinite(z_f1_corr):
        print(
            "  - Z-position: "
            f"f1_r={z_f1_corr:+.3f}, recall_r={z_recall_corr:+.3f}. This checks whether prompts farther from "
            "the tomogram center along Z behave worse, which would be consistent with depth / edge / "
            "thickness artefacts."
        )

    print("\nWorst prompts by mean F1:")
    worst_cols = [
        "prompt_idx",
        "tomo_name",
        "source_snr",
        "f1_mean",
        "recall_mean",
        "c2_axis_to_global_z_deg",
        "c2_rotation_nn_deg",
        "symmetry_alias_gap_deg",
        "quality_score",
        "missing_wedge_anisotropy",
        "z_center_offset_abs",
        "mass_center_shift",
    ]
    worst_cols = [c for c in worst_cols if c in df_prompt_analysis.columns]
    display(df_prompt_analysis.nsmallest(12, "f1_mean")[worst_cols].round(4))

    print("\nCluster summary:")
    display(cluster_summary.round(4))


def run_rotational_issues_analysis(
    df_selected,
    prompt_info,
    subtomos,
    embeddings,
    df_results,
    study_num_prompts,
    results_dir,
    tomo_dir,
    checkpoint_type="multi",
    increment=16,
    prompt_size=37,
):
    del prompt_size  # prompt size is currently implicit in the extracted subtomograms

    results_dir = Path(results_dir)
    analysis_dir = results_dir / f"rotational_issues_analysis_{study_num_prompts}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df_prompts = _prepare_prompt_dataframe(df_selected, prompt_info, study_num_prompts)
    df_focus, checkpoint_type, increment = _select_focus_results(
        df_results=df_results,
        study_num_prompts=study_num_prompts,
        checkpoint_type=checkpoint_type,
        increment=increment,
    )
    if len(df_focus) == 0:
        raise ValueError("No inference rows available for the requested study subset.")

    df_prompt_perf = _build_prompt_performance(df_focus)
    tomo_shapes = _load_tomo_shapes(tomo_dir=tomo_dir, tomo_names=df_prompts["tomo_name"].tolist())
    df_subtomo_features, spectra_by_prompt = _build_subtomo_feature_table(
        df_prompts=df_prompts,
        subtomos=subtomos,
        embeddings=embeddings,
        tomo_shapes=tomo_shapes,
        study_num_prompts=study_num_prompts,
    )

    df_prompt_analysis = df_prompts.merge(df_prompt_perf, on="prompt_idx", how="left")
    df_prompt_analysis = df_prompt_analysis.merge(df_subtomo_features, on="prompt_idx", how="left")

    df_corr = _build_correlation_table(df_prompt_analysis)
    effect_features = [
        "c2_axis_to_global_z_deg",
        "c2_rotation_nn_deg",
        "symmetry_alias_gap_deg",
        "source_snr",
        "quality_score",
        "missing_wedge_anisotropy",
        "freq_ratio_mid_high",
        "gradient_energy",
        "mass_center_shift",
        "inertia_anisotropy",
        "dist_to_center_norm",
        "z_center_offset_abs",
        "emb_dist_to_centroid",
    ]
    df_prompt_analysis, df_effects = _build_effect_table(df_prompt_analysis, effect_features)
    df_clustered, cluster_summary, cluster_features, pca_model, silhouette_by_k = _run_pca_and_clustering(
        df_prompt_analysis
    )

    df_prompt_analysis = df_prompt_analysis.merge(
        df_clustered[["prompt_idx", "feature_pc1", "feature_pc2", "cluster_id"]],
        on="prompt_idx",
        how="left",
    )
    (
        df_causal_model,
        df_causal_summary,
        df_causal_terms,
        df_permutation_summary,
        df_permutation_samples,
    ) = _run_adjusted_rotation_checks(df_prompt_analysis)
    df_c2_summary, df_c2_diagnostics, df_c2_top_pairs = _build_c2_consistency_tables(df_prompt_analysis)

    _plot_overview(df_prompt_analysis, df_corr, analysis_dir)

    df_prompt_analysis.to_csv(analysis_dir / "prompt_rotational_issue_features.csv", index=False)
    df_corr.to_csv(analysis_dir / "feature_correlations.csv", index=False)
    df_effects.to_csv(analysis_dir / "quartile_effect_sizes.csv", index=False)
    cluster_summary.to_csv(analysis_dir / "cluster_summary.csv", index=False)
    if len(df_causal_model) > 0:
        df_causal_model.to_csv(analysis_dir / "causal_rotation_model_dataset.csv", index=False)
    if len(df_causal_summary) > 0:
        df_causal_summary.to_csv(analysis_dir / "causal_rotation_model_summary.csv", index=False)
    if len(df_causal_terms) > 0:
        df_causal_terms.to_csv(analysis_dir / "causal_rotation_model_terms.csv", index=False)
    if len(df_permutation_summary) > 0:
        df_permutation_summary.to_csv(analysis_dir / "causal_rotation_permutation_summary.csv", index=False)
    if len(df_permutation_samples) > 0:
        df_permutation_samples.to_csv(analysis_dir / "causal_rotation_permutation_samples.csv", index=False)
    if len(df_c2_summary) > 0:
        df_c2_summary.to_csv(analysis_dir / "c2_consistency_summary.csv", index=False)
    if len(df_c2_diagnostics) > 0:
        df_c2_diagnostics.to_csv(analysis_dir / "c2_consistency_diagnostics.csv", index=False)
    if len(df_c2_top_pairs) > 0:
        df_c2_top_pairs.to_csv(analysis_dir / "c2_consistency_top_pairs.csv", index=False)

    return {
        "analysis_dir": analysis_dir,
        "df_prompt_analysis": df_prompt_analysis,
        "df_correlations": df_corr,
        "df_effects": df_effects,
        "df_clustered": df_clustered,
        "cluster_summary": cluster_summary,
        "df_causal_model": df_causal_model,
        "df_causal_summary": df_causal_summary,
        "df_causal_terms": df_causal_terms,
        "df_permutation_summary": df_permutation_summary,
        "df_permutation_samples": df_permutation_samples,
        "df_c2_summary": df_c2_summary,
        "df_c2_diagnostics": df_c2_diagnostics,
        "df_c2_top_pairs": df_c2_top_pairs,
        "pca_model": pca_model,
        "spectra_by_prompt": spectra_by_prompt,
    }
