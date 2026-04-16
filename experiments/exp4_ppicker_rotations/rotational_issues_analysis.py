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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    scatter1 = axes[0, 0].scatter(
        df_prompt_analysis["c2_axis_to_global_z_deg"],
        df_prompt_analysis["f1_mean"],
        c=df_prompt_analysis["missing_wedge_anisotropy"],
        cmap="magma",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[0, 0].set_xlabel("Thyroglobulin C2 axis to global Z (deg)")
    axes[0, 0].set_ylabel("Mean F1")
    axes[0, 0].set_title("Symmetry axis orientation vs F1")
    plt.colorbar(scatter1, ax=axes[0, 0], label="Fourier anisotropy proxy")

    scatter2 = axes[0, 1].scatter(
        df_prompt_analysis["missing_wedge_anisotropy"],
        df_prompt_analysis["f1_mean"],
        c=df_prompt_analysis["c2_axis_to_global_z_deg"],
        cmap="viridis",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[0, 1].set_xlabel("Missing-wedge anisotropy proxy")
    axes[0, 1].set_ylabel("Mean F1")
    axes[0, 1].set_title("Acquisition anisotropy proxy vs F1")
    plt.colorbar(scatter2, ax=axes[0, 1], label="C2 axis to global Z (deg)")

    scatter3 = axes[1, 0].scatter(
        df_prompt_analysis["quality_score"],
        df_prompt_analysis["f1_mean"],
        c=df_prompt_analysis["source_snr"],
        cmap="plasma",
        s=80,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[1, 0].set_xlabel("Prompt quality score")
    axes[1, 0].set_ylabel("Mean F1")
    axes[1, 0].set_title("Prompt quality vs F1")
    plt.colorbar(scatter3, ax=axes[1, 0], label="Prompt source SNR")

    heatmap_df = df_prompt_analysis.copy()
    heatmap_df["orientation_bin"] = pd.cut(
        heatmap_df["c2_axis_to_global_z_deg"],
        bins=[0, 15, 30, 45, 60, 75, 90],
        include_lowest=True,
    )
    heatmap_df["snr_bin"] = pd.qcut(
        heatmap_df["source_snr"],
        q=min(3, heatmap_df["source_snr"].nunique()),
        duplicates="drop",
    )
    heatmap = heatmap_df.pivot_table(
        index="snr_bin",
        columns="orientation_bin",
        values="f1_mean",
        aggfunc="mean",
    )
    sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="RdYlGn", ax=axes[1, 1])
    axes[1, 1].set_title("Mean F1 by source SNR and C2-axis bins")
    axes[1, 1].set_xlabel("Thyroglobulin C2 axis to global Z (deg)")
    axes[1, 1].set_ylabel("Prompt source SNR bin")

    fig.suptitle("CryoET prompt failure overview (SO(3)/C2 + anisotropy proxies)", fontsize=16)
    fig.tight_layout()
    fig.savefig(analysis_dir / "rotational_issues_overview.png", dpi=150, bbox_inches="tight")
    plt.show()

    top_corr = df_corr.head(10).copy()
    fig2, axes2 = plt.subplots(2, 2, figsize=(17, 12))

    sns.barplot(
        data=top_corr,
        x="pearson_f1",
        y="feature",
        hue="feature_group",
        dodge=False,
        ax=axes2[0, 0],
    )
    axes2[0, 0].axvline(0.0, color="black", linewidth=1)
    axes2[0, 0].set_title("Top F1 correlations")
    axes2[0, 0].set_xlabel("Pearson r with mean F1")
    axes2[0, 0].set_ylabel("")

    worst = df_prompt_analysis.nsmallest(10, "f1_mean")
    best = df_prompt_analysis.nlargest(10, "f1_mean")
    for _, row in worst.iterrows():
        axes2[0, 1].scatter(
            row["prompt_idx"],
            row["f1_mean"],
            color="crimson",
            s=55,
        )
    for _, row in best.iterrows():
        axes2[0, 1].scatter(
            row["prompt_idx"],
            row["f1_mean"],
            color="darkgreen",
            s=55,
        )
    axes2[0, 1].plot(df_prompt_analysis["prompt_idx"], df_prompt_analysis["f1_mean"], color="steelblue", alpha=0.5)
    axes2[0, 1].set_title("Prompt F1 profile (best and worst highlighted)")
    axes2[0, 1].set_xlabel("Prompt index")
    axes2[0, 1].set_ylabel("Mean F1")

    worst_ids = set(worst["prompt_idx"].tolist())
    best_ids = set(best["prompt_idx"].tolist())
    worst_spectra = [row for idx, row in worst["prompt_idx"].items()]
    best_spectra = [row for idx, row in best["prompt_idx"].items()]
    # The actual spectra are added by the caller in a separate panel.
    axes2[1, 0].set_visible(False)
    axes2[1, 1].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(analysis_dir / "rotational_issues_correlations.png", dpi=150, bbox_inches="tight")
    plt.show()

    return worst_ids, best_ids


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

    print("=" * 90)
    print("ROTATIONAL ISSUES ANALYSIS")
    print("=" * 90)
    print(f"Analysis directory: {analysis_dir}")
    print(f"Prompts analyzed: {len(df_prompt_analysis)}")
    print(f"Focus checkpoint: {checkpoint_type}_inc{increment}")
    print(f"Tomogram shapes loaded: {len(tomo_shapes)}")
    if silhouette_by_k:
        print(f"Silhouette by k: {silhouette_by_k}")
    print(f"Cluster feature set: {cluster_features}")

    print("\nTop F1 correlations (recall kept as secondary):")
    display(
        df_corr[
            [
                "feature_group",
                "feature",
                "pearson_f1",
                "pearson_f1_p",
                "spearman_f1",
                "spearman_f1_p",
                "pearson_recall",
                "pearson_recall_p",
                "spearman_recall",
                "spearman_recall_p",
            ]
        ]
        .head(20)
        .round(4)
    )

    print("\nWorst-vs-best quartile effect sizes (quartiles defined by mean F1):")
    display(df_effects.head(15).round(4))

    _plot_overview(df_prompt_analysis, df_corr, analysis_dir)
    _plot_pca_clusters(df_clustered, cluster_summary, cluster_features, spectra_by_prompt, df_prompt_analysis, analysis_dir)
    _summarize_findings(df_prompt_analysis, df_corr, cluster_summary, checkpoint_type, increment)

    df_prompt_analysis.to_csv(analysis_dir / "prompt_rotational_issue_features.csv", index=False)
    df_corr.to_csv(analysis_dir / "feature_correlations.csv", index=False)
    df_effects.to_csv(analysis_dir / "quartile_effect_sizes.csv", index=False)
    cluster_summary.to_csv(analysis_dir / "cluster_summary.csv", index=False)

    return {
        "analysis_dir": analysis_dir,
        "df_prompt_analysis": df_prompt_analysis,
        "df_correlations": df_corr,
        "df_effects": df_effects,
        "df_clustered": df_clustered,
        "cluster_summary": cluster_summary,
        "pca_model": pca_model,
        "spectra_by_prompt": spectra_by_prompt,
    }
