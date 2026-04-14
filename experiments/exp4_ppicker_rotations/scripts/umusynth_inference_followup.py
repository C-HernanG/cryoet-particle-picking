#!/usr/bin/env python3
"""
Inference script for the EXP4 follow-up run with candidate prompts.

This mirrors the original EXP4 inference flow, but reads prompt embeddings from
the follow-up prompt directory and writes all organized outputs to a dedicated
follow-up results tree so the original EXP4 results are preserved.

Usage:
    cd /path/to/cryoet-particle-picking/tools/ProPicker
    conda activate deepetpicker

    # Run all follow-up inferences:
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference_followup.py

    # Run a specific prompt:
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference_followup.py --prompt-idx 0
"""

import argparse
import copy
import glob
import importlib.util
import os
import shutil
import subprocess
import sys
import time

# Add paths BEFORE any project imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PROPICKER_DIR = os.path.join(PROJECT_ROOT, "tools", "ProPicker")
PROPICKER_INNER_DIR = os.path.join(PROPICKER_DIR, "propicker")

# Add ProPicker tools to path (for utils.mrctools)
sys.path.insert(0, PROPICKER_INNER_DIR)
os.chdir(PROPICKER_INNER_DIR)

# Add project root to path for paths.py
sys.path.insert(0, PROJECT_ROOT)

# Add experiments to path for config import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from utils.mrctools import load_mrc_data, save_mrc_data
from paths import (
    PROPICKER_MODEL_FILE,
    UMU_SYNTH_TOMOS_DIR,
    EXP3_RESULTS_DIR,
    EXP3_CHECKPOINTS_DIR,
    EXP4_FOLLOWUP_RESULTS_DIR,
    EXP4_FOLLOWUP_PROMPTS_DIR,
    EXP4_FOLLOWUP_INFERENCE_DIR,
)
from experiments.config import (
    EXP4_VAL_TOMOS,
    EXP4_BATCH_SIZE,
    EXP4_GPU_ID,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

test_tomos = EXP4_VAL_TOMOS
gpu = EXP4_GPU_ID
batch_size = EXP4_BATCH_SIZE
followup_checkpoint_type = "multi"
followup_increment = 16


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_checkpoint_path(checkpoint_type, increment=None):
    """
    Get the path to a checkpoint based on type and increment.
    """
    if checkpoint_type == "base":
        return None

    if increment is None:
        raise ValueError(f"Increment required for checkpoint type: {checkpoint_type}")

    ckpt_dir = os.path.join(str(EXP3_CHECKPOINTS_DIR), f"increment_{increment}")
    ckpt_file = os.path.join(ckpt_dir, "best_model.ckpt")

    if not os.path.exists(ckpt_file):
        return None

    return ckpt_file


def get_config_path(increment):
    """Get the training config path for a given increment."""
    config_path = os.path.join(
        str(EXP3_CHECKPOINTS_DIR), f"increment_{increment}", "train_config.py"
    )
    if os.path.exists(config_path):
        return config_path

    alt_path = os.path.join(
        str(EXP3_RESULTS_DIR),
        "fine_tuning",
        f"increment_{increment}",
        "configs",
        "train.py",
    )
    if os.path.exists(alt_path):
        return alt_path

    return None


def tree_mtime(path):
    """Return the latest mtime found in a directory tree."""
    latest = os.path.getmtime(path)
    if not os.path.isdir(path):
        return latest

    for root, dirs, files in os.walk(path):
        for name in dirs:
            latest = max(latest, os.path.getmtime(os.path.join(root, name)))
        for name in files:
            latest = max(latest, os.path.getmtime(os.path.join(root, name)))
    return latest


def find_generated_output(src_dir_name, search_roots, min_mtime):
    """
    Find the freshest generated output directory for a given ProPicker artifact.
    """
    candidates = []
    for root in search_roots:
        candidate = os.path.join(root, src_dir_name)
        if os.path.exists(candidate):
            candidates.append((tree_mtime(candidate), candidate))

    if not candidates:
        return None

    fresh = [item for item in candidates if item[0] >= min_mtime]
    pool = fresh if fresh else candidates
    pool.sort(key=lambda item: item[0], reverse=True)
    return pool[0][1]


def cleanup_tmp_dir(tmp_dir):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


def run_inference(prompt_idx, prompt_file, force=False):
    """
    Run follow-up inference for the fixed multi_inc16 checkpoint and one prompt.
    """
    result_name = f"{followup_checkpoint_type}_inc{followup_increment}_prompt{prompt_idx}"
    ckpt_file = get_checkpoint_path(followup_checkpoint_type, followup_increment)
    if ckpt_file is None:
        print(
            "ERROR: Checkpoint not found for "
            f"{followup_checkpoint_type} increment {followup_increment}"
        )
        return None

    print("\n" + "=" * 70)
    print(f"INFERENCE: {result_name}")
    print("=" * 70)

    results_output_dir = os.path.join(str(EXP4_FOLLOWUP_INFERENCE_DIR), result_name)
    coords_check_dir = os.path.join(results_output_dir, "PredictedLabels", "Coords_All")

    if not force and os.path.exists(coords_check_dir) and len(os.listdir(coords_check_dir)) > 0:
        print(f"Results already exist: {results_output_dir}")
        print("Skipping (use --force to re-run)")
        return results_output_dir

    if force and os.path.exists(results_output_dir):
        shutil.rmtree(results_output_dir)

    print(f"  Checkpoint type: {followup_checkpoint_type}")
    print(f"  Increment: {followup_increment}")
    print(f"  Checkpoint: {ckpt_file}")
    print(f"  Prompt file: {prompt_file}")
    print(f"  Output: {results_output_dir}")

    tmp_dir = os.path.join(str(EXP4_FOLLOWUP_RESULTS_DIR), "tmp", result_name)
    cleanup_tmp_dir(tmp_dir)
    os.makedirs(f"{tmp_dir}/raw_data")

    print(f"\nPreparing {len(test_tomos)} test tomograms...")
    for tomo_name in test_tomos:
        print(f"  Loading {tomo_name}...")
        tomo_file = os.path.join(str(UMU_SYNTH_TOMOS_DIR), f"{tomo_name}.mrc")
        tomo = -1 * load_mrc_data(tomo_file).float()
        save_mrc_data(tomo, f"{tmp_dir}/raw_data/{tomo_name}.mrc")
        del tomo

    print("\nCreating preprocessing config...")
    cfg_dir = os.path.join(tmp_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    pre_config_file = f"{cfg_dir}/preprocess.py"

    lines = [
        "pre_config={",
        f'"dset_name": "exp4_followup_{result_name}",',
        f'"base_path": "{tmp_dir}",',
        f'"tomo_path": "{tmp_dir}/raw_data",',
        f'"tomo_format": ".mrc",',
        f'"norm_type": "standardization",',
        f'"skip_coords": "True",',
        f'"skip_labels": "True",',
        f'"skip_ocp": "True"',
        "}",
    ]

    with open(pre_config_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print("\nRunning preprocessing...")
    preprocess_cmd = [
        "python",
        "./DeepETPicker_ProPicker/bin/preprocess.py",
        "--pre_configs",
        pre_config_file,
    ]
    preprocess_rc = subprocess.run(preprocess_cmd, check=False)
    if preprocess_rc.returncode != 0:
        print(f"ERROR: preprocessing failed with exit code {preprocess_rc.returncode}")
        cleanup_tmp_dir(tmp_dir)
        return None

    print("\nCreating test config...")
    config_path = get_config_path(followup_increment)
    if config_path is None:
        print(f"ERROR: training config not found for increment {followup_increment}")
        cleanup_tmp_dir(tmp_dir)
        return None

    module_name = "train_configs_module"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    test_configs = copy.deepcopy(train_module.train_configs)
    test_configs["pre_configs"] = pre_config_file
    test_configs["train_set_ids"] = f"0-{len(test_tomos)-1}"
    test_configs["val_set_ids"] = f"0-{len(test_tomos)-1}"
    test_configs["gpu_ids"] = str(gpu)
    test_configs["batch_size"] = batch_size
    test_configs["dset_name"] = f"exp4_followup_{result_name}"
    test_configs["base_path"] = tmp_dir
    test_configs["tomo_path"] = f"{tmp_dir}/raw_data"

    test_cfg_file = f"{cfg_dir}/test.py"
    with open(test_cfg_file, "w") as f:
        f.write("train_configs=")
        f.write(str(test_configs).replace("'", '"'))

    print("\nRunning inference...")
    inference_cmd = [
        "python",
        "./DeepETPicker_ProPicker/bin/test_bash.py",
        "--train_configs",
        test_cfg_file,
        "--checkpoints",
        ckpt_file,
        "--de_duplication",
        "True",
        "--network",
        "ProPicker",
        "--propicker_model_file",
        str(PROPICKER_MODEL_FILE),
        "--prompt_embed_file",
        str(prompt_file),
        "--prompt_class",
        "thyroglobulin",
    ]

    inference_started_at = time.time()
    inference_rc = subprocess.run(inference_cmd, check=False)

    print(f"\nCopying organized results to: {results_output_dir}")
    os.makedirs(results_output_dir, exist_ok=True)

    ckpt_parent_dir = os.path.dirname(os.path.dirname(ckpt_file))
    search_roots = [
        ckpt_parent_dir,
        tmp_dir,
        str(EXP4_FOLLOWUP_RESULTS_DIR),
    ]

    copied_any = False
    for src_dir_name in ["PredictedLabels", "full_segmentation_output"]:
        src_dir = find_generated_output(src_dir_name, search_roots, inference_started_at)
        if src_dir is None:
            print(f"  WARNING: {src_dir_name} not found in expected locations")
            continue

        dst_dir = os.path.join(results_output_dir, src_dir_name)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        shutil.copytree(src_dir, dst_dir)
        shutil.rmtree(src_dir)
        copied_any = True
        print(f"  Copied {src_dir_name} from {src_dir}")

    print("\nCleaning up...")
    cleanup_tmp_dir(tmp_dir)

    if inference_rc.returncode != 0:
        print(f"WARNING: inference exited with code {inference_rc.returncode}")

    if not copied_any:
        print("ERROR: no inference outputs were captured")
        return None

    if not os.path.exists(coords_check_dir) or len(os.listdir(coords_check_dir)) == 0:
        print(f"ERROR: expected coords were not found in {coords_check_dir}")
        return None

    print(f"Inference complete: {result_name}")
    return results_output_dir


# =============================================================================
# MAIN SCRIPT
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for the EXP4 follow-up candidate prompts"
    )
    parser.add_argument(
        "--prompt-idx",
        type=int,
        default=None,
        help="Specific prompt index to use (default: all prompts)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if results exist",
    )
    args = parser.parse_args()

    for path in [
        EXP4_FOLLOWUP_RESULTS_DIR,
        EXP4_FOLLOWUP_PROMPTS_DIR,
        EXP4_FOLLOWUP_INFERENCE_DIR,
    ]:
        os.makedirs(path, exist_ok=True)

    print("=" * 70)
    print("ProPicker Rotation Invariance Analysis - Follow-up Inference")
    print("=" * 70)
    print(f"Fixed checkpoint: {followup_checkpoint_type}_inc{followup_increment}")

    prompt_dir = str(EXP4_FOLLOWUP_PROMPTS_DIR)
    prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "prompt_*.json")))

    if len(prompt_files) == 0:
        print(f"\nERROR: no prompt files found in {prompt_dir}")
        print("Run the follow-up section of the analysis notebook first.")
        sys.exit(1)

    print(f"\nFound {len(prompt_files)} follow-up prompt files")

    if args.prompt_idx is not None:
        if args.prompt_idx < 0 or args.prompt_idx >= len(prompt_files):
            print(f"ERROR: invalid prompt index {args.prompt_idx}. Max: {len(prompt_files) - 1}")
            sys.exit(1)
        prompt_indices = [args.prompt_idx]
    else:
        prompt_indices = list(range(len(prompt_files)))

    print(f"Prompts to evaluate: {prompt_indices}")

    results = {}
    total_runs = len(prompt_indices)
    current_run = 0

    for prompt_idx in prompt_indices:
        current_run += 1
        print(f"\n[{current_run}/{total_runs}]")

        prompt_file = prompt_files[prompt_idx]
        result_dir = run_inference(
            prompt_idx,
            prompt_file,
            force=args.force,
        )
        key = f"{followup_checkpoint_type}_inc{followup_increment}_p{prompt_idx}"
        results[key] = result_dir

    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    success = sum(1 for result in results.values() if result is not None)
    print(f"\nSuccessful: {success}/{len(results)}")
    print(f"Results saved to: {EXP4_FOLLOWUP_INFERENCE_DIR}")
    print("\nNext step: analyze the follow-up results against the original EXP4 run")
    print("=" * 70)
