#!/usr/bin/env python3
"""
Inference script for ProPicker Rotation Invariance Analysis (EXP4).

This script runs inference using prompts with different rotations to test
if ProPicker is rotation invariant. It evaluates:
1. Base model (no fine-tuning)
2. Fine-tuned models with single-instance prompts (from EXP3)
3. Fine-tuned models with multi-instance prompts (from EXP3)

Usage:
    cd /path/to/cryoet-particle-picking/tools/ProPicker
    conda activate deepetpicker
    
    # Run inference for all checkpoints with all prompts:
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference.py
    
    # Run inference for a specific checkpoint type:
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference.py --checkpoint-type base
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference.py --checkpoint-type single --increment 8
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference.py --checkpoint-type multi --increment 8
    
    # Run inference with a specific prompt:
    python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference.py --prompt-idx 0
"""

import sys
import os
import shutil
import copy
import glob
import argparse
import importlib.util
import json

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

# Now import project modules
from utils.mrctools import load_mrc_data, save_mrc_data
from paths import (
    PROPICKER_MODEL_FILE,
    UMU_SYNTH_TOMOS_DIR,
    EXP3_RESULTS_DIR,
    EXP3_CHECKPOINTS_DIR,
    EXP4_RESULTS_DIR,
    EXP4_PROMPTS_DIR,
    EXP4_INFERENCE_DIR,
)
from experiments.config import (
    EXP4_VAL_TOMOS,
    EXP4_NUM_PROMPTS,
    EXP4_BATCH_SIZE,
    EXP4_GPU_ID,
    EXP4_CHECKPOINTS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Validation tomograms to test
test_tomos = EXP4_VAL_TOMOS

# Inference parameters
gpu = EXP4_GPU_ID
batch_size = EXP4_BATCH_SIZE


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_checkpoint_path(checkpoint_type, increment=None):
    """
    Get the path to a checkpoint based on type and increment.
    
    Args:
        checkpoint_type: 'base', 'single', or 'multi'
        increment: Training increment (required for 'single' and 'multi')
    
    Returns:
        Path to checkpoint file, or None for base model
    """
    if checkpoint_type == "base":
        return None  # Use PROPICKER_MODEL_FILE directly
    
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
        str(EXP3_CHECKPOINTS_DIR), f"increment_{increment}", "train_config.py")
    if os.path.exists(config_path):
        return config_path
    
    # Try alternative location
    alt_path = os.path.join(
        str(EXP3_RESULTS_DIR), "fine_tuning", f"increment_{increment}", "configs", "train.py")
    if os.path.exists(alt_path):
        return alt_path
    
    return None


def run_inference(checkpoint_type, increment, prompt_idx, prompt_file, force=False):
    """
    Run inference for a specific checkpoint and prompt.
    
    Args:
        checkpoint_type: 'base', 'single', or 'multi'
        increment: Training increment (None for base)
        prompt_idx: Index of the prompt to use
        prompt_file: Path to the prompt embedding JSON file
        force: Re-run inference even if results exist
    
    Returns:
        Path to results directory, or None if failed
    """
    # Build result directory name
    if checkpoint_type == "base":
        result_name = f"base_prompt{prompt_idx}"
        ckpt_file = None
    else:
        result_name = f"{checkpoint_type}_inc{increment}_prompt{prompt_idx}"
        ckpt_file = get_checkpoint_path(checkpoint_type, increment)
        if ckpt_file is None:
            print(f"❌ Checkpoint not found for {checkpoint_type} increment {increment}")
            return None
    
    print("\n" + "=" * 70)
    print(f"INFERENCE: {result_name}")
    print("=" * 70)
    
    # Results directory
    results_output_dir = os.path.join(str(EXP4_RESULTS_DIR), "inference", result_name)
    
    # Check if already done
    coords_check_dir = os.path.join(results_output_dir, "PredictedLabels", "Coords_All")
    if not force and os.path.exists(coords_check_dir) and len(os.listdir(coords_check_dir)) > 0:
        print(f"✅ Results already exist: {results_output_dir}")
        print("   Skipping (use --force to re-run)")
        return results_output_dir
    
    print(f"  Checkpoint type: {checkpoint_type}")
    if ckpt_file:
        print(f"  Checkpoint: {ckpt_file}")
    else:
        print(f"  Using base model: {PROPICKER_MODEL_FILE}")
    print(f"  Prompt file: {prompt_file}")
    print(f"  Output: {results_output_dir}")
    
    # Setup temporary directory
    tmp_dir = os.path.join(str(EXP4_RESULTS_DIR), "tmp", result_name)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(f"{tmp_dir}/raw_data")
    
    # Load and preprocess test tomograms
    print(f"\nPreparing {len(test_tomos)} test tomograms...")
    for tomo_name in test_tomos:
        print(f"  Loading {tomo_name}...")
        tomo_file = os.path.join(str(UMU_SYNTH_TOMOS_DIR), f"{tomo_name}.mrc")
        # Invert contrast (same as training)
        tomo = -1 * load_mrc_data(tomo_file).float()
        save_mrc_data(tomo, f"{tmp_dir}/raw_data/{tomo_name}.mrc")
        del tomo
    
    # Create preprocessing config
    print("\nCreating preprocessing config...")
    cfg_dir = os.path.join(tmp_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    pre_config_file = f"{cfg_dir}/preprocess.py"
    
    lines = [
        "pre_config={",
        f'"dset_name": "exp4_{result_name}",',
        f'"base_path": "{tmp_dir}",',
        f'"tomo_path": "{tmp_dir}/raw_data",',
        f'"tomo_format": ".mrc",',
        f'"norm_type": "standardization",',
        f'"skip_coords": "True",',
        f'"skip_labels": "True",',
        f'"skip_ocp": "True"',
        "}"
    ]
    
    with open(pre_config_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    
    # Run preprocessing
    print("\nRunning preprocessing...")
    os.system(
        f"python ./DeepETPicker_ProPicker/bin/preprocess.py --pre_configs {pre_config_file}")
    
    # Create test config
    print("\nCreating test config...")
    
    if checkpoint_type == "base":
        # For base model, create a complete config with all required parameters
        # (mimicking the structure generated by generate_train_config.py)
        test_configs = {
            "dset_name": f"exp4_{result_name}",
            "base_path": tmp_dir,
            "coord_path": f"{tmp_dir}/coords",
            "coord_format": ".coords",
            "tomo_path": f"{tmp_dir}/raw_data",
            "tomo_format": ".mrc",
            "num_cls": 1,
            "label_type": "gaussian",
            "label_diameter": 26,  # ~260Å / 10Å voxel size
            "ocp_type": "sphere",
            "ocp_diameter": "26",
            "norm_type": "standardization",
            "label_name": "gaussian26",
            "label_path": f"{tmp_dir}/gaussian26",
            "ocp_name": "data_ocp",
            "ocp_path": f"{tmp_dir}/data_ocp",
            "model_name": "ResUNet",
            "train_set_ids": f"0-{len(test_tomos)-1}",
            "val_set_ids": f"0-{len(test_tomos)-1}",
            "batch_size": batch_size,
            "patch_size": 72,
            "padding_size": 12,
            "lr": 0.001,
            "max_epochs": 100,
            "seg_thresh": 0.5,
            "gpu_ids": str(gpu),
            "pre_configs": pre_config_file,
        }
        
        # Create dummy directories that the dataloader expects
        os.makedirs(f"{tmp_dir}/coords", exist_ok=True)
        os.makedirs(f"{tmp_dir}/gaussian26", exist_ok=True)
        os.makedirs(f"{tmp_dir}/data_ocp", exist_ok=True)
    else:
        # Load config from training
        config_path = get_config_path(increment)
        if config_path is None:
            print(f"❌ Training config not found for increment {increment}")
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
        test_configs["dset_name"] = f"exp4_{result_name}"
        test_configs["base_path"] = tmp_dir
        test_configs["tomo_path"] = f"{tmp_dir}/raw_data"
    
    test_cfg_file = f"{cfg_dir}/test.py"
    with open(test_cfg_file, "w") as f:
        f.write("train_configs=")
        f.write(str(test_configs).replace("'", '"'))
    
    # Run inference
    print("\nRunning inference...")
    if ckpt_file:
        cmd = (
            f"python ./DeepETPicker_ProPicker/bin/test_bash.py "
            f"--train_configs {test_cfg_file} "
            f"--checkpoints {ckpt_file} "
            f"--de_duplication True "
            f"--network ProPicker "
            f"--propicker_model_file '{PROPICKER_MODEL_FILE}' "
            f"--prompt_embed_file '{prompt_file}' "
            f"--prompt_class thyroglobulin"
        )
    else:
        # Base model - still need to pass a checkpoint file (use propicker model)
        cmd = (
            f"python ./DeepETPicker_ProPicker/bin/test_bash.py "
            f"--train_configs {test_cfg_file} "
            f"--checkpoints '{PROPICKER_MODEL_FILE}' "
            f"--de_duplication True "
            f"--network ProPicker "
            f"--propicker_model_file '{PROPICKER_MODEL_FILE}' "
            f"--prompt_embed_file '{prompt_file}' "
            f"--prompt_class thyroglobulin"
        )
    
    os.system(cmd)
    
    # Copy results to organized directory
    print(f"\n📁 Copying results to: {results_output_dir}")
    os.makedirs(results_output_dir, exist_ok=True)
    
    # Find where results were saved
    # ProPicker saves results relative to the checkpoint path:
    #   out_dir = '/'.join(args.checkpoints.split('/')[:-2]) + f'/{args.out_name}'
    # So we need to check multiple possible locations
    
    ckpt_path = ckpt_file if ckpt_file else str(PROPICKER_MODEL_FILE)
    ckpt_parent_dir = os.path.dirname(os.path.dirname(ckpt_path))
    
    for src_dir_name in ["PredictedLabels", "full_segmentation_output"]:
        # Check multiple possible locations (in order of priority)
        possible_srcs = [
            os.path.join(ckpt_parent_dir, src_dir_name),  # Relative to checkpoint
            os.path.join(tmp_dir, src_dir_name),
            os.path.join(str(EXP4_RESULTS_DIR), src_dir_name),
            os.path.join(str(EXP3_CHECKPOINTS_DIR), src_dir_name),
            os.path.join(PROJECT_ROOT, src_dir_name),  # Project root (fallback)
        ]
        
        found = False
        for src_dir in possible_srcs:
            if os.path.exists(src_dir):
                dst_dir = os.path.join(results_output_dir, src_dir_name)
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
                shutil.rmtree(src_dir)
                print(f"   ✅ {src_dir_name} copied from {src_dir}")
                found = True
                break
        
        if not found:
            print(f"   ⚠️ {src_dir_name} not found in expected locations")
    
    # Cleanup
    print("\nCleaning up...")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    
    print(f"✅ Inference complete: {result_name}")
    return results_output_dir


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for ProPicker rotation invariance analysis")
    parser.add_argument(
        "--checkpoint-type", type=str, choices=["base", "single", "multi"],
        default=None, help="Checkpoint type to evaluate (default: all)")
    parser.add_argument(
        "--increment", type=int, default=None,
        help="Training increment to evaluate (required for single/multi)")
    parser.add_argument(
        "--prompt-idx", type=int, default=None,
        help="Specific prompt index to use (default: all prompts)")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run inference even if results exist")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ProPicker Rotation Invariance Analysis - Inference")
    print("=" * 70)
    
    # Find prompt files
    prompt_dir = str(EXP4_PROMPTS_DIR)
    prompt_files = sorted(glob.glob(os.path.join(prompt_dir, "prompt_*.json")))
    
    if len(prompt_files) == 0:
        print(f"\n❌ ERROR: No prompt files found in {prompt_dir}")
        print("   Run the notebook to generate rotation-diverse prompts first.")
        sys.exit(1)
    
    print(f"\n📌 Found {len(prompt_files)} prompt files")
    
    # Determine which prompts to use
    if args.prompt_idx is not None:
        if args.prompt_idx >= len(prompt_files):
            print(f"❌ Invalid prompt index {args.prompt_idx}. Max: {len(prompt_files)-1}")
            sys.exit(1)
        prompt_indices = [args.prompt_idx]
    else:
        prompt_indices = list(range(min(len(prompt_files), EXP4_NUM_PROMPTS)))
    
    print(f"📌 Prompts to evaluate: {prompt_indices}")
    
    # Determine checkpoints to evaluate
    checkpoint_configs = []
    
    if args.checkpoint_type is None or args.checkpoint_type == "base":
        checkpoint_configs.append(("base", None))
    
    if args.checkpoint_type is None or args.checkpoint_type == "single":
        increments = [args.increment] if args.increment else EXP4_CHECKPOINTS["increment_single"]
        for inc in increments:
            checkpoint_configs.append(("single", inc))
    
    if args.checkpoint_type is None or args.checkpoint_type == "multi":
        increments = [args.increment] if args.increment else EXP4_CHECKPOINTS["increment_multi"]
        for inc in increments:
            checkpoint_configs.append(("multi", inc))
    
    print(f"📌 Checkpoint configurations: {len(checkpoint_configs)}")
    
    # Run inference
    results = {}
    total_runs = len(checkpoint_configs) * len(prompt_indices)
    current_run = 0
    
    for ckpt_type, increment in checkpoint_configs:
        for prompt_idx in prompt_indices:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}]")
            
            prompt_file = prompt_files[prompt_idx]
            result_dir = run_inference(
                ckpt_type, increment, prompt_idx, prompt_file, force=args.force)
            
            key = f"{ckpt_type}_inc{increment}_p{prompt_idx}"
            results[key] = result_dir
    
    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results.values() if r is not None)
    print(f"\n✅ Successful: {success}/{len(results)}")
    print(f"📁 Results saved to: {EXP4_RESULTS_DIR}/inference/")
    
    print("\nNext step: Analyze results in the notebook")
    print("=" * 70)
