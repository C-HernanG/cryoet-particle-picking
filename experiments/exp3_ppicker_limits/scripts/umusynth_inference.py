#!/usr/bin/env python3
"""
Inference script for ProPicker fine-tuned on UMU Synthetic Thyroglobulin dataset.
Adapted for EXP3: evaluates checkpoints from all training increments.

This script runs inference on the validation set using checkpoints from each
incremental training run, enabling analysis of how performance varies with
the amount of training data.

Usage:
    cd /path/to/cryoet-particle-picking/tools/ProPicker
    conda activate deepetpicker
    
    # Run inference on all increments:
    python ../../experiments/exp3_ppicker_limits/scripts/umusynth_inference.py
    
    # Run inference on a specific increment:
    python ../../experiments/exp3_ppicker_limits/scripts/umusynth_inference.py --increment 4
"""

import sys
import os
import shutil
import copy
import glob
import argparse
import importlib.util

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
    EXP3_FINETUNING_DIR,
    EXP3_INFERENCE_DIR,
    EXP3_CHECKPOINTS_DIR,
)
from experiments.config import (
    EXP3_VAL_TOMOS,
    EXP3_INCREMENTS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Validation tomograms to test
test_tomos = EXP3_VAL_TOMOS

# Default prompt embedding file (can be overridden with --prompt-file)
DEFAULT_PROMPT_FILE = os.path.join(
    str(EXP3_RESULTS_DIR), "fixed_prompts_umusynth_thy.json")

# Will be set from command line or default
prompt_embed_file = DEFAULT_PROMPT_FILE

# Inference parameters
gpu = 0
batch_size = 2

# Output suffix for different prompt types (set based on prompt file)
output_suffix = ""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def run_inference_for_increment(n_train_tomos, force=False):
    """
    Run inference using the checkpoint from a specific training increment.
    
    Args:
        n_train_tomos: Number of training tomograms used for this checkpoint
        force: Re-run inference even if results exist
    
    Returns:
        Path to inference results, or None if failed
    """
    print("\n" + "=" * 70)
    print(f"INFERENCE FOR INCREMENT: {n_train_tomos} training tomograms")
    print("=" * 70)
    
    # Paths
    checkpoint_dir = os.path.join(str(EXP3_CHECKPOINTS_DIR), f"increment_{n_train_tomos}")
    ckpt_file = os.path.join(checkpoint_dir, "best_model.ckpt")
    train_cfg_file = os.path.join(checkpoint_dir, "train_config.py")
    
    # Output directory (with optional suffix for different prompt types)
    if output_suffix:
        inference_output_dir = os.path.join(str(EXP3_INFERENCE_DIR), f"increment_{n_train_tomos}_{output_suffix}")
    else:
        inference_output_dir = os.path.join(str(EXP3_INFERENCE_DIR), f"increment_{n_train_tomos}")
    
    # Check if already done
    if not force and os.path.exists(inference_output_dir):
        locmaps_dir = os.path.join(inference_output_dir, "locmaps")
        if os.path.exists(locmaps_dir) and len(os.listdir(locmaps_dir)) > 0:
            print(f"✅ Inference results already exist: {inference_output_dir}")
            print("   Skipping (use --force to re-run)")
            return inference_output_dir
    
    # Check prerequisites
    if not os.path.exists(ckpt_file):
        print(f"❌ Checkpoint not found: {ckpt_file}")
        return None
    
    if not os.path.exists(train_cfg_file):
        # Try to find config in fine-tuning directory
        alt_cfg = os.path.join(str(EXP3_FINETUNING_DIR), f"increment_{n_train_tomos}", "configs", "train.py")
        if os.path.exists(alt_cfg):
            train_cfg_file = alt_cfg
        else:
            print(f"❌ Training config not found: {train_cfg_file}")
            return None
    
    print(f"  Checkpoint: {ckpt_file}")
    print(f"  Config: {train_cfg_file}")
    
    # Setup temporary directory
    tmp_dir = os.path.join(str(EXP3_FINETUNING_DIR), f"increment_{n_train_tomos}", "test")
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
    cfg_dir = os.path.dirname(train_cfg_file)
    pre_config_file = f"{cfg_dir}/preprocess_test.py"
    
    lines = [
        "pre_config={",
        f'"dset_name": "umusynth_test_inc{n_train_tomos}",',
        f'"base_path": "{tmp_dir}",',
        f'"tomo_path": "{tmp_dir}/raw_data",',
        f'"tomo_format": ".mrc",',
        f'"norm_type": "standardization",',
        f'"skip_coords": "True",',
        f'"skip_labels": "True",',
        f'"skip_ocp": "True"',
        "}"
    ]
    
    if os.path.exists(pre_config_file):
        os.remove(pre_config_file)
    with open(pre_config_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    
    # Run preprocessing
    print("\nRunning preprocessing...")
    os.system(
        f"python ./DeepETPicker_ProPicker/bin/preprocess.py --pre_configs {pre_config_file}")
    
    # Create test config
    print("\nCreating test config...")
    module_name = "train_configs_module"
    spec = importlib.util.spec_from_file_location(module_name, train_cfg_file)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    test_configs = copy.deepcopy(train_module.train_configs)
    test_configs["pre_configs"] = pre_config_file
    test_configs["train_set_ids"] = f"0-{len(test_tomos)-1}"
    test_configs["val_set_ids"] = f"0-{len(test_tomos)-1}"
    test_configs["gpu_ids"] = str(gpu)
    test_configs["batch_size"] = batch_size
    test_configs["dset_name"] = f"test_inc{n_train_tomos}"
    test_configs["base_path"] = tmp_dir
    test_configs["tomo_path"] = f"{tmp_dir}/raw_data"
    
    test_cfg_file = f"{cfg_dir}/test.py"
    if os.path.exists(test_cfg_file):
        os.remove(test_cfg_file)
    with open(test_cfg_file, "w") as f:
        f.write("train_configs=")
        f.write(str(test_configs).replace("'", '"'))
    
    # Run inference
    print("\nRunning inference...")
    os.system(
        f"python ./DeepETPicker_ProPicker/bin/test_bash.py "
        f"--train_configs {test_cfg_file} "
        f"--checkpoints {ckpt_file} "
        f"--de_duplication True "
        f"--network ProPicker "
        f"--propicker_model_file '{PROPICKER_MODEL_FILE}' "
        f"--prompt_embed_file '{prompt_embed_file}'"
    )
    
    # Copy results
    print("\nCopying inference results...")
    
    runs_dir = os.path.join(str(EXP3_FINETUNING_DIR), f"increment_{n_train_tomos}", "runs", "train")
    version_dirs = sorted(glob.glob(os.path.join(runs_dir, "*", "version_*")))
    
    if version_dirs:
        latest_version = version_dirs[-1]
        
        # Create output directory
        os.makedirs(inference_output_dir, exist_ok=True)
        
        # Copy segmentation outputs
        seg_output_dir = os.path.join(latest_version, "full_segmentation_output")
        if os.path.exists(seg_output_dir):
            dest_locmaps = os.path.join(inference_output_dir, "locmaps")
            if os.path.exists(dest_locmaps):
                shutil.rmtree(dest_locmaps)
            shutil.copytree(seg_output_dir, dest_locmaps)
            print(f"  Localization maps saved to: {dest_locmaps}")
            
            # Convert .pt to .mrc
            import torch
            mrc_output_dir = os.path.join(inference_output_dir, "locmaps_mrc")
            os.makedirs(mrc_output_dir, exist_ok=True)
            for pt_file in glob.glob(os.path.join(dest_locmaps, "*.pt")):
                tomo_name = os.path.basename(pt_file).replace(".pt", "")
                locmap = torch.load(pt_file)
                if isinstance(locmap, dict):
                    locmap = locmap.get('segmentation', locmap.get('data', list(locmap.values())[0]))
                save_mrc_data(locmap.float(), os.path.join(mrc_output_dir, f"{tomo_name}_locmap.mrc"))
            print(f"  MRC localization maps saved to: {mrc_output_dir}")
        
        # Copy predicted coordinates
        predicted_labels_dir = os.path.join(latest_version, "PredictedLabels")
        for coords_subdir in ["Coords_All", "Coords_withArea"]:
            src_coords = os.path.join(predicted_labels_dir, coords_subdir)
            if os.path.exists(src_coords):
                dest_coords = os.path.join(inference_output_dir, coords_subdir)
                if os.path.exists(dest_coords):
                    shutil.rmtree(dest_coords)
                shutil.copytree(src_coords, dest_coords)
                print(f"  Predicted coordinates saved to: {dest_coords}")
    else:
        print("  WARNING: Could not find inference results in runs directory")
        return None
    
    # Cleanup
    print("\nCleaning up...")
    if os.path.exists(f"{tmp_dir}/raw_data"):
        shutil.rmtree(f"{tmp_dir}/raw_data")
    if os.path.exists(f"{tmp_dir}/data_std"):
        shutil.rmtree(f"{tmp_dir}/data_std")
    
    print(f"✅ Inference complete for increment {n_train_tomos}")
    return inference_output_dir


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on checkpoints from incremental fine-tuning")
    parser.add_argument(
        "--increment", type=int, default=None,
        help="Run inference only for this specific increment")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run inference even if results exist")
    parser.add_argument(
        "--prompt-file", type=str, default=None,
        help="Custom prompt embedding file (JSON). Default uses fixed_prompts_umusynth_thy.json")
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix to add to output directory (e.g., 'multi_prompt_n10')")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ProPicker Inference - Incremental Checkpoint Evaluation")
    print("=" * 70)
    
    # Set prompt file from arguments
    if args.prompt_file is not None:
        prompt_embed_file = args.prompt_file
        print(f"\n📌 Using custom prompt file: {prompt_embed_file}")
    else:
        prompt_embed_file = DEFAULT_PROMPT_FILE
    
    # Set output suffix
    output_suffix = args.output_suffix
    if output_suffix:
        print(f"📌 Output suffix: {output_suffix}")
    
    # Check prerequisites
    if not os.path.exists(prompt_embed_file):
        print(f"\n❌ ERROR: Prompt embeddings not found: {prompt_embed_file}")
        sys.exit(1)
    
    print(f"\n✅ Prerequisites found")
    print(f"  Prompt embeddings: {prompt_embed_file}")
    print(f"  Test tomograms: {test_tomos}")
    
    # Determine which increments to evaluate
    if args.increment is not None:
        if args.increment not in EXP3_INCREMENTS:
            print(f"\n❌ ERROR: Invalid increment {args.increment}")
            print(f"   Valid increments: {EXP3_INCREMENTS}")
            sys.exit(1)
        increments_to_run = [args.increment]
    else:
        increments_to_run = EXP3_INCREMENTS
    
    print(f"\n📋 Increments to evaluate: {increments_to_run}")
    
    # Run inference for each increment
    results = {}
    for n_tomos in increments_to_run:
        result_dir = run_inference_for_increment(n_tomos, force=args.force)
        results[n_tomos] = result_dir
    
    # Summary
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)
    
    for n_tomos, result_dir in results.items():
        status = "✅" if result_dir else "❌"
        print(f"  {status} Increment {n_tomos:2d}: {result_dir or 'FAILED'}")
    
    print(f"\n📁 Results saved to: {EXP3_INFERENCE_DIR}")
    print("\nNext step: Analyze results in the notebook")
    print("=" * 70)
