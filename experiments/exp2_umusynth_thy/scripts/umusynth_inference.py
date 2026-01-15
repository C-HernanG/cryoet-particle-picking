#!/usr/bin/env python3
"""
Inference script for ProPicker fine-tuned on UMU Synthetic Thyroglobulin dataset.
Based on empiar10988_inference.py

Usage:
    cd /home/carlos-hg/Dev/cryoet-particle-picking/tools/ProPicker
    conda activate deepetpicker
    python ../../experiments/exp2_umusynth_thy/scripts/umusynth_inference.py
"""

import sys
import os

# Add ProPicker to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PROPICKER_DIR = os.path.join(PROJECT_ROOT, "tools", "ProPicker")
sys.path.insert(0, PROPICKER_DIR)
os.chdir(PROPICKER_DIR)

import shutil
import copy
import glob
import importlib.util
from utils.mrctools import load_mrc_data, save_mrc_data

# Import project paths
sys.path.insert(0, PROJECT_ROOT)
from paths import PROPICKER_MODEL_FILE, UMU_SYNTH_TOMOS_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

# Tomograms to test (use validation set from fine-tuning)
test_tomos = [
    "tomo_rec_5_snr1.66", "tomo_rec_6_snr1.17", "tomo_rec_7_snr1.13",
    "tomo_rec_8_snr0.57", "tomo_rec_9_snr1.28"
]

# Checkpoint file from fine-tuning (will be auto-detected)
FINETUNING_DIR = os.path.join(PROJECT_ROOT, "results", "exp2_umusynth_thy", "fine_tuning_deepetpicker")
ckpt_file = None  # Will be auto-detected

# Training config file (generated during fine-tuning)
train_cfg_file = os.path.join(FINETUNING_DIR, "configs", "train.py")

# Prompt embedding file
prompt_embed_file = os.path.join(PROJECT_ROOT, "results", "exp2_umusynth_thy", "fixed_prompts_umusynth_thy.json")

# Inference parameters
gpu = 0
batch_size = 2

# Temporary directory for inference
tmp_dir = os.path.join(FINETUNING_DIR, "test")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_best_checkpoint(finetuning_dir):
    """Find the best checkpoint from training"""
    
    # Look for checkpoints in the runs directory
    runs_dir = os.path.join(finetuning_dir, "runs", "train")
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Training runs directory not found: {runs_dir}")
    
    # Find the version directory (usually version_0)
    version_dirs = sorted(glob.glob(os.path.join(runs_dir, "*", "version_*")))
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found in {runs_dir}")
    
    latest_version = version_dirs[-1]
    
    # Find checkpoints
    ckpt_dir = os.path.join(latest_version, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    
    # Return the most recent checkpoint (by epoch number)
    ckpts.sort(key=lambda x: int(x.split("epoch=")[1].split("-")[0]) if "epoch=" in x else 0)
    return ckpts[-1]

# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ProPicker Inference on UMU Synthetic Thyroglobulin Dataset")
    print("=" * 70)
    
    # Find checkpoint if not specified
    if ckpt_file is None:
        print("\nSearching for best checkpoint...")
        ckpt_file = find_best_checkpoint(FINETUNING_DIR)
        print(f"Found checkpoint: {ckpt_file}")
    
    # Check if required files exist
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")
    if not os.path.exists(train_cfg_file):
        raise FileNotFoundError(f"Training config file not found: {train_cfg_file}")
    if not os.path.exists(prompt_embed_file):
        raise FileNotFoundError(f"Prompt embedding file not found: {prompt_embed_file}")
    
    # Create temporary directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(f"{tmp_dir}/raw_data")
    
    print(f"\nProcessing {len(test_tomos)} test tomograms...")
    
    # Load and save test data (same preprocessing as empiar10988)
    for tomo_name in test_tomos:
        print(f"  Loading {tomo_name}...")
        tomo_file = os.path.join(str(UMU_SYNTH_TOMOS_DIR), f"{tomo_name}.mrc")
        tomo = -1 * load_mrc_data(tomo_file).float()  # Invert contrast (same as training)
        save_mrc_data(tomo, f"{tmp_dir}/raw_data/{tomo_name}.mrc")
        del tomo
    
    # Create preprocessing config (same as empiar10988_inference.py)
    print("\nCreating preprocessing config...")
    cfg_dir = os.path.dirname(train_cfg_file)
    pre_config_file = f"{cfg_dir}/preprocess_test.py"
    
    lines = [
        "pre_config={",
        f'"dset_name": "umusynth_test",',
        f'"base_path": "{tmp_dir}",',
        f'"tomo_path": "{tmp_dir}/raw_data",',
        f'"tomo_format": ".mrc",',
        f'"norm_type": "standardization",',
        f'"skip_coords": "True",',  # Don't need coordinates for testing
        f'"skip_labels": "True",',  # Don't need labels for testing
        f'"skip_ocp": "True"',      # Don't need occupancy for testing
        "}"
    ]
    
    if os.path.exists(pre_config_file):
        os.remove(pre_config_file)
    with open(pre_config_file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    
    # Run preprocessing
    print("\nRunning preprocessing...")
    os.system(f"python ./DeepETPicker_ProPicker/bin/preprocess.py --pre_configs {pre_config_file}")
    
    # Modify train config for testing (same as empiar10988_inference.py)
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
    test_configs["dset_name"] = "test"
    test_configs["base_path"] = tmp_dir
    test_configs["tomo_path"] = f"{tmp_dir}/raw_data"
    
    test_cfg_file = f"{cfg_dir}/test.py"
    if os.path.exists(test_cfg_file):
        os.remove(test_cfg_file)
    with open(test_cfg_file, "w") as f:
        f.write("train_configs=")
        f.write(str(test_configs).replace("'", '"'))
    
    # Run inference (same as empiar10988_inference.py)
    print("\n" + "=" * 70)
    print("Running inference...")
    print("=" * 70)
    
    os.system(
        f"python ./DeepETPicker_ProPicker/bin/test_bash.py "
        f"--train_configs {test_cfg_file} "
        f"--checkpoints {ckpt_file} "
        f"--de_duplication True "
        f"--network ProPicker "
        f"--propicker_model_file '{PROPICKER_MODEL_FILE}' "
        f"--prompt_embed_file '{prompt_embed_file}'"
    )
    
    print("\n" + "=" * 70)
    print("Inference complete!")
    print(f"Results saved to: {tmp_dir}")
    print("=" * 70)
    
    # Note: Results are kept for inspection (unlike empiar10988 which removes tmp_dir)
