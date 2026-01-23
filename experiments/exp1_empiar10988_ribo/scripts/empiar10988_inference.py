#!/usr/bin/env python3
"""
Inference script for ProPicker fine-tuned on EMPIAR-10988 Ribosome dataset.
Adapted from ProPicker tutorial2 to work from experiments directory.

Usage:
    cd /path/to/cryoet-particle-picking/tools/ProPicker
    conda activate deepetpicker
    python ../../experiments/exp1_empiar10988_ribo/scripts/empiar10988_inference.py
"""

from utils.mrctools import load_mrc_data, save_mrc_data
from data.preparation_functions.prepare_empiar10988 import empiar10988_ts_to_slice_of_interest
from paths import PROPICKER_MODEL_FILE, EMPIAR10988_BASE_DIR, EXP1_RESULTS_DIR
from experiments.config import EXP1_CROP_DELTA, EXP1_VAL_TS
import importlib.util
import glob
import copy
import shutil
import sys
import os

# Add paths BEFORE any project imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
PROPICKER_DIR = os.path.join(PROJECT_ROOT, "tools", "ProPicker")

# Add ProPicker tools to path (for utils, data, etc.)
sys.path.insert(0, PROPICKER_DIR)
os.chdir(PROPICKER_DIR)

# Add project root to path for paths.py
sys.path.insert(0, PROJECT_ROOT)

# Add experiments to path for config import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

# Now import project modules

# =============================================================================
# CONFIGURATION
# =============================================================================

# Tomograms to test (use validation set from fine-tuning)
test_ts = EXP1_VAL_TS
crop_delta = EXP1_CROP_DELTA

# Paths based on fine-tuning output
FINETUNING_DIR = os.path.join(
    str(EXP1_RESULTS_DIR), "fine_tuning_deepetpicker", f"crop_delta={crop_delta}")
ckpt_file = None  # Will be auto-detected

# Training config file (generated during fine-tuning)
train_cfg_file = os.path.join(FINETUNING_DIR, "configs", "train.py")

# Prompt embedding file
prompt_embed_file = os.path.join(
    str(EXP1_RESULTS_DIR), "fixed_prompts_empiar10988.json")

# Inference parameters
gpu = 0
batch_size = 16

# Temporary directory for inference
tmp_dir = os.path.join(FINETUNING_DIR, "test")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_best_checkpoint(finetuning_dir):
    """Find the best checkpoint from training"""

    runs_dir = os.path.join(finetuning_dir, "runs", "train")
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(
            f"Training runs directory not found: {runs_dir}")

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
    ckpts.sort(key=lambda x: int(x.split("epoch=")[
               1].split("-")[0]) if "epoch=" in x else 0)
    return ckpts[-1]

# =============================================================================
# MAIN SCRIPT
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("ProPicker Inference on EMPIAR-10988 Ribosome Dataset")
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
        raise FileNotFoundError(
            f"Training config file not found: {train_cfg_file}")
    if not os.path.exists(prompt_embed_file):
        raise FileNotFoundError(
            f"Prompt embedding file not found: {prompt_embed_file}")

    # Create temporary directory
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(f"{tmp_dir}/raw_data")

    print(f"\nProcessing {len(test_ts)} test tomograms: {test_ts}")

    # Load and save test data
    for ts_id in test_ts:
        print(f"  Loading {ts_id}...")
        tomo_file = f"{EMPIAR10988_BASE_DIR}/tomograms/{ts_id}.rec"
        slice_of_interest = empiar10988_ts_to_slice_of_interest[ts_id]
        tomo = -1 * load_mrc_data(tomo_file).float()[slice_of_interest]
        save_mrc_data(tomo, f"{tmp_dir}/raw_data/{ts_id}.mrc")
        del tomo

    # Create preprocessing config
    print("\nCreating preprocessing config...")
    cfg_dir = os.path.dirname(train_cfg_file)
    pre_config_file = f"{cfg_dir}/preprocess_test.py"

    lines = [
        "pre_config={",
        f'"dset_name": "empiar10988_test",',
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

    # Modify train config for testing
    print("\nCreating test config...")
    module_name = "train_configs_module"
    spec = importlib.util.spec_from_file_location(module_name, train_cfg_file)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    test_configs = copy.deepcopy(train_module.train_configs)
    test_configs["pre_configs"] = pre_config_file
    test_configs["train_set_ids"] = f"0-{len(test_ts)-1}"
    test_configs["val_set_ids"] = f"0-{len(test_ts)-1}"
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

    # Run inference
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
