"""
Configuration and common utilities for experiments.

This module contains:
1. Experiment 1 (EMPIAR-10988 Ribosome) parameters
2. Experiment 2 (UMU Synth Thyroglobulin) parameters
3. Utilities to configure tool paths like ProPicker

Usage:
    from experiments.config import (
        setup_propicker_paths,
        # EXP1 parameters
        RIBOSOME_NAME, RIBOSOME_DIAMETER, RIBOSOME_PROMPT_SIZE,
        EXP1_TRAIN_TS, EXP1_VAL_TS, EXP1_CROP_DELTA,
        # EXP2 parameters
        THYROGLOBULIN_DIAMETER, EXP2_TRAIN_TOMOS,
    )
    from paths import PROPICKER_MODEL_FILE, TOMOTWIN_MODEL_FILE
    
    setup_propicker_paths()
    
    # Now you can import ProPicker modules
    from clustering_and_picking import get_cluster_centroids_df
"""
from paths import PROPICKER_TOOLS_DIR
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add project root to path to import paths
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import paths needed for setup_propicker_paths

# =============================================================================
# EXP1: EMPIAR-10988 Ribosome Parameters
# =============================================================================

RIBOSOME_NAME = "cyto_ribosome"
RIBOSOME_PROMPT_SIZE = 37         # Required by TomoTwin (37x37x37)
RIBOSOME_DIAMETER = 24            # Approximate diameter in pixels

# EXP1: Training/Validation tomograms
EXP1_TRAIN_TS = ["TS_029"]
EXP1_VAL_TS = ["TS_030"]

# EXP1: Crop size for training (center crop)
EXP1_CROP_DELTA = 64

# EXP1: Epochs per crop size (from tutorial)
EXP1_CROP_DELTA_EPOCHS = {
    64: 75,
    128: 50,
    256: 25,
    512: 15,
}

# EXP1: Training parameters
EXP1_BATCH_SIZE = 8
EXP1_BLOCK_SIZE = 72
EXP1_PAD_SIZE = 12
EXP1_LEARNING_RATE = 1e-3
EXP1_LABEL_DIAMETER = 21
EXP1_GPU_ID = 0
EXP1_USE_BINARY_LABELS = True

# =============================================================================
# EXP2: UMU Synthetic Thyroglobulin Parameters
# =============================================================================

THYROGLOBULIN_LABEL = 7           # Label in CSV
THYROGLOBULIN_NAME = "thyroglobulin"
PROMPT_SIZE = 37                  # Required by TomoTwin (37x37x37)
PROMPT_HALF = PROMPT_SIZE // 2    # 18
THYROGLOBULIN_DIAMETER = 30       # Approximate diameter in pixels
LABEL_DIAMETER = 21               # For gaussian labels

# EXP2: Fine-tuning configuration
EXP2_TRAIN_TOMOS = [
    "tomo_rec_0_snr1.63", "tomo_rec_1_snr1.46", "tomo_rec_2_snr1.07", "tomo_rec_3_snr0.63",
    "tomo_rec_4_snr1.85", "tomo_rec_10_snr0.97", "tomo_rec_11_snr1.41", "tomo_rec_12_snr1.39",
    "tomo_rec_13_snr1.1", "tomo_rec_14_snr0.43", "tomo_rec_15_snr1.03", "tomo_rec_16_snr0.93",
    "tomo_rec_17_snr0.92", "tomo_rec_18_snr0.78", "tomo_rec_19_snr1.62", "tomo_rec_20_snr0.73",
    "tomo_rec_21_snr0.34", "tomo_rec_22_snr0.97", "tomo_rec_23_snr0.29", "tomo_rec_24_snr1.39"
]
EXP2_VAL_TOMOS = [
    "tomo_rec_5_snr1.66", "tomo_rec_6_snr1.17", "tomo_rec_7_snr1.13",
    "tomo_rec_8_snr0.57", "tomo_rec_9_snr1.28"
]

# EXP2: Training parameters
EXP2_MAX_EPOCHS = 20
EXP2_BATCH_SIZE = 2
EXP2_BLOCK_SIZE = 72
EXP2_PAD_SIZE = 12
EXP2_LEARNING_RATE = 1e-3
EXP2_LABEL_DIAMETER = 21
EXP2_GPU_ID = 0
EXP2_USE_BINARY_LABELS = True

# =============================================================================
# EXP3: Incremental Fine-Tuning Analysis Parameters
# =============================================================================

# EXP3: Same base parameters as EXP2
THYROGLOBULIN_LABEL = 7           # Label in CSV (same as EXP2)

# EXP3: Validation set (fixed - same tomograms for all increments)
EXP3_VAL_TOMOS = [
    "tomo_rec_5_snr1.66", "tomo_rec_6_snr1.17", "tomo_rec_7_snr1.13",
    "tomo_rec_8_snr0.57", "tomo_rec_9_snr1.28"
]

# EXP3: Training pool (will be incrementally added)
EXP3_TRAIN_POOL = [
    "tomo_rec_0_snr1.63", "tomo_rec_1_snr1.46", "tomo_rec_2_snr1.07", "tomo_rec_3_snr0.63",
    "tomo_rec_4_snr1.85", "tomo_rec_10_snr0.97", "tomo_rec_11_snr1.41", "tomo_rec_12_snr1.39",
    "tomo_rec_13_snr1.1", "tomo_rec_14_snr0.43", "tomo_rec_15_snr1.03", "tomo_rec_16_snr0.93",
    "tomo_rec_17_snr0.92", "tomo_rec_18_snr0.78", "tomo_rec_19_snr1.62", "tomo_rec_20_snr0.73",
    "tomo_rec_21_snr0.34", "tomo_rec_22_snr0.97", "tomo_rec_23_snr0.29", "tomo_rec_24_snr1.39"
]

# EXP3: Incremental training schedule (number of training tomograms at each step)
# This defines how many tomograms to use at each training increment
EXP3_INCREMENTS = [1, 2, 4, 8, 12, 16, 20]

# EXP3: Training parameters (same as EXP2)
EXP3_MAX_EPOCHS = 20
EXP3_BATCH_SIZE = 2
EXP3_BLOCK_SIZE = 72
EXP3_PAD_SIZE = 12
EXP3_LEARNING_RATE = 1e-3
EXP3_LABEL_DIAMETER = 21
EXP3_GPU_ID = 0
EXP3_USE_BINARY_LABELS = True

# EXP3: Checkpoint saving configuration
EXP3_SAVE_EVERY_N_EPOCHS = 5  # Save checkpoint every N epochs
EXP3_SAVE_BEST = True         # Always save best checkpoint based on val_loss

# =============================================================================
# Utilities
# =============================================================================


def setup_propicker_paths():
    """
    Add ProPicker tools to sys.path to import its modules.

    Returns:
        Path to ProPicker tools directory
    """
    propicker_path = str(PROPICKER_TOOLS_DIR)
    if propicker_path not in sys.path:
        sys.path.insert(0, propicker_path)
    return PROPICKER_TOOLS_DIR
