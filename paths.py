"""
Central paths file for the cryoet-particle-picking project.

This file contains paths to data, models, and tools.
Modify this file to match your system's directory structure.

Usage:
    from paths import DATASETS_DIR, PROPICKER_MODEL_FILE
"""
from pathlib import Path

# =============================================================================
# Project Structure
# =============================================================================

# Project root directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TOOLS_DIR = PROJECT_ROOT / "tools"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Datasets
# =============================================================================

# Base directory for all datasets
# Change this if your datasets are stored elsewhere
DATASETS_DIR = DATA_DIR

# Synthetic datasets
UMU_SYNTH_DIR = DATASETS_DIR / "umu_synth"

# =============================================================================
# Tools
# =============================================================================

# ProPicker
PROPICKER_TOOLS_DIR = TOOLS_DIR / "ProPicker" / "propicker"

# =============================================================================
# Models
# =============================================================================

# ProPicker models
PROPICKER_MODEL_FILE = MODELS_DIR / "propicker.ckpt"
TOMOTWIN_MODEL_FILE = MODELS_DIR / "tomotwin.pth"

# Alternative: models stored in tools directory (uncomment if needed)
# PROPICKER_MODEL_FILE = PROPICKER_TOOLS_DIR / "propicker.ckpt"
# TOMOTWIN_MODEL_FILE = PROPICKER_TOOLS_DIR / "tomotwin.pth"

# =============================================================================
# UMU Synthetic Thyroglobulin Dataset Paths
# =============================================================================

UMU_SYNTH_TOMOS_DIR = UMU_SYNTH_DIR / "tomos"
UMU_SYNTH_LABELS_DIR = UMU_SYNTH_DIR / "thyroglobulin_labels"
UMU_SYNTH_CSV = UMU_SYNTH_DIR / "tomos_motif_list.csv"

# =============================================================================
# EMPIAR-10988 Dataset Path
# =============================================================================

# Data path (external - modify for your system)
EMPIAR10988_BASE_DIR = "/media/carlos-hg/SSDT5/Cryo-ET/data/ProPicker/empiar/10988/data/DEF"

# =============================================================================
# Experiment Output Directories
# =============================================================================

# Experiment 1: Ribosome on EMPIAR-10988
EXP1_RESULTS_DIR = RESULTS_DIR / "exp1_empiar10988_ribo"
EXP1_DATA_DIR = EXP1_RESULTS_DIR / "data"
EXP1_FINETUNING_DIR = EXP1_RESULTS_DIR / "fine_tuning"
EXP1_INFERENCE_DIR = EXP1_RESULTS_DIR / "inference"
EXP1_COORDS_DIR = EXP1_RESULTS_DIR / "coords"

# Experiment 2: UMU Synthetic Thyroglobulin
EXP2_RESULTS_DIR = RESULTS_DIR / "exp2_umusynth_thy"
EXP2_DATA_DIR = EXP2_RESULTS_DIR / "data"
EXP2_FINETUNING_DIR = EXP2_RESULTS_DIR / "fine_tuning"
EXP2_INFERENCE_DIR = EXP2_RESULTS_DIR / "inference"
EXP2_COORDS_DIR = EXP2_RESULTS_DIR / "coords"

# Experiment 3: Incremental Fine-Tuning Analysis
EXP3_RESULTS_DIR = RESULTS_DIR / "exp3_ppicker_limits"
EXP3_DATA_DIR = EXP3_RESULTS_DIR / "data"
EXP3_FINETUNING_DIR = EXP3_RESULTS_DIR / "fine_tuning"
EXP3_INFERENCE_DIR = EXP3_RESULTS_DIR / "inference"
EXP3_COORDS_DIR = EXP3_RESULTS_DIR / "coords"
EXP3_CHECKPOINTS_DIR = EXP3_RESULTS_DIR / "checkpoints"

# Experiment 4: Rotation Invariance Analysis
EXP4_RESULTS_DIR = RESULTS_DIR / "exp4_ppicker_rotations"
EXP4_DATA_DIR = EXP4_RESULTS_DIR / "data"
EXP4_PROMPTS_DIR = EXP4_RESULTS_DIR / "prompts"
EXP4_INFERENCE_DIR = EXP4_RESULTS_DIR / "inference"
EXP4_COORDS_DIR = EXP4_RESULTS_DIR / "coords"
