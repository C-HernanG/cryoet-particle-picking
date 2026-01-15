"""
Configuration and common utilities for experiments.

This module contains:
1. Experiment parameters (particle sizes, labels, names, etc.)
2. Utilities to configure tool paths like ProPicker

Usage:
    from config import setup_propicker_paths, THYROGLOBULIN_DIAMETER, RIBOSOME_NAME
    from paths import PROPICKER_MODEL_FILE, TOMOTWIN_MODEL_FILE
    
    setup_propicker_paths()
    
    # Now you can import ProPicker modules
    from clustering_and_picking import get_cluster_centroids_df
"""
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Add project root to path to import paths
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import paths needed for setup_propicker_paths
from paths import PROPICKER_TOOLS_DIR

# =============================================================================
# UMU Synthetic Thyroglobulin Parameters (Experiment 3)
# =============================================================================

THYROGLOBULIN_LABEL = 7           # Label in CSV
THYROGLOBULIN_NAME = "thyroglobulin"
PROMPT_SIZE = 37                  # Required by TomoTwin (37x37x37)
PROMPT_HALF = PROMPT_SIZE // 2    # 18
THYROGLOBULIN_DIAMETER = 30       # Approximate diameter in pixels
LABEL_DIAMETER = 21               # For gaussian labels

# =============================================================================
# EMPIAR-10988 Ribosome Parameters (Experiment 2)
# =============================================================================

RIBOSOME_NAME = "cyto_ribosome"
RIBOSOME_PROMPT_SIZE = 37         # Required by TomoTwin (37x37x37)
RIBOSOME_DIAMETER = 24            # Approximate diameter in pixels

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
