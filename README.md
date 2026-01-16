# Universal Deep Learning Detectors for Macromolecule Localization in Cryo-ET

> **Note:** This repository contains experimental implementations and benchmarks. A comprehensive paper documenting these methods and results is currently in development (`docs/CryoET_Particle_Picking.pdf`).

---

## Project Structure

```
cryoet-particle-picking/
├── paths.py                       # Central paths configuration file
├── data/                          # Datasets (not tracked by git)
│   └── .gitkeep
├── docs/                          # Documentation and papers
├── experiments/                   # Experiment notebooks and scripts
│   ├── config.py                  # Configuration and utilities for experiments
│   ├── exp1_empiar10988_ribo/     # ProPicker fine-tuning on EMPIAR-10988 ribosomes
│   └── exp2_umusynth_thy/         # ProPicker fine-tuning with UMU synthetic thyroglobulin
├── models/                        # Pre-trained models (not tracked by git)
│   ├── ProPicker/
│   └── TomoTwin/
├── results/                       # Experiment results and outputs (not tracked by git)
├── tools/                         # External tools (not tracked by git)
│   └── ProPicker/                 # Cloned ProPicker repository
└── README.md
```

### Folder Descriptions

| Folder/File | Description |
|-------------|-------------|
| `paths.py` | **Central paths configuration file** with all paths to data, models, and tools. |
| `experiments/config.py` | **Configuration and utilities** for experiments, re-exports paths and provides helper functions. |
| `data/` | Contains datasets (MRC tomograms, coordinates, labels). **Not tracked by git** due to large file sizes. |
| `docs/` | Documentation, papers, and reference materials. |
| `experiments/` | Jupyter notebooks and scripts for each experiment. Each subfolder is self-contained. |
| `models/` | Pre-trained model checkpoints (`.ckpt`, `.pth`). **Not tracked by git**. |
| `results/` | Experiment outputs including predicted coordinates, fine-tuned models, and evaluation metrics. **Not tracked by git**. |
| `tools/` | External repositories and dependencies (e.g., ProPicker). **Not tracked by git**. |

> Folders marked as "not tracked by git" contain `.gitkeep` files to preserve the directory structure.

---

## Initial Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/C-HernanG/cryoet-particle-picking.git
cd cryoet-particle-picking
```

### Step 2: Install Required Tools

Clone the necessary external tools into the `tools/` directory. Follow the installation instructions from each tool's repository:

- **ProPicker**: <https://github.com/MLI-lab/ProPicker>

```bash
cd tools/
git clone https://github.com/MLI-lab/ProPicker.git ProPicker
cd ..
```

### Step 3: Download Pre-trained Models

Download the required model checkpoints following the instructions from each tool's repository. Models can be placed in `models/` for centralized organization.

### Step 4: Download Datasets

Download the required datasets and place them in the `data/` directory.

### Step 5: Configure Paths and Parameters

**Paths are configured in:** `paths.py` at the project root.
**Experiment parameters are in:** `experiments/config.py`.

Edit `paths.py` to match your system:

```python
# paths.py - File system paths only

# Base directory for all datasets
DATASETS_DIR = DATA_DIR  # or use an absolute path

# EMPIAR-10988 dataset
EMPIAR10988_BASE_DIR = DATASETS_DIR / "empiar10988"

# Models
PROPICKER_MODEL_FILE = MODELS_DIR / "propicker.ckpt"
TOMOTWIN_MODEL_FILE = MODELS_DIR / "tomotwin.pth"
```

Experiment parameters (particle sizes, labels, etc.) are in `experiments/config.py`:

```python
# experiments/config.py - Experiment parameters

# Thyroglobulin parameters
THYROGLOBULIN_LABEL = 7
THYROGLOBULIN_DIAMETER = 30

# Ribosome parameters  
RIBOSOME_NAME = "cyto_ribosome"
RIBOSOME_DIAMETER = 24
```

### Step 6: Create Python Environment

It is recommended to create a separate Python environment for each tool following the instructions from their respective repositories.

---

## Running Experiments

Each experiment is located in `experiments/<experiment_name>/` and uses two configuration files:

- **`paths.py`**: File system paths (datasets, models, output directories)
- **`experiments/config.py`**: Experiment parameters (particle sizes, labels, utilities)

```python
# In your notebook or script
import sys
from pathlib import Path

# Add project root and experiments to path
PROJECT_ROOT = Path("../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

# Import paths (file system locations)
from paths import (
    PROPICKER_MODEL_FILE,
    TOMOTWIN_MODEL_FILE,
    EMPIAR10988_BASE_DIR,
)

# Import config (experiment parameters and utilities)
from config import setup_propicker_paths, RIBOSOME_DIAMETER

# Setup ProPicker imports
setup_propicker_paths()

# Now you can import ProPicker modules
from model import ProPicker
from inference import get_pred_locmap_dict
```

### Available Experiments

| Experiment | Description | Files |
|------------|-------------|-------|
| `exp1_empiar10988_ribo` | ProPicker fine-tuning and inference on EMPIAR-10988 ribosomes | `exp1_empiar10988_ribo_ppicker_finetuning.ipynb`<br>`exp1_empiar10988_ribo_ppicker_inference.ipynb` |
| `exp2_umusynth_thy` | ProPicker fine-tuning with UMU synthetic thyroglobulin data | `exp2_umusynth_thy_ppicker_finetuning.ipynb`<br>`scripts/umusynth_fine_tuning.py`<br>`scripts/umusynth_inference.py`<br>`scripts/update_csv_paths.py` |

### Configuration Files

#### `paths.py` (Project Root)

Contains all file system paths:

- `PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `TOOLS_DIR`: Project structure
- `PROPICKER_MODEL_FILE`, `TOMOTWIN_MODEL_FILE`: Model paths
- `EMPIAR10988_BASE_DIR`, `UMU_SYNTH_DIR`: Dataset paths
- `EXP2_*`, `EXP3_*`: Experiment output directories

#### `experiments/config.py`

Contains experiment parameters and utilities:

- `setup_propicker_paths()`: Adds ProPicker tools to `sys.path`
- `THYROGLOBULIN_*`: Thyroglobulin particle parameters
- `RIBOSOME_*`: Ribosome particle parameters
- `PROMPT_SIZE`, `LABEL_DIAMETER`: Common experiment values
