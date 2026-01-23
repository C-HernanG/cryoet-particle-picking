# Experiments

This directory contains all experiments for particle picking in Cryo-ET using deep learning methods.

## Structure

```
experiments/
├── config.py                      # Shared configuration (utilities, common parameters)
├── exp1_empiar10988_ribo/         # Experiment 1: EMPIAR-10988 Ribosomes
│   ├── exp1_empiar10988_ribo_ppicker_finetuning.ipynb
│   ├── exp1_empiar10988_ribo_ppicker_inference.ipynb
│   └── scripts/
│       ├── empiar10988_fine_tuning.py
│       └── empiar10988_inference.py
└── exp2_umusynth_thy/             # Experiment 2: UMU Synthetic Thyroglobulin
    ├── exp2_umusynth_thy_ppicker_finetuning.ipynb
    └── scripts/
        ├── umusynth_fine_tuning.py
        ├── umusynth_inference.py
        └── update_csv_paths.py
```

## Configuration

### Shared Configuration (`config.py`)

Located at `experiments/config.py`, contains:

- `setup_propicker_paths()`: Utility to add ProPicker to Python path
- Common parameters used across experiments (e.g., `PROMPT_SIZE`, `LABEL_DIAMETER`)

### Experiment-Specific Configuration

Each experiment folder may have its own `config.py` with parameters specific to that dataset:

- **exp1**: `RIBOSOME_*` parameters, training/validation splits, hyperparameters
- **exp2**: `EXP2_*` parameters imported from shared config

---

## Experiments Overview

### Experiment 1: EMPIAR-10988 Ribosomes (`exp1_empiar10988_ribo/`)

**Dataset**: EMPIAR-10988 - Cytoplasmic ribosomes from *S. cerevisiae*

**Goal**: Fine-tune ProPicker for ribosome picking using small labeled crops.

**Workflow**:

1. **Notebook** (`exp1_..._ppicker_finetuning.ipynb`):
   - Extract prompts from tomograms
   - Generate TomoTwin embeddings
   - Visualize and evaluate results

2. **Script** (`scripts/empiar10988_fine_tuning.py`):
   - Run fine-tuning via DeepETPicker framework
   - Must be executed from `tools/ProPicker/` directory

3. **Script** (`scripts/empiar10988_inference.py`):
   - Run inference with fine-tuned model
   - Auto-detects best checkpoint

**Key Parameters** (in `exp1/config.py`):

| Parameter | Value |
|-----------|-------|
| Train TS | TS_029 |
| Val TS | TS_030 |
| Crop delta | 64 (128×128×128 voxels) |
| Epochs | 75 |
| Batch size | 8 |
| Learning rate | 1e-3 |

---

### Experiment 2: UMU Synthetic Thyroglobulin (`exp2_umusynth_thy/`)

**Dataset**: UMU Synthetic - 25 tomograms with ~3,327 thyroglobulin particles

**Goal**: Fine-tune ProPicker on synthetic data with known ground truth.

**Workflow**:

1. **Notebook** (`exp2_..._ppicker_finetuning.ipynb`):
   - Update CSV paths to local data
   - Filter thyroglobulin instances (Label=7)
   - Convert coordinates from Angstroms to voxels
   - Extract 37×37×37 prompts and generate TomoTwin embeddings

2. **Script** (`scripts/umusynth_fine_tuning.py`):
   - Run fine-tuning via DeepETPicker framework
   - Uses binary labels from dataset

3. **Script** (`scripts/umusynth_inference.py`):
   - Run inference on validation tomograms

**Key Parameters** (in shared `config.py`):

| Parameter | Value |
|-----------|-------|
| Train tomos | 20 |
| Val tomos | 5 |
| Epochs | 40 |
| Batch size | 8 |
| Learning rate | 1e-3 |

---

## How to Run

### Prerequisites

1. Configure paths in `paths.py` at project root
2. Download required datasets to `data/`
3. Download model checkpoints to `models/`
4. Create conda environments following tool instructions

### Running Fine-Tuning Scripts

All fine-tuning scripts must be run from the `tools/ProPicker/` directory:

```bash
# Activate DeepETPicker environment
conda activate deepetpicker

# Navigate to ProPicker directory
cd /path/to/cryoet-particle-picking/tools/ProPicker

# Run experiment 1
python ../../experiments/exp1_empiar10988_ribo/scripts/empiar10988_fine_tuning.py

# Run experiment 2
python ../../experiments/exp2_umusynth_thy/scripts/umusynth_fine_tuning.py
```

### Running Notebooks

Notebooks can be run directly from VS Code or Jupyter:

1. Open the experiment notebook
2. Ensure the correct Python environment is selected
3. Run all preprocessing cells first
4. Then run fine-tuning script from terminal
5. Return to notebook for evaluation

---

## Output Structure

Results are saved to `results/<experiment_name>/`:

```
results/
├── exp1_empiar10988_ribo/
│   ├── fixed_prompts_empiar10988.json    # TomoTwin embeddings
│   └── fine_tuning_deepetpicker/
│       └── crop_delta=64/
│           ├── configs/                   # Training configs
│           ├── data_std/                  # Normalized tomograms
│           ├── gaussian21/                # Labels
│           └── runs/                      # Checkpoints & logs
└── exp2_umusynth_thy/
    ├── coords/                            # Extracted coordinates
    ├── data/                              # Preprocessed data
    ├── fixed_prompts_umusynth_thy.json   # TomoTwin embeddings
    └── fine_tuning_deepetpicker/
        ├── configs/
        └── runs/
```

---

## Adding New Experiments

1. Create a new folder: `experiments/exp<N>_<dataset>_<particle>/`
2. Create experiment-specific `config.py` if needed
3. Create preprocessing notebook following existing patterns
4. Create fine-tuning and inference scripts based on templates
5. Update this README with experiment details
