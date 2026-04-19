# Universal Deep Learning Detectors for Macromolecule Localization in Cryo-ET

This repository contains the experimental code, notebooks and thesis material for a ProPicker-based study of promptable cryo-ET particle picking. The work combines one real-data proof of concept and a controlled synthetic benchmark to study four questions:

1. whether prompt-only transfer is useful on real data;
2. how strongly domain shift degrades performance;
3. how much annotated data is needed after fine-tuning;
4. how sensitive the detector remains to the prompt after fine-tuning.

The repository is organized as a reproducible research workspace rather than a single training script. Preprocessing, prompt generation, fine-tuning, inference, evaluation and robustness analysis are split across notebooks and long-run scripts.

## Current Status

Completed experiments:

| Experiment | Focus | Status |
| --- | --- | --- |
| `exp1_empiar10988_ribo` | EMPIAR-10988 ribosome proof of concept | Complete |
| `exp2_umusynth_thy` | UMU synthetic thyroglobulin baseline | Complete |
| `exp3_ppicker_limits` | Data efficiency and single vs multi prompt inference | Complete |
| `exp4_ppicker_rotations` | Prompt robustness, orientation, and prompt-quality analysis | Complete |

## Main Findings

- `EXP1`: on EMPIAR-10988 ribosomes, prompt-only inference is informative but unbalanced, and fine-tuning raises the operating point from `F1 = 0.3074` to `F1 = 0.5714`.
- `EXP2`: on UMU synthetic thyroglobulin, the pretrained model collapses under domain shift (`F1 ~ 0.008`), while fine-tuning restores strong performance (`mean F1 = 0.891`).
- `EXP3`: most of the attainable performance is recovered early; the best observed configuration is `multi_prompt_n10` with 12 training tomograms (`F1 = 0.919`), while 4 tomograms already provide a strong data-efficient regime.
- `EXP4`: prompt averages remain strong (`F1 = 0.780 +/- 0.283` over the 125-prompt study subset), but some prompts collapse in recall. The final conclusion is that a bad prompt is best defined operationally as a prompt that fails to generalize to held-out tomograms, not as one identifiable by source SNR alone or by a single rotation-to-`Z` rule.

## Repository Layout

```text
cryoet-particle-picking/
├── README.md
├── paths.py
├── data/                          # Local datasets, not tracked by git
├── docs/                          # Thesis, figures, PDFs, and supporting material
├── experiments/
│   ├── README.md
│   ├── config.py
│   ├── exp1_empiar10988_ribo/
│   ├── exp2_umusynth_thy/
│   ├── exp3_ppicker_limits/
│   └── exp4_ppicker_rotations/
├── models/                        # Checkpoints, not tracked by git
├── results/                       # Derived outputs, prompts, logs, plots
└── tools/                         # External dependencies, not tracked by git
    └── ProPicker/
```

Important directories:

- `paths.py`: central filesystem registry for datasets, checkpoints, tools and result directories.
- `experiments/config.py`: shared constants and helper utilities reused across experiments.
- `experiments/<exp>/`: experiment-specific notebooks, scripts and sometimes local configuration.
- `results/`: generated prompts, coordinates, inference outputs, plots and checkpoints.
- `docs/`: thesis source, final PDF, figures extracted from experiments and supporting documentation.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/C-HernanG/cryoet-particle-picking.git
cd cryoet-particle-picking
```

### 2. Clone ProPicker into `tools/`

```bash
cd tools
git clone https://github.com/MLI-lab/ProPicker.git ProPicker
cd ..
```

### 3. Download checkpoints

Place the required checkpoints in `models/`. The repository expects at least:

- a ProPicker checkpoint;
- a TomoTwin prompt-encoder checkpoint.

### 4. Download datasets

Populate `data/` with the local copies of the datasets used by the experiments:

- EMPIAR-10988 for the ribosome proof of concept;
- UMU synthetic tomograms and annotations for the thyroglobulin benchmark.

### 5. Configure local paths

Edit `paths.py` so it points to your local datasets, checkpoints, tools and result directories. Shared experiment constants live in `experiments/config.py`.

Minimal example:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TOOLS_DIR = PROJECT_ROOT / "tools"

EMPIAR10988_BASE_DIR = DATA_DIR / "empiar10988"
PROPICKER_MODEL_FILE = MODELS_DIR / "propicker.ckpt"
TOMOTWIN_MODEL_FILE = MODELS_DIR / "tomotwin.pth"
PROPICKER_DIR = TOOLS_DIR / "ProPicker"
```

### 6. Prepare the execution environment

The long-run scripts are written around the ProPicker / DeepETPicker stack. In practice this means:

- create the Python or conda environment expected by ProPicker;
- install the dependencies required by the notebooks;
- run the long-run training and inference scripts from inside `tools/ProPicker/` when the script expects that working directory.

## Running the Experiments

The typical workflow is:

1. open the notebook for the target experiment;
2. run preprocessing, prompt generation and config-export cells;
3. launch fine-tuning or inference from the corresponding script;
4. return to the notebook for evaluation, visualization and robustness analysis.

Most scripts assume that the notebook has already generated the prompt JSON files, converted coordinates and experiment config files.

Example long-run invocation pattern:

```bash
conda activate deepetpicker
cd tools/ProPicker

python ../../experiments/exp2_umusynth_thy/scripts/umusynth_fine_tuning.py
python ../../experiments/exp3_ppicker_limits/scripts/umusynth_inference.py
python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference_large.py
```

For a more detailed walkthrough of each experiment, see [experiments/README.md](experiments/README.md).

## Experiment Summary

| Experiment | Dataset | Main question | Main outputs |
| --- | --- | --- | --- |
| `exp1_empiar10988_ribo` | EMPIAR-10988 | Can promptable transfer work on real cryo-ET ribosomes? | Prompt embeddings, fine-tuned ribosome checkpoint, real-data evaluation |
| `exp2_umusynth_thy` | UMU synthetic | How severe is domain shift and how much does fine-tuning recover? | Fixed thyroglobulin prompts, stable 20/5 split, synthetic evaluation |
| `exp3_ppicker_limits` | UMU synthetic | How much data is needed and does multi-prompt inference help? | Increment checkpoints, single prompt and multi-prompt prompt files, comparison plots |
| `exp4_ppicker_rotations` | UMU synthetic | What makes a prompt fail after fine-tuning? | Large prompt population, quality-filtered rotation-diverse prompts, prompt-level robustness analysis |

## Results and Outputs

The `results/` directory is not treated as a dump folder. It stores structured outputs that are consumed again by later notebooks:

- `fixed_prompts_*.json`: prompt embeddings generated from subtomograms;
- `single_instance_prompt.json` and `multi_instance_prompt_n10.json`: operational prompts for `EXP3`;
- checkpoint families for increment studies and follow-up inference;
- prompt-level analysis tables and plots for `EXP4`.

Keeping filenames stable matters because later analysis notebooks read these files directly.

## Documentation

The thesis material lives in `docs/`. The repository keeps the compiled manuscript and exported figures; the local LaTeX working source is ignored in Git.

Key files:

- [docs/HernanGuirao_Carlos_TFG_CryoET_UniversalDetectors.pdf](docs/HernanGuirao_Carlos_TFG_CryoET_UniversalDetectors.pdf)
- `docs/assets/figures/results/`: figures exported from the experiment notebooks

## Reproducibility Notes

- Keep the UMU train/validation split fixed across `EXP2`, `EXP3`, and `EXP4`.
- Generate prompts before running the long-run scripts.
- Run scripts from the working directory expected by the ProPicker helpers.
- Preserve the distinction between source code in `experiments/` and generated artifacts in `results/`.
- Export localization maps to MRC when qualitative volumetric inspection is needed.

## License and Attribution

This repository contains original experiment code and documentation, but it depends on external tools and pretrained models such as ProPicker and TomoTwin. Check the upstream licenses of those projects before redistributing code, checkpoints or datasets.
