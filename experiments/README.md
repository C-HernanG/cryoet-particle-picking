# Experiments

This directory contains the experiment notebooks and long-run scripts used throughout the thesis. The experiments are cumulative: each one answers a narrower question while reusing the same basic preprocessing and evaluation logic whenever possible.

## Directory Structure

```text
experiments/
├── README.md
├── config.py
├── exp1_empiar10988_ribo/
│   ├── exp1_empiar10988_ribo_ppicker_finetuning.ipynb
│   ├── exp1_empiar10988_ribo_ppicker_inference.ipynb
│   └── scripts/
│       ├── empiar10988_fine_tuning.py
│       └── empiar10988_inference.py
├── exp2_umusynth_thy/
│   ├── exp2_umusynth_thy_ppicker_finetuning.ipynb
│   ├── exp2_umusynth_thy_ppicker_inference.ipynb
│   └── scripts/
│       ├── umusynth_fine_tuning.py
│       ├── umusynth_inference.py
│       └── update_csv_paths.py
├── exp3_ppicker_limits/
│   ├── exp3_ppicker_limits.ipynb
│   ├── exp3_ppicker_limits_inference.ipynb
│   └── scripts/
│       ├── umusynth_fine_tuning.py
│       ├── umusynth_inference.py
│       └── update_csv_paths.py
└── exp4_ppicker_rotations/
    ├── exp4_ppicker_rotations.ipynb
    ├── exp4_large_prompt_population_inference.ipynb
    ├── exp4_inference_prompt_mix.ipynb
    ├── exp4_underperforming_prompts_analysis.ipynb
    ├── rotational_issues_analysis.py
    └── scripts/
        ├── extract_prompt_subvolumes_3d.py
        ├── umusynth_inference.py
        ├── umusynth_inference_followup.py
        └── umusynth_inference_large.py
```

## Shared Configuration

`experiments/config.py` provides the shared pieces reused across experiments:

- path-setup helpers for importing ProPicker code;
- common constants such as prompt size and label diameter;
- split or parameter definitions that should stay aligned across experiments.

The rule of thumb in this repository is simple: if a constant needs to remain stable across multiple experiments, keep it in the shared config rather than duplicating it in each notebook.

## Experiment Overview

| Experiment | Goal | Dataset | Main deliverables |
| --- | --- | --- | --- |
| `exp1_empiar10988_ribo` | Validate the workflow on real cryo-ET ribosomes | EMPIAR-10988 | Fine-tuned checkpoint, prompt embeddings, proof-of-concept evaluation |
| `exp2_umusynth_thy` | Build a stable single-class synthetic baseline | UMU synthetic thyroglobulin | Stable 20/5 split, fixed prompts, synthetic-domain evaluation |
| `exp3_ppicker_limits` | Measure data efficiency and single vs multi prompt inference | UMU synthetic thyroglobulin | Increment study, prompt-selection logic, multi-prompt files, learning curves |
| `exp4_ppicker_rotations` | Study prompt robustness, orientation, and failure modes | UMU synthetic thyroglobulin | Rotation-diverse prompt population, prompt-level metrics, robustness figures |

## Experiment Details

### Experiment 1: `exp1_empiar10988_ribo`

Focus:

- establish a real-data proof of concept on cytoplasmic ribosomes;
- compare prompt-only inference, threshold tuning, and fine-tuning.

Main notebooks:

- `exp1_empiar10988_ribo_ppicker_finetuning.ipynb`
- `exp1_empiar10988_ribo_ppicker_inference.ipynb`

Main scripts:

- `scripts/empiar10988_fine_tuning.py`
- `scripts/empiar10988_inference.py`

Important characteristics:

- training tomogram: `TS_029`;
- validation tomogram: `TS_030`;
- centered crop with `crop_delta = 64`;
- cluster-based post-processing is part of the experiment, not an afterthought.

### Experiment 2: `exp2_umusynth_thy`

Focus:

- measure how strongly the pretrained detector fails under domain shift;
- construct the stable thyroglobulin benchmark used later by `EXP3` and `EXP4`.

Main notebooks:

- `exp2_umusynth_thy_ppicker_finetuning.ipynb`
- `exp2_umusynth_thy_ppicker_inference.ipynb`

Main scripts:

- `scripts/umusynth_fine_tuning.py`
- `scripts/umusynth_inference.py`
- `scripts/update_csv_paths.py`

Important characteristics:

- label of interest: `thyroglobulin = 7`;
- 25 tomograms total, with a fixed 20/5 train/validation split;
- coordinates are converted from angstroms to voxels before training and evaluation;
- prompt crops are extracted at `37 x 37 x 37`.

### Experiment 3: `exp3_ppicker_limits`

Focus:

- study how performance scales with the number of training tomograms;
- compare single-prompt and multi-prompt inference after fine-tuning.

Main notebooks:

- `exp3_ppicker_limits.ipynb`
- `exp3_ppicker_limits_inference.ipynb`

Main scripts:

- `scripts/umusynth_fine_tuning.py`
- `scripts/umusynth_inference.py`
- `scripts/update_csv_paths.py`

Important characteristics:

- increment schedule: `1, 2, 4, 8, 12, 16, 20` training tomograms;
- prompt selection is no longer informal: a quality score is used to choose the best single prompt;
- a multi-prompt representation is built by averaging multiple prompt embeddings;
- the same validation split as `EXP2` is kept fixed.

Key output files:

- `results/exp3_ppicker_limits/prompts/single_instance_prompt.json`
- `results/exp3_ppicker_limits/prompts/multi_instance_prompt_n10.json`
- increment-specific checkpoints and inference outputs.

### Experiment 4: `exp4_ppicker_rotations`

Focus:

- move from average performance to prompt-level robustness;
- test whether prompt failures are explained by rotation, acquisition anisotropy, or prompt quality;
- expand prompt analysis from a small set to a large prompt population.

Main notebooks:

- `exp4_ppicker_rotations.ipynb`
- `exp4_large_prompt_population_inference.ipynb`
- `exp4_inference_prompt_mix.ipynb`
- `exp4_underperforming_prompts_analysis.ipynb`

Main scripts:

- `rotational_issues_analysis.py`
- `scripts/extract_prompt_subvolumes_3d.py`
- `scripts/umusynth_inference.py`
- `scripts/umusynth_inference_followup.py`
- `scripts/umusynth_inference_large.py`

Important characteristics:

- prompt candidates are filtered by subtomogram quality;
- quaternion distances are used to enforce diversity in orientation space;
- the final large analysis is symmetry-aware for thyroglobulin through `SO(3)/C2`;
- prompt performance is evaluated on held-out validation tomograms with fixed matching rules.

Key output files:

- large prompt JSON bundles in `results/exp4_ppicker_rotations/`;
- prompt-level tables and figures under `results/exp4/underperforming_analysis/`;
- exported figures later reused in the thesis appendix and results chapter.

## How to Run the Experiments

The working pattern is similar across folders:

1. run the notebook cells that prepare prompts, coordinates, and configs;
2. launch the long-run script from the environment expected by ProPicker;
3. use the analysis notebook to aggregate metrics and create plots.

Typical long-run command pattern:

```bash
conda activate deepetpicker
cd /path/to/cryoet-particle-picking/tools/ProPicker

python ../../experiments/exp1_empiar10988_ribo/scripts/empiar10988_fine_tuning.py
python ../../experiments/exp2_umusynth_thy/scripts/umusynth_fine_tuning.py
python ../../experiments/exp3_ppicker_limits/scripts/umusynth_inference.py
python ../../experiments/exp4_ppicker_rotations/scripts/umusynth_inference_large.py
```

If a script fails because prompts or configs are missing, it usually means the notebook-generated setup stage was skipped.

## Reproducibility Rules

- keep the UMU validation split unchanged between `EXP2`, `EXP3`, and `EXP4`;
- generate prompt JSON files before running the long-run scripts;
- keep source notebooks and scripts under `experiments/`, and write generated artifacts to `results/`;
- preserve stable naming such as `fixed_prompts_*.json`, `single_instance_prompt.json`, and `multi_instance_prompt_n10.json`;
- export MRC localization maps when qualitative 3D inspection is part of the analysis.

## Adding New Experiments

When a new experiment is added, follow the repository convention:

1. create `experiments/exp<N>_<dataset>_<goal>/`;
2. add at least one notebook for preprocessing or analysis;
3. add a `scripts/` folder for long runs;
4. register any new output paths in `paths.py`;
5. update this file so the experiment is documented alongside the others.
