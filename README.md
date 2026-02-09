# DcSE

## Reproducibility

This section provides the exact information needed to reproduce at least one main table entry end-to-end.

### Commit hash

- Git commit: 0a111b8388e96fb1e97893d431c18f670bd226af

### Environment

- OS: Windows (verified in this repo state)
- Python: Use the environment below
- Full package list: `src/requirments.txt`

Create and install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r src\requirments.txt
```

### Data preprocessing (splits + negatives)

Input CSVs (already provided):

- `src/data/datasets/combined_shuffled_human.csv`
- `src/data/datasets/combined_shuffled_mouse.csv`

These CSVs already contain labeled positives/negatives. The split script recreates 10 non-overlapping train/test splits with a fixed seed schedule.

1) Create train/test splits (edit the `species` variable in `src/data/datasets/split.py` and run once per species):

```bash
cd src\data\datasets
python split.py
```

Outputs:

- `train_{species}_split_0..9.csv`
- `test_{species}_split_0..9.csv`

2) Convert CSV splits into `.pt` datasets (edit the `species` variable in `src/data_pt.py` and run once per species):

```bash
cd src
python data_pt.py
```

Outputs:

- `src/data/datasets/datasets_{species}_split_0..9.pt`

### Fixed random seeds

- Split generation: base seed 42, incremented for each split (see `src/data/datasets/split.py`).
- Cross-validation: `StratifiedKFold(shuffle=True, random_state=42)` (see `src/train.py`).

### Training command

Runs 10 splits with 5-fold CV per split (human + mouse), using default hyperparameters and fixed random seed settings.

```bash
cd src
python train.py
```

Checkpoints and logs are written to:

- `src/Demo_split_{split_id}/{species}_model/fold_{k}/best.pth`
- `src/Demo_split_{split_id}/{species}_model/fold_{k}/final.pth`

### Released checkpoints

We provide released checkpoints for the main results in:

- `src/DcSEResult/True_model_split_0..9/{species}_model/fold_1..5/final.pth`

### Evaluation and inference (end-to-end)

To reproduce one main table entry (human, 10 splits, 5-fold ensemble), run:

```bash
cd src
python test_demo.py
```

This script loads the released checkpoints from `src/DcSEResult/True_model_split_*`, evaluates each split, and writes the per-split metrics plus aggregate mean/std.

