# EuroSAT Land-Use Classification — CW1

**Module:** NB6057CEM — Artificial Neural Networks  
**Task:** Classify 64×64 Sentinel-2 satellite patches into 10 land-use classes.  
**Dataset:** EuroSAT RGB (Helber et al., 2019) — 27,000 images, 10 classes.

---

## Project Summary

This project implements and evaluates three models for satellite image classification:

| Model | Type | Framework | Params |
|-------|------|-----------|--------|
| HOG + Logistic Regression | Classical ML baseline | scikit-learn | ~minimal |
| MLP (2 hidden layers) | Neural network | PyTorch | ~6.4M |
| CNN (3 conv blocks) | Neural network | PyTorch | ~94K |

A fourth model — a one-hidden-layer MLP in pure NumPy — is built as a teaching exercise (T2) demonstrating manual backpropagation on a reduced 4-class subset.

---

## Folder Map

```
eurosat-cw1/
├── README.md
├── requirements.txt             ← pinned dependencies
├── .gitignore
├── configs/
│   ├── baseline.yaml            ← HOG + LR config
│   ├── mlp.yaml                 ← PyTorch MLP config
│   ├── cnn.yaml                 ← PyTorch CNN config
│   └── cnn_ablation.yaml        ← T4c ablation (dropout removed)
├── splits/
│   ├── split_indices.npz        ← 70/15/15 stratified split (seed=42)
│   └── preprocessing.json       ← per-channel mean/std (train only)
├── src/
│   ├── data.py                  ← EuroSATDataset, DataLoaders, augmentation
│   ├── models.py                ← extract_features(), make_mlp(), make_cnn()
│   ├── train.py                 ← training loop with early stopping
│   ├── evaluate.py              ← metrics, confusion matrix, stress test
│   └── infer.py                 ← CLI inference pipeline (T5b)
├── numpy_mlp/
│   ├── mlp_numpy.py             ← T2: hand-coded MLP (no autograd)
│   └── gradcheck.py             ← T2c: numerical gradient verification
├── checkpoints/                 ← .pt files (not in git; stored on Drive)
├── metrics/
│   ├── results.csv              ← master results table (T6b)
│   └── *.json                   ← per-model detailed results
├── figures/                     ← all saved plots
└── notebooks/
    ├── 01_data_download.ipynb
    ├── 02_eda_split.ipynb
    ├── 03_numpy_mlp.ipynb
    ├── 04_baseline_hog_lr.ipynb
    ├── 05_mlp_pytorch.ipynb
    ├── 06_cnn_pytorch.ipynb
    ├── 07_experiments_evaluation.ipynb
    └── 08_recommendation_inference.ipynb
```

---

## Setup

**Python version:** 3.10 (tested on Google Colab)

```bash
pip install -r requirements.txt
```

**Data acquisition** (run inside Colab — data is NOT redistributed in this repo):

```python
from torchvision.datasets import EuroSAT
ds = EuroSAT(root='/content/data', download=True)
# Then copy to Drive: see notebook 01_data_download.ipynb
```

---

## Running Each Stage (in order)

> Run all notebooks in **Google Colab** in the order below.  
> Clone the repo first: `!git clone https://github.com/<your-username>/eurosat-cw1.git`

| Step | Notebook | Task | Outputs |
|------|----------|------|---------|
| 1 | `01_data_download.ipynb` | Download EuroSAT to Drive | Drive copy, metadata |
| 2 | `02_eda_split.ipynb` | Audit + plots + split | `splits/`, `figures/class_dist*.png` |
| 3 | `03_numpy_mlp.ipynb` | NumPy MLP (T2) | loss curve, gradcheck table |
| 4 | `04_baseline_hog_lr.ipynb` | HOG+LR baseline (T3a) | `checkpoints/baseline_hog_lr.pkl` |
| 5 | `05_mlp_pytorch.ipynb` | PyTorch MLP 3 seeds (T3b) | `checkpoints/mlp_seed*.pt` |
| 6 | `06_cnn_pytorch.ipynb` | PyTorch CNN 3 seeds + ablation (T3b, T4c) | `checkpoints/cnn_seed*.pt` |
| 7 | `07_experiments_evaluation.ipynb` | Metrics, errors, stress test (T4) | `metrics/results.csv` |
| 8 | `08_recommendation_inference.ipynb` | Recommendation + inference demo (T5) | `figures/inference_demo_*.png` |

---

## Seeds, Configs and Checkpoints

| Model | Config | Seeds | Checkpoint |
|-------|--------|-------|-----------|
| Baseline (HOG+LR) | `configs/baseline.yaml` | 42 | `checkpoints/baseline_hog_lr.pkl` |
| PyTorch MLP | `configs/mlp.yaml` | 0, 1, 2 | `checkpoints/mlp_seed{0,1,2}.pt` |
| PyTorch CNN | `configs/cnn.yaml` | 0, 1, 2 | `checkpoints/cnn_seed{0,1,2}.pt` |
| CNN Ablation | `configs/cnn_ablation.yaml` | 0, 1, 2 | `checkpoints/cnn_ablation_seed{0,1,2}.pt` |

---

## Smoke Test

Run one epoch on 50 images — should finish in under 1 minute and exit cleanly:

```python
import torch, numpy as np
from src.models import make_cnn
from src.data import EuroSATDataset, make_loaders
import json

data   = np.load('/content/drive/MyDrive/ann_cw1/eurosat_64.npz')
X, y   = data['X'][:50], data['y'][:50]
with open('splits/preprocessing.json') as f:
    prep = json.load(f)

idx = np.arange(50)
loader, _, _ = make_loaders(X, y, idx[:40], idx[40:], idx[40:],
                             prep['mean'], prep['std'], batch_size=8)

model = make_cnn(); criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
model.train()
for xb, yb in loader:
    optimizer.zero_grad()
    criterion(model(xb), yb).backward()
    optimizer.step()
print("Smoke test passed.")
```

---

## Metrics File

`metrics/results.csv` columns:

```
model, config, seed, val_macro_f1, test_macro_f1, accuracy, params, inference_ms
```

---

## Citation

Helber, P., Bischke, B., Dengel, A., Borth, D. (2019).  
*EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.*  
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

Source imagery: Sentinel-2 (Copernicus programme).  
Raw dataset not redistributed. Download via `torchvision.datasets.EuroSAT`.

---

## AI Usage Declaration

Portions of this code were written or assisted by [AI tool name, date, description].  
All code has been reviewed, understood, and can be explained line-by-line.

---

## Dependencies (pin after first Colab run)

```
# Run pip freeze in Colab and paste pinned versions here
torch==<version>
torchvision==<version>
numpy==<version>
pandas==<version>
matplotlib==<version>
scikit-learn==<version>
scikit-image==<version>
pillow==<version>
pyyaml==<version>
joblib==<version>
```

