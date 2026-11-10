# CW2 Evidence File — NB6057CEM
## Compiled from CW1 Project: EuroSAT Land Use Classification

### Model Comparison (from metrics/results.csv and notebooks)

| Model | Val F1 (mean±std) | Test F1 (mean±std) | Test Accuracy | Params | Inference ms (batch=64) |
|---|---|---|---|---|---|
| Baseline (HOG+LR) | 0.7299 (single) | 0.7324 (single) | 74.05% | 17,880 | 0.32 (CPU) |
| MLP (PyTorch) | 0.5057 ± 0.0239 | 0.5100 ± 0.0250 | ~50% | ~6,425,610 | 0.16 (cuda) |
| CNN (PyTorch) | 0.9068 ± 0.0148 | 0.9103 ± 0.0129 | ~92.91% | ~94,730 | 0.73 (cuda) |

### Per-Seed Results
- MLP Seed 0: Val=0.4735, Test=0.4746
- MLP Seed 1: Val=0.5306, Test=0.5268
- MLP Seed 2: Val=0.5131, Test=0.5285
- CNN Seed 0: Val=0.9262, Test=0.9263
- CNN Seed 1: Val=0.9040, Test=0.9100
- CNN Seed 2: Val=0.8902, Test=0.8946

### Ablation (dropout removed, CNN)
- Ablation mean val F1 = 0.9141 ± 0.0082
- Full CNN mean val F1 = 0.9068 ± 0.0148
- Delta = +0.0073 (removing dropout slightly improved val F1)

### NumPy MLP (T2)
- 4-class subset, 16x16 grayscale, 256 features, hidden=32, tanh, softmax
- Initial loss: 1.4459 (expected ln(4)=1.3863)
- Final loss: 0.5041 (200 epochs, lr=0.1, batch=64)
- Gradient check: 12 entries, all rel_err < 1e-10, eps=1e-5
- Overfit test: 10 samples, 2000 epochs, final loss=0.001095

### Stress Test (Gaussian noise on CNN best seed)
- sigma=0.00 -> F1=0.9263
- sigma=0.05 -> F1=0.8542
- sigma=0.10 -> F1=0.5822
- sigma=0.20 -> F1=0.1656

### Error Analysis
- Total misclassified: 288 / 4,050 (7.1%)
- Confusions: River<->Highway, Pasture<->HerbaceousVegetation

### Split & Preprocessing
- 70/15/15 stratified, seed=42
- Train=18,900, Val=4,050, Test=4,050
- 0 leakage overlaps
- Mean (train-only): [0.2167, 0.2167, 0.2167]
- Std (train-only): [0.2201, 0.1861, 0.1872]

### Figures
- figures/class_distribution.png
- figures/sample_grid.png
- figures/pixel_histograms.png
- figures/t2_loss_curve.png
- figures/t2_overfit_diagnosis.png
- figures/confusion_baseline_val.png
- figures/mlp_loss_curves.png
- figures/confusion_mlp_test.png
- figures/cnn_loss_curves.png
- figures/confusion_cnn_test.png
- figures/error_analysis_cnn.png
- figures/stress_test.png

### MISSING
- Exact GPU type in Colab (likely T4 but not explicitly logged)
- Pinned dependency versions from pip freeze
- MLP and CNN inference_ms not recorded in results.csv (shows 0); measured in NB08 as 0.16 and 0.73 ms

