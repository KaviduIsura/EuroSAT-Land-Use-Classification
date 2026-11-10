# NB6057CEM Artificial Neural Networks — Project Links & Submission Manifest
**Module:** NB6057CEM Artificial Neural Networks  
**Coursework:** Coursework 1 (Practical) & Coursework 2 (Theory)  
**Project Title:** EuroSAT Land Use Classification with Artificial Neural Networks  
**Student ID:** [I WILL ADD]  

---

## 🔗 Official Submission Links

| Deliverable | Official Link | Description |
| :--- | :--- | :--- |
| **Demonstration Video** | [https://drive.google.com/file/d/1RjGWZ38y6NRWB9XuY0VWC1I8HsutZUzg/view?usp=sharing](https://drive.google.com/file/d/1RjGWZ38y6NRWB9XuY0VWC1I8HsutZUzg/view?usp=sharing) | 5–7 minute video walkthrough covering all 8 notebooks, code explanation, and live inference tool |
| **GitHub Repository** | [https://github.com/KaviduIsura/EuroSAT-Land-Use-Classification](https://github.com/KaviduIsura/EuroSAT-Land-Use-Classification) | Complete source code, 8 executable Jupyter notebooks, configs, metrics, and documentation |

---

## 📦 Submission Package Structure (CW1 Practical ZIP)

```
eurosat-cw1.zip
├── README.md                           <- Full reproducibility guide & smoke test
├── requirements.txt                    <- Pinned dependencies
├── SUBMISSION_LINKS.pdf                <- Official links document
├── SUBMISSION_LINKS.docx               <- Editable links document
├── Technical_Report_NB6057CEM.docx     <- Formal 1,500-word Technical Report (DOCX)
├── Technical_Report_NB6057CEM.pdf      <- Formal 1,500-word Technical Report (PDF)
├── DEMO_SCRIPT.md                      <- Spoken demonstration script
├── EuroSAT_CW1_Demo_Script.pdf         <- Printable demonstration guide
├── CELL_BY_CELL_DEMO_SCRIPT.md         <- Cell-by-cell walkthrough for notebooks
├── configs/
│   ├── baseline.yaml                   <- HOG + Logistic Regression configuration
│   ├── mlp.yaml                        <- PyTorch MLP configuration
│   ├── cnn.yaml                        <- PyTorch CNN configuration
│   └── cnn_ablation.yaml               <- Ablation study configuration
├── splits/
│   ├── split_indices.npz               <- Stratified 70/15/15 split indices (seed=42)
│   └── preprocessing.json              <- Per-channel RGB mean & std (train only)
├── checkpoints/
│   ├── baseline_hog_lr.pkl             <- Trained Logistic Regression model
│   ├── cnn_seed0.pt                    <- PyTorch CNN best checkpoint (Seed 0)
│   ├── cnn_seed1.pt                    <- PyTorch CNN checkpoint (Seed 1)
│   ├── cnn_seed2.pt                    <- PyTorch CNN checkpoint (Seed 2)
│   ├── mlp_seed0.pt                    <- PyTorch MLP checkpoint (Seed 0)
│   ├── mlp_seed1.pt                    <- PyTorch MLP checkpoint (Seed 1)
│   └── mlp_seed2.pt                    <- PyTorch MLP checkpoint (Seed 2)
├── metrics/
│   ├── results.csv                     <- Machine-readable master evaluation table
│   ├── baseline_result.json            <- Baseline detailed performance
│   ├── mlp_results.json                <- MLP 3-seed evaluation results
│   ├── cnn_results.json                <- CNN 3-seed evaluation results
│   ├── cnn_ablation.json               <- Dropout ablation results
│   └── stress_test.json                <- Gaussian noise stress test results
├── figures/
│   ├── class_distribution.png          <- Class balance audit bar chart
│   ├── sample_grid.png                 <- 3 sample images per class
│   ├── pixel_histograms.png            <- RGB pixel distribution histograms
│   ├── t2_loss_curve.png               <- Hand-coded NumPy MLP training curve
│   ├── t2_overfit_diagnosis.png        <- 10-sample overfit diagnosis curve
│   ├── confusion_baseline_val.png      <- Baseline validation confusion matrix
│   ├── mlp_loss_curves.png             <- MLP 3-seed training & validation loss
│   ├── confusion_mlp_test.png          <- MLP test set confusion matrix
│   ├── cnn_loss_curves.png             <- CNN 3-seed loss & accuracy curves
│   ├── confusion_cnn_test.png          <- CNN test set confusion matrix
│   ├── error_analysis_cnn.png          <- Error analysis: misclassified images
│   ├── stress_test.png                 <- Gaussian noise robustness curve
│   └── inference_demo_valid.png        <- Inference tool sample prediction
├── src/
│   ├── data.py                         <- Dataset loading and augmentation
│   ├── models.py                       <- Baseline, MLP, and CNN architectures
│   ├── train.py                        <- Training functions and early stopping
│   ├── evaluate.py                     <- Evaluation and confusion matrix tools
│   └── infer.py                        <- Production inference pipeline
├── numpy_mlp/
│   ├── mlp_numpy.py                    <- Hand-coded MLP from scratch (pure NumPy)
│   ├── gradcheck.py                    <- Finite-difference gradient verification
│   └── q1_check.py                     <- CW2 Q1 numerical verification script
└── notebooks/
    ├── 01_data_download.ipynb          <- Data acquisition and Drive backup
    ├── 02_eda_split.ipynb              <- Data audit, plots, and stratified split
    ├── 03_numpy_mlp.ipynb              <- NumPy MLP from first principles
    ├── 04_baseline_hog_lr.ipynb        <- HOG + Logistic Regression baseline
    ├── 05_mlp_pytorch.ipynb            <- PyTorch MLP training across 3 seeds
    ├── 06_cnn_pytorch.ipynb            <- PyTorch CNN training and ablation
    ├── 07_experiments_evaluation.ipynb <- Comprehensive evaluation & stress test
    └── 08_recommendation_inference.ipynb <- Model recommendation & inference demo
```
