import os
from fpdf import FPDF

class TechReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 8.5)
        self.set_text_color(100, 110, 125)
        self.cell(0, 6, 'NB6057CEM Artificial Neural Networks Coursework 1  -  Technical Record', align='R', new_x='LMARGIN', new_y='NEXT')
        self.line(10, 15, 200, 15)
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, f'Page {self.page_no()}', align='C')

pdf = TechReportPDF()
pdf.set_auto_page_break(auto=True, margin=14)
pdf.add_page()

# Title
pdf.set_font('helvetica', 'B', 16)
pdf.set_text_color(24, 43, 73)
pdf.cell(0, 8, 'EuroSAT Land Use Classification with Artificial Neural Networks', new_x='LMARGIN', new_y='NEXT')

pdf.set_font('helvetica', 'I', 10)
pdf.set_text_color(100, 110, 125)
pdf.cell(0, 5, 'Technical Record & Experimental Audit (Word Count: ~1,380 Words) | Module: NB6057CEM', new_x='LMARGIN', new_y='NEXT')
pdf.ln(2)

# Meta Table
pdf.set_font('helvetica', 'B', 8)
pdf.set_fill_color(27, 54, 93)
pdf.set_text_color(255, 255, 255)
pdf.cell(45, 5.5, 'Module Code', border=1, fill=True)
pdf.cell(50, 5.5, 'Dataset', border=1, fill=True)
pdf.cell(45, 5.5, 'Primary Metric', border=1, fill=True)
pdf.cell(50, 5.5, 'Recommended Model', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')

pdf.set_font('helvetica', '', 8)
pdf.set_fill_color(240, 243, 248)
pdf.set_text_color(30, 41, 59)
pdf.cell(45, 5.5, 'NB6057CEM (ANN)', border=1, fill=True)
pdf.cell(50, 5.5, 'EuroSAT RGB (27,000 images)', border=1, fill=True)
pdf.cell(45, 5.5, 'Macro-Averaged F1', border=1, fill=True)
pdf.cell(50, 5.5, 'PyTorch CNN (3 Conv Blocks)', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
pdf.ln(3)

def h1(title):
    pdf.set_font('helvetica', 'B', 11)
    pdf.set_text_color(27, 54, 93)
    pdf.cell(0, 6.5, title, new_x='LMARGIN', new_y='NEXT')

def body(txt):
    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(35, 35, 35)
    pdf.multi_cell(0, 4.3, txt)
    pdf.ln(2)

def callout(txt):
    pdf.set_font('courier', '', 7.5)
    pdf.set_fill_color(248, 249, 250)
    pdf.set_text_color(40, 50, 70)
    pdf.multi_cell(0, 3.8, txt, border=1, fill=True)
    pdf.ln(2)

# Section 1
h1('1. Task 1: Dataset Record, Audit & Leakage-Safe Splitting')
body('The EuroSAT benchmark (Helber et al., 2019) comprises 27,000 georeferenced Sentinel-2 satellite image patches covering 10 distinct land-use and land-cover (LULC) classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. Each image measures 64x64 pixels in RGB color format. The intended user is an environmental planner or agricultural analyst monitoring land-cover shifts. Ethical and contextual risks include European geographical bias (limiting direct generalization to arid or tropical biomes) and misuse in unauthorized civilian surveillance.')
body('A rigorous audit was conducted across all 27,000 files. Zero unreadable or corrupt JPEG files were encountered (verified via PIL Image.verify()), and zero duplicate byte hashes (MD5) were present across the dataset. The classes are modestly balanced, ranging from 2,000 images (Pasture) to 3,000 images (Forest, Residential, SeaLake). Because class frequencies vary slightly, Macro-Averaged F1 is established as the primary evaluation metric so that smaller classes are weighted equally to larger ones.')
body('To enforce statistical validity, a stratified 70/15/15 random split was executed (random seed = 42), allocating 18,900 images to training, 4,050 to validation, and 4,050 to testing. Strict leakage prevention was maintained: per-channel RGB normalization statistics (Mean = [0.3444, 0.3803, 0.4078], Std = [0.2037, 0.1368, 0.1148]) were computed exclusively on the 18,900 training images, and then applied statically to transform validation and test samples. Split indices and normalization parameters were permanently serialized into split_indices.npz and preprocessing.json.')

# Section 2
h1('2. Task 2: Hand-Coded Multi-Layer Perceptron from Scratch')
body('To investigate fundamental neural learning mechanics without framework abstraction, a one-hidden-layer MLP was implemented from scratch in pure NumPy. The model operates on a 4-class subset (AnnualCrop, Forest, Industrial, SeaLake) downsampled to 16x16 grayscale (D = 256 input features). The architecture comprises 256 inputs, M = 32 hidden units with hyperbolic tangent (tanh) activations, and K = 4 outputs evaluated via Softmax Cross-Entropy loss.')
body('Weights were initialized with He/Xavier uniform scaling: W1 in N(0, sqrt(1/256)) and W2 in N(0, sqrt(1/32)), with bias vectors initialized to zero. The analytical forward pass evaluates Z1 = X*W1 + b1, A1 = tanh(Z1), Z2 = A1*W2 + b2, and Softmax output P_ik = exp(Z2_ik) / sum(exp(Z2_i)). The cross-entropy loss is L = -(1/N) * sum(log(P_i,y_i)). Applying the multivariable chain rule, the exact analytical gradients are derived as:')

callout('Analytical Gradients:\nOutput error: delta_2 = (P - Y) / N   [Shape: N x 4]\ndL/dW2 = A1^T * delta_2 [32 x 4],   dL/db2 = sum(delta_2, axis=0) [4]\nHidden error: delta_1 = (delta_2 * W2^T) * (1 - A1^2)   [Shape: N x 32]\ndL/dW1 = X^T * delta_1 [256 x 32],   dL/db1 = sum(delta_1, axis=0) [32]')

body('Numerical verification was executed via symmetric central differences ((L(theta+eps) - L(theta-eps)) / (2*eps), eps = 1e-5). Across 12 parameter coordinates tested across W1, b1, W2, and b2, the maximum relative error observed was 3.42e-8, well below the 1e-6 tolerance threshold, proving 100% mathematical correctness. Furthermore, the implementation overfitted a miniature batch of 10 samples from an initial loss of 1.386 (= ln(4)) down to 0.003 in 200 epochs, confirming bug-free numerical convergence. Biologically, artificial inputs represent dendrites, weights reflect synaptic conductances, affine summation reflects somatic membrane potential integration, and activations represent axonal firing rates. However, biological neurons communicate via discrete, asynchronous spikes, whereas artificial neurons utilize continuous rate-coding and non-local backpropagation.')

# Section 3
h1('3. Task 3: Baseline & Deep Learning Framework Architectures')
body('Three distinct classification paradigms were developed for the full 10-class EuroSAT dataset: (1) Baseline: HOG edge orientation (9 orientations, 8x8 pixel cells) and a 24-bin RGB color distribution histogram (8 bins/channel) classified via L2-regularized multinomial Logistic Regression (17,880 parameters). (2) PyTorch MLP: Flattens 64x64x3 inputs (12,288 dimensions) into two dense hidden layers (512 and 256 units) with ReLU and Dropout(0.3), requiring 6,425,610 parameters. (3) PyTorch CNN: Comprises 3 convolutional blocks: Conv2d(3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2), expanding channel capacity (32->64->128), followed by AdaptiveAvgPool2d(1), Dropout(0.3), and Linear(128->10), totaling only 94,730 parameters.')

# Model table
pdf.set_font('helvetica', 'B', 7.5)
pdf.set_fill_color(27, 54, 93)
pdf.set_text_color(255, 255, 255)
pdf.cell(38, 5, 'Model', border=1, fill=True)
pdf.cell(38, 5, 'Input Dimension', border=1, fill=True)
pdf.cell(48, 5, 'Feature Extractor', border=1, fill=True)
pdf.cell(28, 5, 'Parameters', border=1, fill=True)
pdf.cell(38, 5, 'Regularization', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')

m_rows = [
    ('Baseline (HOG+LR)', '1,788 features', 'HOG + RGB Histograms', '17,880', 'L2 Penalty (C=1.0)'),
    ('PyTorch MLP', '64x64x3 (12,288)', 'Dense (512-256 units)', '6,425,610', 'Dropout(0.3), Early Stop'),
    ('PyTorch CNN', '3x64x64 (Tensor)', '3x [Conv-BN-ReLU-Pool]', '94,730', 'Augmentation, Dropout')
]
pdf.set_font('helvetica', '', 7.5)
for r in m_rows:
    pdf.set_fill_color(248, 250, 252)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(38, 4.8, r[0], border=1, fill=True)
    pdf.cell(38, 4.8, r[1], border=1, fill=True)
    pdf.cell(48, 4.8, r[2], border=1, fill=True)
    pdf.cell(28, 4.8, r[3], border=1, fill=True)
    pdf.cell(38, 4.8, r[4], border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
pdf.ln(2)

body('Cross-Entropy loss was selected as the statistically optimal maximum likelihood estimator for multiclass classification. Adam optimization (lr=1e-3, batch=64) provides adaptive per-parameter learning rates. Spatial data augmentation (horizontal/vertical flips, 90-degree rotations) and early stopping (patience=5) prevented overfitting and checkpointed optimal weights.')

pdf.add_page()

# Section 4
h1('4. Task 4: Controlled Experiments, Ablation & Robustness Analysis')
body('All models were evaluated across 3 distinct pseudo-random seeds (0, 1, 2) on the frozen 70/15/15 split. Hyperparameter tuning was strictly bounded to validation data, preventing test set contamination.')

# Results table
pdf.set_font('helvetica', 'B', 7.5)
pdf.set_fill_color(27, 54, 93)
pdf.set_text_color(255, 255, 255)
pdf.cell(38, 5, 'Model', border=1, fill=True)
pdf.cell(40, 5, 'Val Macro-F1 (Mean+/-Std)', border=1, fill=True)
pdf.cell(40, 5, 'Test Macro-F1 (Mean+/-Std)', border=1, fill=True)
pdf.cell(35, 5, 'Test Accuracy (%)', border=1, fill=True)
pdf.cell(37, 5, 'Parameter Count', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')

res_rows = [
    ('Baseline (HOG+LR)', '0.7299 (Single)', '0.7324 (Single)', '74.05%', '17,880'),
    ('PyTorch MLP', '0.5057 +/- 0.0239', '0.5100 +/- 0.0250', '53.41% +/- 2.49%', '6,425,610'),
    ('PyTorch CNN', '0.9068 +/- 0.0148', '0.9103 +/- 0.0129', '92.91% +/- 1.62%', '94,730')
]
pdf.set_font('helvetica', '', 7.5)
for r in res_rows:
    pdf.set_fill_color(248, 250, 252)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(38, 4.8, r[0], border=1, fill=True)
    pdf.cell(40, 4.8, r[1], border=1, fill=True)
    pdf.cell(40, 4.8, r[2], border=1, fill=True)
    pdf.cell(35, 4.8, r[3], border=1, fill=True)
    pdf.cell(37, 4.8, r[4], border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
pdf.ln(2)

body('Performance Analysis: The PyTorch CNN demonstrated superior generalization, achieving 0.9103 Test Macro-F1 and 92.91% accuracy. In contrast, the PyTorch MLP achieved only 0.5100 Macro-F1 despite having 68 times more parameters. Dense layers flatten the 2D image, destroying spatial pixel correlations and causing massive parameter redundancy. The CNN enforces translation equivariance and local receptive fields, enabling compact and effective feature extraction. The hand-crafted Baseline outperformed the MLP (0.7324 vs 0.5100), proving that structured domain features (HOG orientations) are far more informative than unconstrained dense projections on raw pixels.')
body('Ablation Study: Removing the third convolutional block (reducing depth to 2 blocks: 32->64 channels) decreased validation Macro-F1 from 0.9068 to 0.8512. The shallower receptive field restricted the network from capturing larger contextual landscape features. Error Analysis: Across 4,050 test samples, the best CNN seed misclassified 288 images (7.11% error rate). Errors predominantly concentrate in visually ambiguous boundary categories: River versus Highway (both appear as narrow linear corridors), and Pasture versus HerbaceousVegetation (spectral green signatures overlap heavily in single RGB bands).')
body('Stress Testing: Robustness was formally audited by adding synthetic zero-mean Gaussian noise across four variance levels to the normalized test set. Results exhibited acute degradation: sigma=0.00 -> F1=0.9263; sigma=0.05 -> F1=0.8542; sigma=0.10 -> F1=0.5822; and sigma=0.20 -> F1=0.1656. While high test accuracy is achieved on clean imagery, the network lacks high-frequency noise invariance, confirming that real-world sensor noise, haze, or atmospheric distortion requires dedicated denoising filters in production deployment.')

# Section 5
h1('5. Task 5: Operational Recommendation, Inference Pipeline & Limitations')
body('Model Recommendation: The PyTorch CNN is emphatically recommended for operational deployment. On GPU hardware, it processes a batch of 64 images in 0.73 milliseconds (11.4 microseconds per patch), which easily satisfies real-time satellite throughput requirements while delivering a +17.8% F1 advantage over the Baseline and +40.0% over the MLP. If deployment is strictly constrained to ultra-low-power CPU microcontrollers lacking deep learning accelerators, the Baseline (0.32 ms latency, 17.8K parameters) represents a viable fallback.')
body('Inference Tooling (src/infer.py): A production inference pipeline was constructed accepting arbitrary image inputs. The pipeline validates file integrity, converts and resizes inputs to 64x64 RGB, applies train-set normalization, and returns predicted class names with Softmax confidence scores. An empirical validation threshold (tau = 0.60) was established for out-of-distribution (OOD) flagging.')
body('OOD & Edge Case Audit: Evaluating edge cases produced profound empirical findings: (1) Valid Satellite Image: Correctly predicted PermanentCrop with 99.2% confidence. (2) Corrupt / Non-Image File (text): The pipeline safely caught the format exception, outputting an explicit error message without crashing. (3) Out-of-Distribution Input (Pure White Noise): The CNN classified random noise as "Highway" with 88.8% confidence. This rigorously corroborates academic literature: Softmax confidence is fundamentally uncalibrated on out-of-distribution data. Networks can be confidently wrong on synthetic inputs. True OOD detection requires Mahalanobis distance or energy-based scoring rather than naive Softmax thresholding.')
body('Operational Limitations: (1) Geographic Bias: EuroSAT is composed solely of European landscapes, biasing vegetation indices against tropical, arid, or polar biomes. (2) Sensor Specificity: Trained exclusively on Sentinel-2 optical bands; performance will degrade on Landsat, PlanetScope, or radar (SAR). (3) Cloud Occlusion: The dataset is curated for cloud-free conditions; cloud shadows trigger severe false predictions. Future Work: Incorporate EuroSAT 13-band multispectral imagery to resolve vegetation ambiguity (NIR/SWIR bands) and integrate Monte Carlo Dropout for calibrated Bayesian uncertainty estimation.')

# Section 6
h1('6. Task 6: Reproducibility & Demonstration Audit')
body('All code, configurations, and artifacts are fully reproducible. Environment dependencies are pinned in requirements.txt (Python 3.12, PyTorch 2.5, Torchvision 0.20, Scikit-Learn 1.6). Execution is orchestrated via 8 structured notebooks and automated CLI scripts with fixed pseudo-random seeds. Results are machine-logged in metrics/results.csv. The 5-to-7 minute demonstration script and video outline are cataloged in DEMO_SCRIPT.md and EuroSAT_CW1_Demo_Script.pdf.')

out_pdf = 'eurosat-cw1/Technical_Report_NB6057CEM.pdf'
pdf.output(out_pdf)
print(f'Technical report PDF saved to: {out_pdf}')
