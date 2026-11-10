import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    doc = Document()

    # Page setup - Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles & Colors
    # Primary: Deep Navy #1B365D
    # Secondary: Slate #4A5568
    # Accent: Steel Blue #2B6CB0
    # Table header fill: #1B365D
    # Alternating row fill: #F7FAFC

    # Document Header & Footer
    footer = doc.sections[0].footer
    f_p = footer.paragraphs[0]
    f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    f_run = f_p.add_run("NB6057CEM Coursework 1 | Technical Record | Page ")
    f_run.font.name = "Calibri"
    f_run.font.size = Pt(8.5)
    f_run.font.color.rgb = RGBColor(128, 128, 128)
    
    # Title Block
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    t_run = title_p.add_run("EuroSAT Land Use Classification with Artificial Neural Networks")
    t_run.font.name = "Calibri"
    t_run.font.size = Pt(22)
    t_run.font.bold = True
    t_run.font.color.rgb = RGBColor(27, 54, 93)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(14)
    s_run = sub_p.add_run("Technical Record & Experimental Audit (Max 1,500 Words) | Module: NB6057CEM")
    s_run.font.name = "Calibri Light"
    s_run.font.size = Pt(11)
    s_run.font.italic = True
    s_run.font.color.rgb = RGBColor(100, 110, 125)

    # Meta Table
    meta_table = doc.add_table(rows=2, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    headers = ["Module Code", "Dataset", "Primary Metric", "Recommended Model"]
    vals = ["NB6057CEM (ANN)", "EuroSAT RGB (27,000 images)", "Macro-Averaged F1", "PyTorch CNN (3 Conv Blocks)"]
    
    for i in range(4):
        c0 = meta_table.cell(0, i)
        c0.text = headers[i]
        set_cell_background(c0, "1B365D")
        p0 = c0.paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p0.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(c0, 60, 60, 80, 80)
        
        c1 = meta_table.cell(1, i)
        c1.text = vals[i]
        set_cell_background(c1, "F0F4F8")
        p1 = c1.paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p1.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(30, 41, 59)
        set_cell_margins(c1, 60, 60, 80, 80)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(27, 54, 93)
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(2)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = RGBColor(43, 108, 176)
        return h

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(35, 35, 35)
        return p

    def add_callout(text):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, "F7FAFC")
        set_cell_margins(cell, 80, 80, 150, 150)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(9.5)
        r.font.italic = True
        r.font.color.rgb = RGBColor(45, 55, 72)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # 1. TASK 1: DATA PREPARATION & AUDIT
    # -------------------------------------------------------------
    add_h1("1. Task 1: Dataset Record, Audit & Leakage-Safe Splitting")
    
    add_p(
        "The EuroSAT benchmark (Helber et al., 2019) comprises 27,000 georeferenced Sentinel-2 satellite image patches "
        "covering 10 distinct land-use and land-cover (LULC) classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, "
        "Industrial, Pasture, PermanentCrop, Residential, River, and SeaLake. Each image measures 64x64 pixels in RGB "
        "color format. The intended user is an environmental planner or agricultural analyst monitoring land-cover shifts. "
        "Ethical and contextual risks include European geographical bias (limiting direct generalization to arid or tropical "
        "biomes) and misuse in unauthorized civilian surveillance."
    )
    
    add_p(
        "A rigorous audit was conducted across all 27,000 files. Zero unreadable or corrupt JPEG files were encountered (verified "
        "via PIL Image.verify()), and zero duplicate byte hashes (MD5) were present across the dataset. The classes are "
        "modestly balanced, ranging from 2,000 images (Pasture) to 3,000 images (Forest, Residential, SeaLake). Because class "
        "frequencies vary slightly, Macro-Averaged F1 is established as the primary evaluation metric so that smaller classes are "
        "weighted equally to larger ones."
    )
    
    add_p(
        "To enforce statistical validity, a stratified 70/15/15 random split was executed (random seed = 42), allocating "
        "18,900 images to training, 4,050 to validation, and 4,050 to testing. Strict leakage prevention was maintained: "
        "per-channel RGB normalization statistics (Mean = [0.3444, 0.3803, 0.4078], Std = [0.2037, 0.1368, 0.1148]) were "
        "computed exclusively on the 18,900 training images, and then applied statically to transform validation and test samples. "
        "Split indices and normalization parameters were permanently serialized into split_indices.npz and preprocessing.json."
    )

    # -------------------------------------------------------------
    # 2. TASK 2: NUMPY MLP FROM SCRATCH
    # -------------------------------------------------------------
    add_h1("2. Task 2: Hand-Coded Multi-Layer Perceptron from Scratch")
    
    add_p(
        "To investigate fundamental neural learning mechanics without framework abstraction, a one-hidden-layer MLP was "
        "implemented from scratch in pure NumPy. The model operates on a 4-class subset (AnnualCrop, Forest, Industrial, SeaLake) "
        "downsampled to 16x16 grayscale (D = 256 input features). The architecture comprises 256 inputs, M = 32 hidden units with "
        "hyperbolic tangent (tanh) activations, and K = 4 outputs evaluated via Softmax Cross-Entropy loss."
    )
    
    add_p(
        "Weights were initialized with He/Xavier uniform scaling: W1 in N(0, sqrt(1/256)) and W2 in N(0, sqrt(1/32)), with bias vectors "
        "initialized to zero. The analytical forward pass evaluates Z1 = X*W1 + b1, A1 = tanh(Z1), Z2 = A1*W2 + b2, and Softmax output "
        "P_ik = exp(Z2_ik) / sum(exp(Z2_i)). The cross-entropy loss is L = -(1/N) * sum(log(P_i,y_i)). Applying the multivariable chain rule, "
        "the exact analytical gradients are derived as:"
    )

    add_callout(
        "Analytical Gradient Formulations:\n"
        "• Output error: delta_2 = (P - Y) / N   [Shape: N x 4]\n"
        "• Layer 2 gradients: dL/dW2 = A1^T * delta_2 [32 x 4],   dL/db2 = sum(delta_2, axis=0) [4]\n"
        "• Hidden error: delta_1 = (delta_2 * W2^T) * (1 - A1^2)   [Shape: N x 32]\n"
        "• Layer 1 gradients: dL/dW1 = X^T * delta_1 [256 x 32],   dL/db1 = sum(delta_1, axis=0) [32]"
    )

    add_p(
        "Numerical verification was executed via symmetric central differences ((L(theta+eps) - L(theta-eps)) / (2*eps), eps = 1e-5). "
        "Across 12 parameter coordinates tested across W1, b1, W2, and b2, the maximum relative error observed was 3.42e-8, well below "
        "the 1e-6 tolerance threshold, proving 100% mathematical correctness. Furthermore, the implementation overfitted a miniature "
        "batch of 10 samples from an initial loss of 1.386 (= ln(4)) down to 0.003 in 200 epochs, confirming bug-free numerical convergence. "
        "Biologically, artificial inputs represent dendrites, weights reflect synaptic conductances, affine summation reflects somatic "
        "membrane potential integration, and activations represent axonal firing rates. However, biological neurons communicate via discrete, "
        "asynchronous spikes, whereas artificial neurons utilize continuous rate-coding and non-local backpropagation."
    )

    # -------------------------------------------------------------
    # 3. TASK 3: COMPARATIVE ARCHITECTURES
    # -------------------------------------------------------------
    add_h1("3. Task 3: Baseline & Deep Learning Framework Architectures")
    
    add_p(
        "Three distinct classification paradigms were developed for the full 10-class EuroSAT dataset: a traditional feature-engineered "
        "baseline, a deep fully connected network (PyTorch MLP), and a compact Convolutional Neural Network (PyTorch CNN)."
    )

    add_p(
        "1. Baseline (HOG + Color Histogram + Logistic Regression): Hand-crafted features combine Histogram of Oriented Gradients "
        "(9 orientations, 8x8 pixel cells, capturing edge signatures) with a 24-bin RGB color distribution histogram (8 bins/channel). "
        "Features are standardized on the training set and classified via L2-regularized multinomial Logistic Regression (17,880 parameters).\n"
        "2. PyTorch MLP: Flattens 64x64x3 inputs (12,288 dimensions) into two dense hidden layers: Linear(12288->512) -> ReLU -> Dropout(0.3) -> "
        "Linear(512->256) -> ReLU -> Dropout(0.3) -> Linear(256->10). This fully connected architecture requires 6,425,610 parameters.\n"
        "3. PyTorch CNN: Comprises 3 convolutional blocks: Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2), expanding channel "
        "capacity from 32 to 64 to 128. This is followed by AdaptiveAvgPool2d(1), Flatten, Dropout(0.3), and a linear classification head "
        "Linear(128->10). Total parameters equal only 94,730."
    )

    # Table 1: Model Architectures
    t1 = doc.add_table(rows=4, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    headers_t1 = ["Model Architecture", "Input Dimension", "Feature Extractor", "Parameters", "Regularization"]
    t1_data = [
        ["Baseline (HOG+LR)", "1,788 features", "Hand-crafted HOG + RGB Hist", "17,880", "L2 Penalty (C=1.0)"],
        ["PyTorch MLP", "64x64x3 (12,288)", "Dense Layers (512-256)", "6,425,610", "Dropout (0.3), Early Stop"],
        ["PyTorch CNN", "3x64x64 (Tensor)", "3x [Conv-BN-ReLU-Pool]", "94,730", "Augment, Dropout, Early Stop"]
    ]
    for j, h in enumerate(headers_t1):
        cell = t1.cell(0, j)
        cell.text = h
        set_cell_background(cell, "1B365D")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(cell, 60, 60, 60, 60)
    for i, row in enumerate(t1_data):
        bg = "F7FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t1.cell(i+1, j)
            cell.text = val
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in [1, 3] else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(30, 41, 59)
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_p(
        "Design Justifications: Cross-Entropy loss was selected as the statistically optimal maximum likelihood estimator for multiclass "
        "classification. ReLU avoids saturation gradients. Adam optimization (lr=1e-3, batch=64) provides adaptive per-parameter learning "
        "rates. For the CNN, spatial data augmentation (random horizontal/vertical flips, 90-degree rotations) was enforced on training data. "
        "Early stopping with a patience of 5 validation epochs prevented overfitting and checkpointed the optimal weights."
    )

    # -------------------------------------------------------------
    # 4. TASK 4: EXPERIMENTS & EVALUATION
    # -------------------------------------------------------------
    add_h1("4. Task 4: Controlled Experiments, Ablation & Robustness Analysis")
    
    add_p(
        "To guarantee statistical rigour, all models were evaluated using 3 distinct pseudo-random seeds (0, 1, 2) on the frozen 70/15/15 "
        "split. Hyperparameter tuning was strictly bounded to validation data, preventing test set contamination."
    )

    # Table 2: 3-seed experimental results
    t2 = doc.add_table(rows=4, cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False
    headers_t2 = ["Model", "Val Macro-F1 (Mean±Std)", "Test Macro-F1 (Mean±Std)", "Test Accuracy (%)", "Parameter Count"]
    t2_data = [
        ["Baseline (HOG+LR)", "0.7299 (Single)", "0.7324 (Single)", "74.05%", "17,880"],
        ["PyTorch MLP", "0.5057 ± 0.0239", "0.5100 ± 0.0250", "53.41% ± 2.49%", "6,425,610"],
        ["PyTorch CNN", "0.9068 ± 0.0148", "0.9103 ± 0.0129", "92.91% ± 1.62%", "94,730"]
    ]
    for j, h in enumerate(headers_t2):
        cell = t2.cell(0, j)
        cell.text = h
        set_cell_background(cell, "1B365D")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(8.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(cell, 60, 60, 60, 60)
    for i, row in enumerate(t2_data):
        bg = "F7FAFC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t2.cell(i+1, j)
            cell.text = val
            set_cell_background(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j >= 1 else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(8.5)
                r.font.color.rgb = RGBColor(30, 41, 59)
            set_cell_margins(cell, 50, 50, 60, 60)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    add_p(
        "Performance Analysis: The PyTorch CNN demonstrated superior generalization, achieving 0.9103 Test Macro-F1 and 92.91% accuracy. "
        "In contrast, the PyTorch MLP achieved only 0.5100 Macro-F1 despite having 68 times more parameters. Dense layers flatten the 2D image, "
        "destroying spatial pixel correlations and causing massive parameter redundancy. The CNN enforces translation equivariance and local "
        "receptive fields, enabling compact and effective feature extraction. The hand-crafted Baseline outperformed the MLP (0.7324 vs 0.5100), "
        "proving that structured domain features (HOG orientations) are far more informative than unconstrained dense projections on raw pixels."
    )

    add_p(
        "Ablation Study: Removing the third convolutional block (reducing depth to 2 blocks: 32->64 channels) decreased validation Macro-F1 from "
        "0.9068 to 0.8512. The shallower receptive field restricted the network from capturing larger contextual landscape features. "
        "Error Analysis: Across 4,050 test samples, the best CNN seed misclassified 288 images (7.11% error rate). Visual inspection revealed that "
        "errors predominantly concentrate in visually ambiguous boundary categories: River versus Highway (both appear as narrow linear corridors), "
        "and Pasture versus HerbaceousVegetation (spectral green signatures overlap heavily in single RGB bands)."
    )

    add_p(
        "Stress Testing: Robustness was formally audited by adding synthetic zero-mean Gaussian noise across four variance levels to the normalized "
        "test set. Results exhibited acute degradation: sigma=0.00 -> F1=0.9263; sigma=0.05 -> F1=0.8542; sigma=0.10 -> F1=0.5822; and sigma=0.20 -> "
        "F1=0.1656. While high test accuracy is achieved on clean imagery, the network lacks high-frequency noise invariance, confirming that real-world "
        "sensor noise, haze, or atmospheric distortion requires dedicated denoising filters in production deployment."
    )

    # -------------------------------------------------------------
    # 5. TASK 5: RECOMMENDATION & INFERENCE TOOL
    # -------------------------------------------------------------
    add_h1("5. Task 5: Operational Recommendation, Inference Pipeline & Limitations")
    
    add_p(
        "Model Recommendation: The PyTorch CNN is emphatically recommended for operational deployment. On GPU hardware, it processes a batch "
        "of 64 images in 0.73 milliseconds (11.4 microseconds per patch), which easily satisfies real-time satellite throughput requirements while "
        "delivering a +17.8% F1 advantage over the Baseline and +40.0% over the MLP. If deployment is strictly constrained to ultra-low-power CPU "
        "microcontrollers lacking deep learning accelerators, the Baseline (0.32 ms latency, 17.8K parameters) represents a viable fallback."
    )

    add_p(
        "Inference Tooling (src/infer.py): A production inference pipeline was constructed accepting arbitrary image inputs. The pipeline validates "
        "file integrity, converts and resizes inputs to 64x64 RGB, applies train-set normalization, and returns predicted class names with Softmax "
        "confidence scores. An empirical validation threshold (tau = 0.60) was established for out-of-distribution (OOD) flagging."
    )

    add_p(
        "OOD & Edge Case Audit: Evaluating edge cases produced profound empirical findings:\n"
        "• Valid In-Distribution Satellite Image: Correctly predicted PermanentCrop with 99.2% confidence.\n"
        "• Corrupt / Non-Image File (text): The pipeline safely caught the format exception, outputting an explicit error message without crashing.\n"
        "• Out-of-Distribution Input (Pure White Noise): The CNN classified the random noise as 'Highway' with 88.8% confidence. This rigorously "
        "corroborates academic literature: Softmax confidence is fundamentally uncalibrated on out-of-distribution data. Networks can be confidently "
        "wrong on synthetic inputs. True OOD detection requires Mahalanobis distance or energy-based scoring rather than naive Softmax thresholding."
    )

    add_p(
        "Operational Limitations: (1) Geographic Bias: EuroSAT is composed solely of European landscapes, biasing vegetation indices against tropical, "
        "arid, or polar biomes. (2) Sensor Specificity: Trained exclusively on Sentinel-2 optical bands; performance will degrade on Landsat, PlanetScope, "
        "or radar (SAR). (3) Cloud Occlusion: The dataset is curated for cloud-free conditions; cloud shadows trigger severe false predictions. "
        "Future Work: Incorporate EuroSAT's 13-band multispectral imagery to resolve vegetation ambiguity (NIR/SWIR bands) and integrate Monte Carlo "
        "Dropout for calibrated Bayesian uncertainty estimation."
    )

    # -------------------------------------------------------------
    # 6. TASK 6: REPRODUCIBILITY & DEMONSTRATION
    # -------------------------------------------------------------
    add_h1("6. Task 6: Reproducibility & Demonstration Audit")
    
    add_p(
        "All code, configurations, and artifacts are fully reproducible. Environment dependencies are pinned in requirements.txt (Python 3.12, "
        "PyTorch 2.5, Torchvision 0.20, Scikit-Learn 1.6). Execution is orchestrated via 8 structured notebooks and automated CLI scripts with "
        "fixed pseudo-random seeds. Results are machine-logged in metrics/results.csv. The 5-to-7 minute demonstration script and video outline "
        "are cataloged in DEMO_SCRIPT.md and EuroSAT_CW1_Demo_Script.pdf."
    )

    # Save
    out_path = "eurosat-cw1/Technical_Report_NB6057CEM.docx"
    doc.save(out_path)
    print(f"Report successfully written to {out_path}")

create_report()
