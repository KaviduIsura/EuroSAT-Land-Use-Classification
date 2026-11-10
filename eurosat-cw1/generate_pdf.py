import os
from fpdf import FPDF

class DemoPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'NB6057CEM ANN Coursework 1 - 5 to 7 Minute Demonstration Guide', align='R', new_x='LMARGIN', new_y='NEXT')
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

pdf = DemoPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font('helvetica', 'B', 18)
pdf.set_text_color(24, 43, 73)
pdf.cell(0, 10, 'EuroSAT Land Use Classification - Demo Script', new_x='LMARGIN', new_y='NEXT')

pdf.set_font('helvetica', 'I', 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, 'Module: NB6057CEM | Target Time: 5 to 7 Minutes | Coverage: Notebooks 01 to 08', new_x='LMARGIN', new_y='NEXT')
pdf.ln(3)

# Timing Table
pdf.set_font('helvetica', 'B', 12)
pdf.set_text_color(30, 80, 150)
pdf.cell(0, 8, '1. Recommended Time Management', new_x='LMARGIN', new_y='NEXT')

pdf.set_font('helvetica', 'B', 9)
pdf.set_fill_color(240, 243, 248)
pdf.set_text_color(40, 40, 40)
pdf.cell(40, 7, 'Section', border=1, fill=True)
pdf.cell(65, 7, 'Notebooks Covered', border=1, fill=True)
pdf.cell(40, 7, 'Allocated Time', border=1, fill=True)
pdf.cell(45, 7, 'Importance', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')

timing_data = [
    ('Part 1: Data & Split', 'NB 01 (Download) & NB 02 (EDA)', '0:00 - 0:45 (45s)', 'Important (T1: 8 marks)'),
    ('Part 2: NumPy MLP', 'NB 03 (Hand-coded Backprop)', '0:45 - 2:00 (1m 15s)', 'CRITICAL (T2: 12 marks)'),
    ('Part 3: Baseline & Models', 'NB 04 (HOG), NB 05 (MLP), NB 06 (CNN)', '2:00 - 3:30 (1m 30s)', 'High (T3: 14 marks)'),
    ('Part 4: Experiments', 'NB 07 (Evaluation & Noise Stress)', '3:30 - 4:45 (1m 15s)', 'CRITICAL (T4: 14 marks)'),
    ('Part 5: Live Inference', 'NB 08 (Recommendation & Demo)', '4:45 - 6:00 (1m 15s)', 'High (T5/T6: 12 marks)')
]

pdf.set_font('helvetica', '', 9)
for row in timing_data:
    pdf.cell(40, 6, row[0], border=1)
    pdf.cell(65, 6, row[1], border=1)
    pdf.cell(40, 6, row[2], border=1)
    pdf.cell(45, 6, row[3], border=1, new_x='LMARGIN', new_y='NEXT')

pdf.ln(5)

def add_section(title, screen, speech):
    pdf.set_font('helvetica', 'B', 11.5)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 7, title, new_x='LMARGIN', new_y='NEXT')
    
    pdf.set_font('helvetica', 'I', 8.5)
    pdf.set_text_color(160, 50, 50)
    pdf.cell(0, 5, f'Action on Screen: {screen}', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(1)
    
    pdf.set_font('helvetica', '', 9)
    pdf.set_text_color(30, 30, 30)
    pdf.set_fill_color(248, 249, 250)
    pdf.multi_cell(0, 5, speech, border=1, fill=True)
    pdf.ln(3)

add_section(
    'Part 1: Introduction, Data Acquisition & Split (45 seconds)',
    'Open Notebook 01 (01_data_download.ipynb) and then Notebook 02 (02_eda_split.ipynb)',
    'Hello everyone. Today I will demonstrate my entire EuroSAT Land Use Classification coursework.\n\n'
    'In Notebook 1, I downloaded the EuroSAT RGB dataset containing 27,000 satellite images across 10 classes. Each image has 64 by 64 RGB pixels. I saved the dataset as an NPZ archive to Google Drive for fast local loading.\n\n'
    'In Notebook 2, I performed a data audit. Zero corrupt images and zero exact duplicates were found. I created a stratified 70% train, 15% validation, and 15% test split. To strictly prevent data leakage, channel mean and standard deviation were calculated only from the training set.'
)

add_section(
    'Part 2: NumPy Multi-Layer Perceptron from Scratch (1 minute 15 seconds)',
    'Open Notebook 03 (03_numpy_mlp.ipynb) - Show Python class and gradient check table',
    'In Notebook 3, I implemented an MLP completely from scratch using NumPy without PyTorch autograd.\n\n'
    'I trained it on a 4-class subset resized to 16 by 16 grayscale. The architecture is 256 inputs, one hidden layer of 32 units with tanh activation, and 4 softmax outputs.\n\n'
    'I manually derived the backward pass using the chain rule to update weights and biases. To rigorously verify my math, I implemented a numerical gradient check with central differences. All relative errors are under 1e-7, proving the analytical gradients are 100% correct. I also overfitted a tiny 10-image batch to confirm training loss reaches zero.'
)

add_section(
    'Part 3: Baseline & Deep Learning Models (1 minute 30 seconds)',
    'Open Notebooks 04, 05, and 06 (04_baseline_hog_lr, 05_mlp_pytorch, 06_cnn_pytorch)',
    'Next, I trained three comparative models on all 10 classes:\n\n'
    'In Notebook 4, the Baseline extracts hand-crafted HOG edge orientations and color histograms with Logistic Regression. It reached 0.73 Test Macro-F1 with 17,880 parameters.\n\n'
    'In Notebook 5, the PyTorch MLP has two hidden layers of 512 and 256 neurons. It has 6.4 million parameters but only achieved 0.51 Macro-F1. Flat linear layers ignore spatial 2D relationships, leading to high parameters and poor generalization.\n\n'
    'In Notebook 6, the PyTorch CNN uses 3 convolutional blocks with 32, 64, and 128 channels, batch normalization, and data augmentation. It achieved 0.91 Macro-F1 and 93% accuracy with only 94,730 parameters - 68 times fewer parameters than the MLP.'
)

pdf.add_page()

add_section(
    'Part 4: Comparative Evaluation & Noise Stress Test (1 minute 15 seconds)',
    'Open Notebook 07 (07_experiments_evaluation.ipynb) - Show summary table, confusion matrix & noise plot',
    'In Notebook 7, all models were evaluated across 3 random seeds:\n\n'
    'The summary confirms CNN is the top model with 0.9103 Test Macro-F1 versus 0.5100 for MLP and 0.7324 for Baseline. Individual classes like Forest, Residential, and SeaLake reached 97% to 98% F1. Misclassifications mainly occurred between visually similar terrain like River and Highway.\n\n'
    'I also conducted a Gaussian noise stress test. As noise sigma increased from 0.0 to 0.20, Macro-F1 dropped sharply from 0.92 to 0.16. This reveals that the CNN is sensitive to atmospheric sensor noise and would require denoising in a production satellite pipeline.'
)

add_section(
    'Part 5: Live Inference Pipeline & Edge Case Demo (1 minute 15 seconds)',
    'Open Notebook 08 (08_recommendation_inference.ipynb) - Run or highlight the prediction output cells',
    'Finally, in Notebook 8, I benchmarked inference latency and deployed the prediction pipeline:\n\n'
    'Inference takes only 0.73 milliseconds per batch of 64 on GPU, proving it is ready for real-time applications.\n\n'
    'Here is the live inference demonstration:\n'
    '1. Valid Satellite Image: Correctly classified as PermanentCrop with 99.2% confidence.\n'
    '2. Invalid Text File: The pipeline safely catches the error and reports an unreadable file.\n'
    '3. Out-of-Scope White Noise: The network predicts Highway with 88.8% confidence. This demonstrates an essential academic finding: Softmax confidence is imperfect for out-of-distribution detection, as neural networks can be confidently wrong on synthetic noise.\n\n'
    'In conclusion, I recommend the CNN for deployment due to high accuracy, low parameters, and sub-millisecond latency. Thank you.'
)

# Examiner Q&A Cheat Sheet
pdf.set_font('helvetica', 'B', 11.5)
pdf.set_text_color(30, 80, 150)
pdf.cell(0, 7, '3. Examiner Q&A Cheat Sheet (Short Answers)', new_x='LMARGIN', new_y='NEXT')

qa_pairs = [
    ('Q1: Why use Macro-F1 instead of Accuracy?',
     'A: Macro-F1 evaluates all classes equally. Accuracy can hide poor performance on smaller minority classes.'),
    ('Q2: Why does CNN outperform MLP with 68x fewer parameters?',
     'A: CNN has spatial inductive bias: local receptive fields and shared convolutional weights capture spatial features regardless of pixel location.'),
    ('Q3: What is the biological analogy of an artificial neuron?',
     'A: Inputs represent dendrites, weights represent synaptic strengths, summation is the cell soma membrane potential, and activation is the axon firing action.'),
    ('Q4: Why was gradient check done with tanh instead of ReLU?',
     'A: ReLU has a non-differentiable kink at zero which causes finite difference checks to fail. Tanh is smooth and differentiable everywhere.'),
    ('Q5: What was the main finding of the noise stress test?',
     'A: It proved high clean test accuracy does not guarantee robustness against sensor degradation; performance dropped from 0.92 to 0.16 at sigma=0.20.')
]

for q, a in qa_pairs:
    pdf.set_font('helvetica', 'B', 8.5)
    pdf.set_text_color(180, 40, 40)
    pdf.cell(0, 4.5, q, new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 4.2, a)
    pdf.ln(1)

out_pdf = 'eurosat-cw1/EuroSAT_CW1_Demo_Script.pdf'
pdf.output(out_pdf)
print(f'PDF successfully saved to: {out_pdf}')

