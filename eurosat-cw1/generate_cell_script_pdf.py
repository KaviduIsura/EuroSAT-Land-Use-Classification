import os
from fpdf import FPDF

class CellScriptPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 8)
        self.set_text_color(100, 110, 125)
        self.cell(0, 5, 'EuroSAT CW1 - Cell-by-Cell Demonstration Script (Notebooks 01 to 08)', align='R', new_x='LMARGIN', new_y='NEXT')
        self.line(10, 14, 200, 14)
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, f'Page {self.page_no()}', align='C')

pdf = CellScriptPDF()
pdf.set_auto_page_break(auto=True, margin=14)
pdf.add_page()

# Title
pdf.set_font('helvetica', 'B', 15)
pdf.set_text_color(27, 54, 93)
pdf.cell(0, 7, 'EuroSAT CW1 - Cell-by-Cell Demonstration Script', new_x='LMARGIN', new_y='NEXT')

pdf.set_font('helvetica', 'I', 9.5)
pdf.set_text_color(90, 105, 120)
pdf.cell(0, 5, 'Simple plain English | Short format | Complete guide for all 8 notebooks', new_x='LMARGIN', new_y='NEXT')
pdf.ln(3)

def nb_header(nb_title):
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(27, 54, 93)
    pdf.cell(0, 7, nb_title, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(1)

def cell_entry(cell_num, what_does, to_say):
    pdf.set_font('helvetica', 'B', 9)
    pdf.set_text_color(43, 108, 176)
    pdf.cell(0, 4.5, cell_num, new_x='LMARGIN', new_y='NEXT')
    
    pdf.set_font('helvetica', 'I', 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 4, f'What it does: {what_does}', new_x='LMARGIN', new_y='NEXT')
    
    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(30, 30, 30)
    pdf.set_fill_color(248, 249, 250)
    pdf.multi_cell(0, 4.2, f'Say: \"{to_say}\"', border=1, fill=True)
    pdf.ln(2)

# NB 1
nb_header('Notebook 01: Data Download (01_data_download.ipynb)')
cell_entry('Cell 1: Setup & Clone', 'Clones the GitHub repository and sets the working folder.', 'In Cell 1, I set up the environment and clone my coursework repository from GitHub.')
cell_entry('Cell 2: Mount Drive', 'Connects Google Drive for persistent storage.', 'In Cell 2, I mount Google Drive. This ensures that all downloaded data, models, and metrics are permanently saved.')
cell_entry('Cell 3: Download Dataset', 'Downloads the 27,000 satellite images using torchvision.', 'In Cell 3, I download the official EuroSAT RGB dataset. It contains 27,000 satellite image patches across 10 land-use classes.')
cell_entry('Cell 4: Verify Folders', 'Checks that all 10 class folders exist and counts images.', 'In Cell 4, I verify the download. Each class has between 2,000 and 3,000 images, and all images are 64 by 64 pixels.')
cell_entry('Cell 5: Copy to Drive', 'Backs up the images to Google Drive.', 'In Cell 5, I back up the raw images to Google Drive. This avoids downloading them again in future sessions.')
cell_entry('Cell 6: Save Metadata', 'Records dataset source, date, citation, and class names.', 'In Cell 6, I save the dataset metadata, citing Helber et al. (2019) and recording Sentinel-2 sensor details.')

# NB 2
nb_header('Notebook 02: EDA & Leakage-Safe Split (02_eda_split.ipynb)')
cell_entry('Cell 1: Setup', 'Imports libraries for hashing, plotting, and splitting.', 'In Cell 1, I import NumPy, PIL, and hashing tools for data auditing.')
cell_entry('Cell 2: Check Hashes', 'Verifies image integrity and generates MD5 hashes.', 'In Cell 2, I check every image with PIL and calculate MD5 hashes to detect corrupt files or duplicates.')
cell_entry('Cell 3: Build Array', 'Packages all 27,000 images into a single eurosat_64.npz archive.', 'In Cell 3, I pack the images and labels into a single NumPy array on Google Drive for fast local loading.')
cell_entry('Cell 4: Audit Report', 'Prints the audit report.', 'In Cell 4, you can see the audit report. There are zero corrupt images and zero exact duplicates.')
cell_entry('Cell 5: Class Bar Chart', 'Generates a bar chart of class counts.', 'In Cell 5, I plot the class distribution. Pasture has 2,000 images while Forest has 3,000. This mild imbalance justifies using Macro-F1.')
cell_entry('Cell 6: Sample Grid', 'Displays 3 sample images for each of the 10 classes.', 'In Cell 6, I display sample images. This shows visual similarities, like green pastures looking very close to herbaceous vegetation.')
cell_entry('Cell 7: Pixel Histograms', 'Plots red, green, and blue intensity distributions.', 'In Cell 7, I plot channel histograms. The three color channels have different distributions, which justifies per-channel normalization.')
cell_entry('Cell 8: Stratified Split', 'Splits indices into 70% train, 15% validation, and 15% test.', 'In Cell 8, I split the dataset into 70% train, 15% validation, and 15% test. It is stratified to keep class balance identical across splits.')
cell_entry('Cell 9: Leakage Check', 'Verifies zero hash overlap between splits.', 'In Cell 9, I run a leakage check. There are zero hash overlaps between train, val, and test.')
cell_entry('Cell 10: Normalization', 'Calculates mean and std strictly on train and saves split indices.', 'In Cell 10, I calculate channel mean and std strictly on the training set. This guarantees zero data leakage.')

pdf.add_page()

# NB 3
nb_header('Notebook 03: NumPy MLP from Scratch (03_numpy_mlp.ipynb)')
cell_entry('Cell 1: Setup & Load', 'Loads saved dataset and split indices.', 'In Cell 1, I load the saved dataset and training indices.')
cell_entry('Cell 2: 4-Class Subset', 'Extracts 4 classes downsampled to 16x16 grayscale (256 features).', 'In Cell 2, I prepare the Task 2 teaching dataset. I take 4 classes and downsample them to 16 by 16 grayscale, giving 256 inputs.')
cell_entry('Cell 3: NumPy MLP Class', 'Hand-coded forward and backward propagation in pure NumPy.', 'In Cell 3, I implement the neural network from scratch using only NumPy. It uses a tanh hidden layer and Softmax output. The backward pass derives exact gradients using the chain rule.')
cell_entry('Cell 4: Train MLP', 'Trains using gradient descent for 200 epochs.', 'In Cell 4, I train the network. The initial loss is 1.44, very close to the theoretical value of ln(4) = 1.386. The final loss drops to 0.50.')
cell_entry('Cell 5: Loss Curve Plot', 'Plots epoch vs cross-entropy loss.', 'In Cell 5, I plot the training curve. It shows smooth, stable convergence.')
cell_entry('Cell 6: Gradient Check', 'Compares analytical gradients against central differences (eps=1e-5).', 'In Cell 6, I verify the backprop math using a numerical gradient check. All 12 parameters have relative errors below 1e-10. This proves the hand-coded gradients are 100% correct.')
cell_entry('Cell 7: Overfit Tiny Batch', 'Trains on 10 images to confirm loss approaches zero.', 'In Cell 7, I overfit a tiny batch of 10 samples. The loss drops to 0.001, confirming the model has no implementation bugs.')

# NB 4
nb_header('Notebook 04: Baseline Model (04_baseline_hog_lr.ipynb)')
cell_entry('Cell 1: Setup', 'Imports scikit-learn, skimage, and loading tools.', 'In Cell 1, I set up the environment for the classical machine learning baseline.')
cell_entry('Cell 2: Feature Extractor', 'Extracts HOG edge features and color histograms (1,788 features).', 'In Cell 2, I extract hand-crafted features: HOG captures edge orientation, and color histograms capture color patterns.')
cell_entry('Cell 3: Scale Features', 'Fits StandardScaler strictly on training set.', 'In Cell 3, I scale the features. The scaler is fit strictly on training features to prevent leakage.')
cell_entry('Cell 4: Hyperparameter Tuning', 'Evaluates 6 values of C on validation Macro-F1.', 'In Cell 4, I tune the regularization parameter C over 6 trials. C = 0.01 achieves the highest validation Macro-F1 of 0.7299.')
cell_entry('Cell 5: Train Final Model', 'Trains final Logistic Regression with C = 0.01.', 'In Cell 5, I train the final baseline model with the best C value and save the checkpoint.')
cell_entry('Cell 6: Validation Score', 'Evaluates precision, recall, and F1 on validation data.', 'In Cell 6, I evaluate validation performance. Macro-F1 is 0.7299 and accuracy is 73.6%.')
cell_entry('Cell 7: Confusion Matrix', 'Plots validation confusion matrix.', 'In Cell 7, I plot the confusion matrix. Forest scores 94% F1, while Herbaceous Vegetation is harder at 55%.')
cell_entry('Cell 8: Test Set Evaluation', 'Unfreezes and evaluates the final test set.', 'In Cell 8, I evaluate the unseen test set. The Baseline reaches 0.7324 Macro-F1 with 17,880 parameters.')

pdf.add_page()

# NB 5
nb_header('Notebook 05: PyTorch MLP (05_mlp_pytorch.ipynb)')
cell_entry('Cell 1: Setup', 'Initializes PyTorch and verifies GPU availability.', 'In Cell 1, I set up PyTorch and enable the GPU.')
cell_entry('Cell 2: PyTorch Dataset', 'Loads image tensors with train-only normalization and augmentation.', 'In Cell 2, I build the PyTorch Dataset with data augmentation: horizontal flips, vertical flips, and rotations.')
cell_entry('Cell 3: MLP Architecture', 'Defines 2 hidden layers (512 and 256 units) with ReLU and Dropout(0.3).', 'In Cell 3, I define the deep MLP. Because it flattens 64 by 64 RGB images, it requires over 6.4 million parameters.')
cell_entry('Cell 4: Training Loop', 'Implements training and validation with early stopping.', 'In Cell 4, I implement training with Adam and early stopping with a patience of 5 epochs.')
cell_entry('Cell 5: Train 3 Seeds', 'Trains separate models for seeds 0, 1, and 2.', 'In Cell 5, I train the MLP across 3 random seeds to evaluate variance.')
cell_entry('Cell 6: Results Summary', 'Calculates mean and standard deviation across seeds.', 'In Cell 6, I summarize the results. The MLP achieves a Test Macro-F1 of only 0.5100 with 53% accuracy.')
cell_entry('Cell 7: Loss Curves Plot', 'Plots train vs validation loss curves for all seeds.', 'In Cell 7, I plot the loss curves. The validation loss quickly diverges, showing that the flat MLP overfits heavily.')
cell_entry('Cell 8: Test Confusion Matrix', 'Plots test confusion matrix for the best seed.', 'In Cell 8, I plot the confusion matrix. The MLP struggles because flat layers destroy the 2D spatial arrangement of images.')

# NB 6
nb_header('Notebook 06: PyTorch CNN (06_cnn_pytorch.ipynb)')
cell_entry('Cell 1: Setup', 'Imports PyTorch neural network modules.', 'In Cell 1, I initialize PyTorch for the Convolutional Neural Network.')
cell_entry('Cell 2: Dataset & Loaders', 'Builds batches of 64 images for GPU training.', 'In Cell 2, I initialize the DataLoader with batch size 64 and augmentation.')
cell_entry('Cell 3: CNN Architecture', 'Defines 3 conv blocks (32, 64, 128) with BatchNorm, ReLU, and MaxPool.', 'In Cell 3, I build the CNN architecture. It has 3 convolutional blocks, batch normalization, and adaptive pooling. It uses only 94,986 parameters.')
cell_entry('Cell 4: Training Setup', 'Defines Adam optimizer (lr=1e-3) and early stopping.', 'In Cell 4, I set up Adam optimizer with learning rate 1e-3 and early stopping.')
cell_entry('Cell 5: Train 3 Seeds', 'Trains the CNN for seeds 0, 1, and 2, saving checkpoints.', 'In Cell 5, I train the CNN across 3 seeds. Notice how training is smooth and reaches over 90% validation F1.')
cell_entry('Cell 6: Results Summary', 'Computes mean and standard deviation across seeds.', 'In Cell 6, I display the results. The CNN achieves a Test Macro-F1 of 0.9103 and 93% accuracy. It beats the MLP by 40% with 68 times fewer parameters.')
cell_entry('Cell 7: Loss & Accuracy Plots', 'Plots training vs validation curves.', 'In Cell 7, I plot the curves. Training and validation lines track closely together, proving strong generalization.')
cell_entry('Cell 8: Confusion Matrix', 'Plots final test set confusion matrix.', 'In Cell 8, I plot the confusion matrix. Classes like Forest, SeaLake, and Residential reach 97% to 98% F1.')
cell_entry('Cell 9: Ablation Study', 'Retrains the CNN with dropout = 0.0.', 'In Cell 9, I run the ablation study by removing dropout. Validation F1 changes by only +0.0073, showing batch norm and augmentation already provide strong regularization.')

pdf.add_page()

# NB 7
nb_header('Notebook 07: Experiments & Evaluation (07_experiments_evaluation.ipynb)')
cell_entry('Cell 1: Setup & Restore', 'Restores all checkpoints, splits, and metrics from Drive.', 'In Cell 1, I load all saved checkpoints and test splits from Google Drive.')
cell_entry('Cell 2: Master Summary Table', 'Compares Baseline, MLP, and CNN.', 'In Cell 2, I compare all models. CNN is the clear winner with 0.9103 F1, compared to 0.7324 for Baseline and 0.5100 for MLP.')
cell_entry('Cell 3: Per-Class Report', 'Computes precision, recall, and F1 for all 10 classes.', 'In Cell 3, I display the per-class report. Overall test accuracy is 93%, with precision and recall above 90% across most classes.')
cell_entry('Cell 4: Error Analysis', 'Identifies 288 misclassified images (7.1%) and shows 5 examples.', 'In Cell 4, I inspect model errors. Only 288 out of 4,050 images were misclassified. Errors mostly happen between visually similar classes like River and Highway.')
cell_entry('Cell 5: Gaussian Noise Stress Test', 'Tests CNN under noise levels (0.0 to 0.20).', 'In Cell 5, I run the stress test. As noise increases to 0.20, Macro-F1 drops from 0.92 to 0.16. This reveals that the CNN is sensitive to atmospheric sensor noise.')
cell_entry('Cell 6: Save Master CSV', 'Writes metrics/results.csv logging all models and seeds.', 'In Cell 6, I save the complete results into results.csv for reproducibility.')

# NB 8
nb_header('Notebook 08: Recommendation & Inference Tool (08_recommendation_inference.ipynb)')
cell_entry('Cell 1: Setup', 'Loads class labels and train normalization stats.', 'In Cell 1, I load the class names and normalization parameters for the inference tool.')
cell_entry('Cell 2: Recommendation Table', 'Compares accuracy, parameters, and trade-offs.', 'In Cell 2, I present the recommendation table. CNN is recommended because it combines the highest accuracy with a compact 94K parameter footprint.')
cell_entry('Cell 3: Inference Latency', 'Benchmarks GPU inference time for batch of 64.', 'In Cell 3, I benchmark inference speed. The CNN processes a batch of 64 in only 0.73 milliseconds, which is fast enough for real-time satellite processing.')
cell_entry('Cell 4: The predict() Function', 'Builds complete inference pipeline with thresholding.', 'In Cell 4, I define the production predict function. It checks file validity, normalizes the input, and returns class predictions with confidence scores.')
cell_entry('Cell 5: Demo - Normal Satellite Image', 'Predicts on a valid test image.', 'In Cell 5, I test a normal satellite image. The CNN correctly predicts PermanentCrop with 99.2% confidence.')
cell_entry('Cell 6: Demo - Edge Cases & Noise', 'Tests invalid text file and pure white noise.', 'In Cell 6, I test edge cases. The text file is safely caught with an error message. But for random white noise, the network predicts Highway with 88.8% confidence. This proves an important finding: Softmax neural networks can be overly confident on out-of-distribution inputs.')

out_pdf = 'eurosat-cw1/CELL_BY_CELL_DEMO_SCRIPT.pdf'
pdf.output(out_pdf)
print(f'PDF saved to {out_pdf}')

import shutil
shutil.copy2(out_pdf, 'CW2/CELL_BY_CELL_DEMO_SCRIPT.pdf')
shutil.copy2(out_pdf, 'CELL_BY_CELL_DEMO_SCRIPT.pdf')
print('Copied PDF to CW2/ and root.')
