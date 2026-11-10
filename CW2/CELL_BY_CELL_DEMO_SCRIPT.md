# EuroSAT CW1 — Cell-by-Cell Demonstration Script
**Module:** NB6057CEM Artificial Neural Networks  
**Format:** Simple, plain English, short sentences.  
**How to use:** Scroll through your notebooks cell by cell. Read the bold line to yourself, and say the quoted text out loud to the examiner.

---

# Notebook 01: Data Download (`01_data_download.ipynb`)

### Cell 1: Setup & Clone
* **What it does:** Clones the GitHub repository and sets the working folder.
* **What to say:**  
  > "In Cell 1, I set up the environment and clone my coursework repository from GitHub."

### Cell 2: Mount Google Drive
* **What it does:** Connects Google Drive for persistent storage.
* **What to say:**  
  > "In Cell 2, I mount Google Drive. This ensures that all downloaded data, models, and metrics are permanently saved."

### Cell 3: Download EuroSAT RGB
* **What it does:** Downloads the 27,000 satellite images using `torchvision`.
* **What to say:**  
  > "In Cell 3, I download the official EuroSAT RGB dataset. It contains 27,000 satellite image patches across 10 land-use classes."

### Cell 4: Verify Folder Structure
* **What it does:** Checks that all 10 class folders exist and counts images.
* **What to say:**  
  > "In Cell 4, I verify the download. Each class has between 2,000 and 3,000 images, and all images are 64 by 64 pixels."

### Cell 5: Copy Dataset to Google Drive
* **What it does:** Backs up the images to Google Drive.
* **What to say:**  
  > "In Cell 5, I back up the raw images to Google Drive. This avoids downloading them again in future sessions."

### Cell 6: Save Metadata JSON
* **What it does:** Records dataset source, date, citation, and class names into `metadata.json`.
* **What to say:**  
  > "In Cell 6, I save the dataset metadata, citing Helber et al. (2019) and recording the Sentinel-2 sensor details."

---

# Notebook 02: EDA & Leakage-Safe Split (`02_eda_split.ipynb`)

### Cell 1: Setup
* **What it does:** Imports libraries for hashing, plotting, and splitting.
* **What to say:**  
  > "In Cell 1, I import NumPy, PIL, and hashing tools for data auditing."

### Cell 2: Compute MD5 Hashes
* **What it does:** Verifies image integrity and generates MD5 hashes.
* **What to say:**  
  > "In Cell 2, I check every image with PIL and calculate MD5 hashes to detect corrupt files or duplicates."

### Cell 3: Save NumPy Array
* **What it does:** Packages all 27,000 images into a single `eurosat_64.npz` archive.
* **What to say:**  
  > "In Cell 3, I pack the images and labels into a single NumPy array on Google Drive for fast local loading."

### Cell 4: Data Audit Report
* **What it does:** Prints the audit results.
* **What to say:**  
  > "In Cell 4, you can see the audit report. There are zero corrupt images and zero exact duplicates."

### Cell 5: Plot 1 — Class Distribution
* **What it does:** Generates a bar chart of class counts.
* **What to say:**  
  > "In Cell 5, I plot the class distribution. Pasture has 2,000 images while Forest has 3,000. This mild imbalance justifies using Macro-F1."

### Cell 6: Plot 2 — Sample Grid
* **What it does:** Displays 3 sample images for each of the 10 classes.
* **What to say:**  
  > "In Cell 6, I display sample images. This shows visual similarities, like green pastures looking very close to herbaceous vegetation."

### Cell 7: Plot 3 — Channel Pixel Histograms
* **What it does:** Plots red, green, and blue intensity distributions.
* **What to say:**  
  > "In Cell 7, I plot channel histograms. The three color channels have different distributions, which justifies per-channel normalization."

### Cell 8: Stratified Split (70 / 15 / 15)
* **What it does:** Splits indices into 18,900 train, 4,050 val, and 4,050 test.
* **What to say:**  
  > "In Cell 8, I split the dataset into 70% train, 15% validation, and 15% test. It is stratified to keep class balance identical across splits."

### Cell 9: Leakage Check
* **What it does:** Checks that no image hash exists in more than one partition.
* **What to say:**  
  > "In Cell 9, I run a leakage check. There are zero hash overlaps between train, val, and test."

### Cell 10: Train-Only Normalization
* **What it does:** Calculates mean and std strictly on train, and saves `split_indices.npz`.
* **What to say:**  
  > "In Cell 10, I calculate channel mean and std strictly on the training set. This guarantees zero data leakage."

---

# Notebook 03: NumPy MLP from Scratch (`03_numpy_mlp.ipynb`)

### Cell 1: Setup & Load
* **What it does:** Loads `eurosat_64.npz` and split indices.
* **What to say:**  
  > "In Cell 1, I load the saved dataset and training indices."

### Cell 2: Create 4-Class Subset
* **What it does:** Selects 4 classes and resizes images to 16×16 grayscale (256 features).
* **What to say:**  
  > "In Cell 2, I prepare the Task 2 teaching dataset. I take 4 classes and downsample them to 16 by 16 grayscale, giving 256 inputs."

### Cell 3: The Hand-Coded MLP Class
* **What it does:** Complete forward and backward propagation in pure NumPy without autograd.
* **What to say:**  
  > "In Cell 3, I implement the neural network from scratch using only NumPy. It uses a tanh hidden layer and a Softmax output. The backward pass derives exact gradients using the chain rule."

### Cell 4: Train NumPy MLP
* **What it does:** Trains the model using mini-batch gradient descent for 200 epochs.
* **What to say:**  
  > "In Cell 4, I train the network. The initial loss is 1.44, very close to the theoretical value of ln(4) = 1.386. The final loss drops to 0.50."

### Cell 5: Plot Loss Curve
* **What it does:** Plots epoch vs cross-entropy loss.
* **What to say:**  
  > "In Cell 5, I plot the training curve. It shows smooth, stable convergence."

### Cell 6: Gradient Check (Central Differences)
* **What it does:** Compares analytical gradients against finite differences with $\epsilon = 10^{-5}$.
* **What to say:**  
  > "In Cell 6, I verify the backprop math using a numerical gradient check. All 12 parameters have relative errors below 1e-10. This proves the hand-coded gradients are 100% correct."

### Cell 7: Overfit a Tiny Batch
* **What it does:** Trains on only 10 images to verify that loss reaches near zero.
* **What to say:**  
  > "In Cell 7, I overfit a tiny batch of 10 samples. The loss drops to 0.001, confirming the model has no implementation bugs."

---

# Notebook 04: Baseline Model (`04_baseline_hog_lr.ipynb`)

### Cell 1: Setup
* **What it does:** Imports scikit-learn, skimage, and loading tools.
* **What to say:**  
  > "In Cell 1, I set up the environment for the classical machine learning baseline."

### Cell 2: Feature Extractor Function
* **What it does:** Extracts HOG edge features and color histograms (1,788 features per image).
* **What to say:**  
  > "In Cell 2, I extract hand-crafted features: HOG captures edge orientation, and color histograms capture color patterns."

### Cell 3: Scale Features
* **What it does:** Fits `StandardScaler` strictly on training features and transforms val/test.
* **What to say:**  
  > "In Cell 3, I scale the features. The scaler is fit strictly on training features to prevent leakage."

### Cell 4: Hyperparameter Tuning (6 Trials)
* **What it does:** Trains Logistic Regression with 6 values of $C$ on the validation set.
* **What to say:**  
  > "In Cell 4, I tune the regularization parameter C over 6 trials. C = 0.01 achieves the highest validation Macro-F1 of 0.7299."

### Cell 5: Train Final Model
* **What it does:** Trains the final classifier with $C = 0.01$ and saves `baseline_hog_lr.pkl`.
* **What to say:**  
  > "In Cell 5, I train the final baseline model with the best C value and save the checkpoint."

### Cell 6: Validation Evaluation
* **What it does:** Evaluates precision, recall, and F1 on the validation set.
* **What to say:**  
  > "In Cell 6, I evaluate validation performance. Macro-F1 is 0.7299 and accuracy is 73.6%."

### Cell 7: Confusion Matrix
* **What it does:** Plots the validation confusion matrix.
* **What to say:**  
  > "In Cell 7, I plot the confusion matrix. Forest scores 94% F1, while Herbaceous Vegetation is harder at 55%."

### Cell 8: Final Test Set Evaluation
* **What it does:** Unfreezes and evaluates the test set.
* **What to say:**  
  > "In Cell 8, I evaluate the unseen test set. The Baseline reaches 0.7324 Macro-F1 with 17,880 parameters."

---

# Notebook 05: PyTorch MLP (`05_mlp_pytorch.ipynb`)

### Cell 1: Setup
* **What it does:** Loads PyTorch and verifies GPU availability.
* **What to say:**  
  > "In Cell 1, I set up PyTorch and enable the GPU."

### Cell 2: PyTorch Dataset Class
* **What it does:** Loads image tensors with train-only normalization and augmentation.
* **What to say:**  
  > "In Cell 2, I build the PyTorch Dataset with data augmentation: horizontal flips, vertical flips, and rotations."

### Cell 3: MLP Architecture
* **What it does:** Defines 2 hidden layers (512 and 256 units) with ReLU and Dropout(0.3).
* **What to say:**  
  > "In Cell 3, I define the deep MLP. Because it flattens 64 by 64 RGB images, it requires over 6.4 million parameters."

### Cell 4: Training & Early Stopping Functions
* **What it does:** Implements `train_epoch` and `validate` with Cross-Entropy loss.
* **What to say:**  
  > "In Cell 4, I implement training with Adam and early stopping with a patience of 5 epochs."

### Cell 5: Train Across 3 Seeds
* **What it does:** Trains separate models for seeds 0, 1, and 2.
* **What to say:**  
  > "In Cell 5, I train the MLP across 3 random seeds to evaluate variance."

### Cell 6: Results Summary
* **What it does:** Calculates mean and std across seeds.
* **What to say:**  
  > "In Cell 6, I summarize the results. The MLP achieves a Test Macro-F1 of only 0.5100 with 53% accuracy."

### Cell 7: Loss Curves Plot
* **What it does:** Plots train vs validation loss curves for all seeds.
* **What to say:**  
  > "In Cell 7, I plot the loss curves. The validation loss quickly diverges, showing that the flat MLP overfits heavily."

### Cell 8: Test Confusion Matrix
* **What it does:** Plots the test confusion matrix for the best seed.
* **What to say:**  
  > "In Cell 8, I plot the confusion matrix. The MLP struggles because flat layers destroy the 2D spatial arrangement of images."

---

# Notebook 06: PyTorch CNN (`06_cnn_pytorch.ipynb`)

### Cell 1: Setup
* **What it does:** Imports PyTorch neural network modules.
* **What to say:**  
  > "In Cell 1, I initialize PyTorch for the Convolutional Neural Network."

### Cell 2: Dataset & Loaders
* **What it does:** Builds batches of 64 images for GPU training.
* **What to say:**  
  > "In Cell 2, I initialize the DataLoader with batch size 64 and augmentation."

### Cell 3: CNN Architecture
* **What it does:** Defines 3 conv blocks (32, 64, 128 channels) with BatchNorm, ReLU, and MaxPool.
* **What to say:**  
  > "In Cell 3, I build the CNN architecture. It has 3 convolutional blocks, batch normalization, and adaptive pooling. It uses only 94,986 parameters."

### Cell 4: Training Loop
* **What it does:** Standard training and validation logic.
* **What to say:**  
  > "In Cell 4, I set up Adam optimizer with learning rate 1e-3 and early stopping."

### Cell 5: Train 3 Seeds
* **What it does:** Trains the CNN for seeds 0, 1, and 2, saving the best checkpoints.
* **What to say:**  
  > "In Cell 5, I train the CNN across 3 seeds. Notice how training is smooth and reaches over 90% validation F1."

### Cell 6: Results Summary
* **What it does:** Computes mean and standard deviation.
* **What to say:**  
  > "In Cell 6, I display the results. The CNN achieves a Test Macro-F1 of 0.9103 and 93% accuracy. It beats the MLP by 40% with 68 times fewer parameters."

### Cell 7: Loss & Accuracy Curves
* **What it does:** Plots training vs validation loss and accuracy curves.
* **What to say:**  
  > "In Cell 7, I plot the curves. Training and validation lines track closely together, proving strong generalization."

### Cell 8: Test Confusion Matrix
* **What it does:** Plots the final test confusion matrix.
* **What to say:**  
  > "In Cell 8, I plot the confusion matrix. Classes like Forest, SeaLake, and Residential reach 97% to 98% F1."

### Cell 9: Ablation Study (No Dropout)
* **What it does:** Retrains the CNN with `dropout = 0.0`.
* **What to say:**  
  > "In Cell 9, I run the ablation study by removing dropout. Validation F1 changes by only +0.0073, showing batch norm and augmentation already provide strong regularization."

---

# Notebook 07: Experiments & Evaluation (`07_experiments_evaluation.ipynb`)

### Cell 1: Setup & Restore
* **What it does:** Restores all saved checkpoints, splits, and metrics from Google Drive.
* **What to say:**  
  > "In Cell 1, I load all saved checkpoints and test splits from Google Drive."

### Cell 2: Master Results Summary
* **What it does:** Prints the comparison table across Baseline, MLP, and CNN.
* **What to say:**  
  > "In Cell 2, I compare all models. CNN is the clear winner with 0.9103 F1, compared to 0.7324 for Baseline and 0.5100 for MLP."

### Cell 3: Best CNN Per-Class Classification Report
* **What it does:** Computes precision, recall, and F1 for all 10 classes on the test set.
* **What to say:**  
  > "In Cell 3, I display the per-class report. Overall test accuracy is 93%, with precision and recall above 90% across most classes."

### Cell 4: Error Analysis (Misclassified Images)
* **What it does:** Identifies 288 misclassified test images (7.1%) and displays 5 examples.
* **What to say:**  
  > "In Cell 4, I inspect model errors. Only 288 out of 4,050 images were misclassified. Errors mostly happen between visually similar classes like River and Highway."

### Cell 5: Gaussian Noise Stress Test
* **What it does:** Tests the CNN under increasing noise levels ($\sigma = 0.0, 0.05, 0.10, 0.20$).
* **What to say:**  
  > "In Cell 5, I run the stress test. As noise increases to 0.20, Macro-F1 drops from 0.92 to 0.16. This reveals that the CNN is sensitive to atmospheric sensor noise."

### Cell 6: Generate Master `results.csv`
* **What it does:** Writes the final machine-readable CSV table.
* **What to say:**  
  > "In Cell 6, I save the complete results into results.csv for reproducibility."

---

# Notebook 08: Recommendation & Inference Tool (`08_recommendation_inference.ipynb`)

### Cell 1: Setup
* **What it does:** Loads class labels and train normalization stats.
* **What to say:**  
  > "In Cell 1, I load the class names and normalization parameters for the inference tool."

### Cell 2: Model Recommendation Table
* **What it does:** Compares test F1, parameter counts, and deployment trade-offs.
* **What to say:**  
  > "In Cell 2, I present the recommendation table. CNN is recommended because it combines the highest accuracy with a compact 94K parameter footprint."

### Cell 3: Measure Inference Latency
* **What it does:** Benchmarks inference time for a batch of 64 on GPU.
* **What to say:**  
  > "In Cell 3, I benchmark inference speed. The CNN processes a batch of 64 in only 0.73 milliseconds, which is fast enough for real-time satellite processing."

### Cell 4: The `predict()` Function
* **What it does:** Complete inference pipeline with image resizing, normalization, and confidence threshold.
* **What to say:**  
  > "In Cell 4, I define the production predict function. It checks file validity, normalizes the input, and returns class predictions with confidence scores."

### Cell 5: Live Demo — Normal Satellite Image
* **What it does:** Predicts on a valid test image.
* **What to say:**  
  > "In Cell 5, I tested a normal satellite image from the test set. As shown in the output on my screen, the model correctly classified it as PermanentCrop with 99.2% confidence."

### Cell 6: Live Demo — Edge Cases & Out-of-Scope Noise
* **What it does:** Tests an invalid text file and a pure random noise image.
* **What to say:**  
  > "In Cell 6, I tested two edge cases and obtained these results on screen. First, the text file was safely caught with an explicit error message. Second, for pure random white noise, the network predicted Highway with 88.8% confidence. This proves an essential finding: Softmax neural networks can be confidently wrong on out-of-distribution inputs."
