# Distorted Visual Sequence Pattern Recognition using Deep Learning
**CIG AI/ML Open Project 2026**

---

## Problem Statement
Build a deep learning model that can correctly recognize and reconstruct text sequences from visually distorted grayscale CAPTCHA images containing background noise, overlapping symbols, blur, shape deformation, and random patches.

---

## Approach
This problem is treated as a **fixed-length sequence classification task** since all CAPTCHAs are exactly 6 characters long.

Instead of using CTC loss (designed for variable-length sequences), we use a **pretrained ResNet18 backbone** modified for grayscale input, with a fully connected head that predicts all 6 character positions simultaneously.

### Model Architecture
- **Backbone:** ResNet18 (pretrained on ImageNet)
- **Input:** Grayscale images resized to `100 × 200` pixels
- **Head:** Linear(512 → 6 × 31) reshaped to `[batch, 6, 31]`
- **Loss:** Cross-Entropy averaged across all 6 positions
- **Optimizer:** Adam (lr = 1e-4)

---

## Dataset
- **Training set:** 20,000 labeled CAPTCHA images
- **Test set:** 5,000 unlabeled images
- **Vocabulary:** 31 characters (digits 2–9, uppercase A–Z excluding O, I, L)
- **Label length:** Fixed at 6 characters

---

## Results

| Metric | Value |
|---|---|
| Character Accuracy | 99.30% |
| Full CAPTCHA Accuracy | 95.85% |
| Approximate CER | 0.70% |
| Best Validation Loss | 0.1014 |
| Epochs Trained | 10 |

---


## How to Run
1. Upload `dataset.zip` to your Google Drive at `MyDrive/OCR_Project/dataset.zip`
2. Open `CIG_OCR_Submission.ipynb` in Google Colab
3. Set runtime to **GPU (T4)**
4. Run all cells top to bottom
5. `submission.csv` will be auto-downloaded at the end

---

## Tech Stack
- Python 3.10
- PyTorch 2.x
- Torchvision (ResNet18 pretrained weights)
- Pillow, pandas, scikit-learn, tqdm, matplotlib
