# Skin Cancer Classifier

This repository contains our project on multi-class skin lesion classification using dermoscopic and smartphone images. We compare two approaches:

1. **Primary Model** – EfficientNet-B2 CNN trained end-to-end on the ISIC 2019 Challenge dataset, fine-tuned for 8-class classification.
2. **Baseline Model** – LinearSVC classifier using ResNet-extracted features.

---

## Models in This Repository

- `progress_primary_model.ipynb` – Primary model code and results at the **Progress Report** stage, trained on the **HAM10000 dataset (7-class classification)**.
- `progress_base_model.ipynb` – Baseline model code and results at the **Progress Report** stage, trained on the **HAM10000 dataset (7-class classification)**.
- `final_primary_model.ipynb` – Finalized primary model code and results used in the **Final Report**, trained on the **ISIC 2019 dataset (8-class classification)**.
- `final_base_model.ipynb` – Finalized baseline model code and results used in the **Final Report**, trained on the **ISIC 2019 dataset (8-class classification)**.

Model training details, hyperparameters, and evaluation procedures are documented in the provided Jupyter notebooks.

---

## Final Performance (ISIC 2019)

| Model             | Validation Accuracy | Test Accuracy |
|-------------------|---------------------|---------------|
| **Primary CNN**   | **85.3%**           | **81.1%**     |
| **Baseline SVM**  | **49.3%**           | **61.0%**     |

**Note:**  
Skin lesions are inherently challenging to classify, even for dermatologists, due to subtle visual similarities between benign and malignant lesions. The ISIC 2019 dataset is highly imbalanced, with the majority class (**Nevus, NV**) comprising over 50% of samples, while some classes (e.g., DF, VASC) have fewer than 260 examples. While we used targeted data augmentation and class-balanced sampling to reduce bias, these techniques cannot fully eliminate imbalance effects.

---

## Dataset

- **HAM10000** – ~10,000 dermatoscopic images used for the **progress stage (7 classes)**.  
  Provided a simpler dataset for initial experimentation before moving to ISIC 2019.  

- **ISIC 2019 Challenge Dataset** – 25,331 training images, 8,238 test images  
  - Aggregates **BCN_20000**, **HAM10000**, and **MSK** datasets.  
  - Over 50% of diagnoses confirmed by histopathology, with the rest verified by expert consensus, follow-up, or confocal microscopy.  
  - Diverse acquisition devices and protocols to improve generalization.

- **PAD-UFES-20** – ~900 smartphone images for external testing  
  - Adds real-world variability beyond high-quality dermatoscope images.

---

## Primary Model (EfficientNet-B2)

- **Base Model:** EfficientNet-B2 (ImageNet-pretrained)  
- **Input Resolution:** 260×260 RGB  
- **Final Layer:** 8-unit output layer for lesion categories  
- **Loss Function:** Weighted Cross Entropy  
- **Optimizer:** AdamW (LR=1e-4, weight decay=1e-5)  
- **Batch Size:** 64, trained for 20–30 epochs with early stopping  
- **Augmentation:** Horizontal/vertical flips, rotation, color jitter, AutoAugment, TrivialAugmentWide, class-balanced sampling  

---

## Acknowledgements

- HAM10000 Dataset: [HAM10000](https://www.nature.com/articles/sdata2018161)  
- ISIC 2019 Challenge Dataset: [ISIC Archive](https://challenge.isic-archive.com/)  
- PAD-UFES-20 Dataset: [PAD-UFES-20 GitHub](https://github.com/tiagoespsantos/pad-ufes-20)  
- EfficientNet-B2: [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)

---
