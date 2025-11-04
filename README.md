# Embed3D — Fault Detection in 3D Industrial Scans

**Embed3D** is a deep learning project for **fault and defect segmentation** in 3D industrial scans using a **U-Net** architecture.
It leverages the **[pmchard/3D-ADAM](https://huggingface.co/datasets/pmchard/3D-ADAM)** dataset and performs segmentation to detect anomalies such as structural defects and manufacturing inconsistencies.

---

## Overview

The pipeline integrates:

* **Data extraction** (10 % subset from Hugging Face)
* **Custom dataset loader**
* **U-Net-based binary segmentation**
* **Training with early stopping and learning-rate scheduling**
* **Evaluation** using IoU and Dice metrics
* **Visualization** of predicted masks and overlays
* **Fault prediction** on custom input images

---

## Architecture

A lightweight **U-Net** model is used for pixel-wise segmentation:

* Encoder–decoder structure with skip connections
* Batch normalization and dropout for regularization
* Trained using `CrossEntropyLoss` and optimized with `Adam`

---

## Dependencies

Install requirements via:

```bash
pip install -r requirements.txt
```

**Key libraries:**
`torch`, `torchvision`, `datasets`, `matplotlib`, `opencv-python`, `numpy`, `Pillow`, `tqdm`

---

## Project Structure

```
Embed3D/
├──src_1
  ├── main.py                 # Main training and orchestration script
  ├── model_unet.py           # U-Net architecture
  ├── dataset_class.py        # Custom dataset loader for 3D-ADAM
  ├── data_module.py          # Data preparation utilities
  ├── trainer.py              # Training loop with logging and early stopping
  ├── eval.py                 # Evaluation metrics (IoU, Dice)
  ├── visualisations.py       # Visualization of training and predictions
  ├── predictions.py          # Fault prediction on single images
  ├── utils.py                # Helper utilities
  ├── checkpoints/            # Saved model weights
  └── results/                # Evaluation outputs and visualizations
├── .gitignore
└── requirements.txt
```

---

## Training & Evaluation

Run the full pipeline:

```bash
python src_1/main.py
```

Steps performed:

1. Load cached 10 % subset of the 3D-ADAM dataset (`adam_5pct_subset.pkl`)
2. Split into train / validation / test sets
3. Train model with early stopping
4. Save best model weights and full model pickle
5. Evaluate on validation and test sets
6. Generate visualizations and fault overlays

**Evaluation Metrics**

| Metric                            | Description                                      |
| --------------------------------- | ------------------------------------------------ |
| **IoU (Intersection over Union)** | Overlap between predicted and ground truth masks |
| **Dice Coefficient**              | Measures segmentation accuracy                   |

---

## Visualizations

After training, you’ll find results under `results/visuals/`:

* `sample_*.png`: Input, ground truth, predicted mask, probability overlay
* `loss_curve.png`: Training and validation loss curves

---

## Fault Prediction

You can test the model on your own image:

```bash
python main.py --predict sample_image.png
```

Predictions are saved under:

```
results/predictions/fault_overlay.png
```

This overlay highlights regions detected as **faults or defects**.

---

## Example Outputs

*  Dice Score ≈ 0.94
*  IoU Score ≈ 0.89
*  Fault overlay images with clear visual localization

---

## Future Work

* Integrate **3D U-Net** for volumetric anomaly detection
* Deploy interactive inference app with **Streamlit or Gradio**
* Explore **Vision Transformers** (ViT) for improved generalization
* Incorporate explainability via **Grad-CAM**

---
