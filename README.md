# 🧠 Brain Tumor Classification with Hybrid CNN + Transformer + SAM

An end-to-end pipeline for **brain tumor classification** using a **hybrid CNN–Transformer architecture**, trained with the **Sharpness-Aware Minimization (SAM) optimizer** for improved generalization.

---

## ✨ Features

- 📦 Custom dataset loading with **`tf.data`** and **Albumentations**
- 🖼️ Advanced augmentations: CLAHE, flips, rotations, brightness/contrast
- 🧠 Hybrid CNN + Transformer architecture for rich feature learning
- ⚡ SAM optimizer for sharper minima & better generalization
- 🔍 Explainability via Grad-CAM heatmaps
- 🖥️ Compatible with local setups and Kaggle GPUs

---

## 📊 Dataset

This project uses the **[Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets)**

Dataset structure should look like this:

```yaml
dataset/
test/
glioma/
meningioma/
pituitary/
no_tumor/
train/
glioma/
meningioma/
pituitary/
no_tumor/
val/
glioma/
meningioma/
pituitary/
no_tumor/
```

---

## 📂 Project Structure

```yaml
project/
  data/
    augmentations.py: "Augmentation pipeline"
    preprocessing.py: "Preprocessing & tf.data pipeline"
    dataloader.py: "Dataset utilities"

  model/
    hybrid_classifier_model.py: "CNN + Transformer hybrid model"

  optim/
    sam_optimizer.py: "SAM optimizer implementation"

  explainability/
    grad_cam.py: "Grad-CAM visualizer"

  outputs/
    saved_models/: "Model weights/checkpoints"

  train_sam.py: "Training script"
  evaluate.py: "Evaluation script"
  main.py: "Unified entrypoint"
```

---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_sam.py --data_root ./dataset --out_dir ./outputs --epochs 10
```

### 3. Evaluate the model

```bash
python evaluate.py --data_root ./dataset --weights outputs/saved_models/best_weights.weights.h5
```

### 4. Generate Grad-CAM heatmaps

```bash
python main.py --cmd gradcam --data_root ./dataset --weights outputs/saved_models/best_weights.weights.h5
```

---

## 📝 Notes

- ✅ Augmentations include **random CLAHE** for contrast enhancement
- ✅ Models are saved in both **`.weights.h5`** and **`.keras`** formats
- ✅ Training works on **local machines** and **Kaggle GPUs**

---

## 📌 Future Improvements

- 🔬 Add advanced Transformer variants (e.g., Swin, ViT)
- ⚙️ Hyperparameter tuning for better performance
- 🧩 Multi-class explainability with Grad-CAM++

---
