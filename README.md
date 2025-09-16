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
- 🧪 TTA (Test Time Augmentation) and simple ensemble in evaluation
- 🧩 New model: Hybrid-Attention CNN (DWConv + CBAM + ViT bottleneck)

---

## 📊 Dataset

This project uses the **[Masoud Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** (PNG MRIs, 256×256, grayscale).

- Classes: `glioma`, `meningioma`, `pituitary`, `no_tumor`
- Note: Images are grayscale but the pipeline converts them to RGB automatically to match model expectations.

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
    hybrid_classifier_model.py: "EfficientNetB0 + CBAM + Transformer + Hybrid-Attention CNN"

  optim/
    sam_optimizer.py: "SAM optimizer implementation"

  explainability/
    grad_cam.py: "Grad-CAM visualizer"

  outputs/
    saved_models/: "Model weights/checkpoints"
    plots/: "Grad-CAM heatmaps"
    prediction_scores/: "Evaluation CSVs"

  train_sam.py: "Training script"
  evaluate.py: "Evaluation script with TTA/ensemble"
  main.py: "Unified entrypoint"
```

---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Hybrid-Attention CNN (default):

```bash
python project/train_sam.py --data_root ./dataset --out_dir ./outputs \
  --model hybrid_attention_cnn --img_size 256 --epochs 20
```

EfficientNet Hybrid:

```bash
python project/train_sam.py --data_root ./dataset --out_dir ./outputs \
  --model efficientnet_hybrid --freeze_backbone --epochs 10 --unfreeze_backbone --finetune_epochs 10
```

### SAM on/off

- `--use_sam`: enable SAM (two-step update, better generalization)
- `--no_sam`: disable SAM (single-step Adam, faster/lighter). Default is disabled.

Examples:

- Full run with SAM:

```bash
python project/train_sam.py --data_root ./dataset --model hybrid_attention_cnn \
  --img_size 256 --epochs 20 --batch_size 32 --use_sam
```

- Lightweight sanity run (no SAM, capped steps):

```bash
python project/train_sam.py --data_root ./dataset --model hybrid_attention_cnn \
  --img_size 256 --epochs 1 --batch_size 8 --no_sam \
  --max_steps_per_epoch 20 --max_val_steps 10
```

### Scaling to larger datasets / higher resolution

- Reduce attention tokens to avoid OOM: downsample feature map before Transformer.

```bash
--pre_transform_downsample --downsample_type avg   # or: conv
```

- Add weight decay for better generalization on big data:

```bash
--weight_decay 1e-4   # switches optimizer to AdamW
```

- Example (high-res friendly):

```bash
python project/train_sam.py --data_root ./dataset --model hybrid_attention_cnn \
  --img_size 256 --epochs 30 --batch_size 32 --use_sam \
  --pre_transform_downsample --downsample_type avg --weight_decay 1e-4 \
  --base_ch 32 --d_model 128 --num_heads 4 --trans_layers 1
```

### 3. Evaluate the model (supports TTA and ensemble)

Single checkpoint, with TTA:

```bash
python project/evaluate.py --data_root ./dataset \
  --weights outputs/saved_models/best_weights.weights.h5 \
  --model hybrid_attention_cnn --tta
```

Ensemble of checkpoints (comma separated):

```bash
python project/evaluate.py --data_root ./dataset \
  --weights outputs/saved_models/best1.weights.h5,outputs/saved_models/best2.weights.h5 \
  --model hybrid_attention_cnn --tta
```

### 4. Generate Grad-CAM heatmaps

```bash
python project/explainability/grad_cam.py \
  --weights outputs/saved_models/best_weights.weights.h5 \
  --data_root ./dataset --model hybrid_attention_cnn --img_size 256 \
  --preprocess efficientnet
```

---

## ⚙️ Quick start (your absolute paths)

- Train:

```bash
python /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/train_sam.py \
  --data_root /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/dataset \
  --model hybrid_attention_cnn --img_size 256 --epochs 20 --batch_size 32 --use_sam \
  --out_dir /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/outputs
```

- Evaluate with TTA:

```bash
python /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/evaluate.py \
  --data_root /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/dataset \
  --weights /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/outputs/saved_models/best_weights.weights.h5 \
  --model hybrid_attention_cnn --tta
```

- Grad-CAM:

```bash
python /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/explainability/grad_cam.py \
  --weights /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/outputs/saved_models/best_weights.weights.h5 \
  --data_root /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/dataset \
  --model hybrid_attention_cnn --img_size 256 --preprocess efficientnet
```

---

## 📝 Notes

- ✅ Augmentations include **random CLAHE** for contrast enhancement
- ✅ Models are saved in both **`.weights.h5`** and **`.keras`** formats
- ✅ Training works on **local machines** and **Kaggle GPUs**
- ✅ Evaluation shows Accuracy, F1, macro ROC-AUC, and saves prediction CSV
- ✅ Grayscale MRIs are auto-converted to RGB for model compatibility

---

## 🧮 Expected performance

- Accuracy: 95% – 98%
- F1 Score (weighted): 0.93 – 0.96
- ROC-AUC (macro): 0.96 – 0.98
- Inference: < 50 ms per image (single GPU)
- Model size: ~1M–5M parameters

---

## 🖥️ Hardware guidance

- Recommended: NVIDIA GPU with ≥ 8 GB VRAM for 256×256, batch size 32
- On smaller GPUs, reduce `--batch_size` (e.g., 16 or 8)

---

## 📌 Future Improvements

- 🔬 Add advanced Transformer variants (e.g., Swin, ViT)
- ⚙️ Hyperparameter tuning for better performance
- 🧩 Multi-class explainability with Grad-CAM++

---
