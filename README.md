# 🧠 Brain Tumor Classification with Hybrid CNN + Transformer + SAM

An end-to-end pipeline for brain tumor classification using two model variants:
- EfficientNet Hybrid (EfficientNetB0 + CBAM + Transformer bottleneck)
- Hybrid-Attention CNN (Depthwise Separable Convs + CBAM at multiple stages + Transformer bottleneck)

Training supports the Sharpness-Aware Minimization (SAM) optimizer for improved generalization. The project includes TTA for evaluation and Grad-CAM for explainability. Works locally and on Kaggle GPUs.

---

## ✨ Features

- Custom dataset loading with `tf.data` and Albumentations
- Augmentations: flips, light rotation, brightness/contrast, optional CLAHE
- Two model variants with attention and transformer bottlenecks
- SAM optimizer option for sharper minima and better generalization
- Explainability via Grad-CAM heatmaps
- TTA (Test-Time Augmentation) and simple ensemble in evaluation
- Clean command-line interface via `main.py` for train/eval/gradcam

---

## 🧠 Model Variants

- EfficientNet Hybrid (`build_hybrid_model` in `model/hybrid_classifier_model.py`)
  - Pretrained EfficientNetB0 backbone (frozen initially), then 1×1 projection to `d_model`
  - CBAM attention over backbone features
  - Transformer bottleneck block(s)
  - Head: GAP → Dropout → Dense logits
  - Recommended when you want strong transfer learning performance quickly

- Hybrid-Attention CNN (`build_hybrid_attention_cnn` in `model/hybrid_classifier_model.py`)
  - Custom encoder: Depthwise Separable Conv blocks (MobileNetV1 style) with stage-wise downsampling
  - CBAM applied at the end of each stage (multi-scale attention)
  - Optional pre-transform downsampling; 1×1 projection to `d_model`; Transformer bottleneck
  - Head: GAP → Dropout → Dense logits
  - Recommended as a novel, lightweight architecture; pairs well with SAM

Select the variant at runtime via `--model efficientnet_hybrid` or `--model hybrid_attention_cnn`.

---

## 📊 Dataset

This repo is compatible with the Masoud Nickparvar Brain Tumor MRI Dataset on Kaggle (PNG MRIs, 256×256, grayscale), but any image dataset with the same folder layout will work.

- Classes (typical): `glioma`, `meningioma`, `pituitary`, `no_tumor`
- Grayscale MRIs are auto-converted to RGB in the pipeline

Directory structure:

```
dataset/
  train/
    glioma/ ...
    meningioma/ ...
    pituitary/ ...
    no_tumor/ ...
  val/
    glioma/ ...
    meningioma/ ...
    pituitary/ ...
    no_tumor/ ...
  test/            # optional
```

---

## 📂 Project Structure

```
project/
  data/
    augmentations.py    # Augmentation pipeline (Albumentations)
    preprocessing.py    # Preprocessing & tf.data pipeline
    dataloader.py       # Dataset utilities

  model/
    hybrid_classifier_model.py  # EfficientNet Hybrid + Hybrid-Attention CNN
    cbam_module.py              # CBAM (Channel + Spatial Attention)
    dw_sep_conv.py              # Depthwise Separable Conv block
    transformer_bottleneck.py   # Transformer bottleneck block

  optim/
    sam_optimizer.py     # SAM optimizer wrapper

  explainability/
    grad_cam.py          # Grad-CAM visualizer

  outputs/
    saved_models/        # Model weights/checkpoints
    plots/               # Grad-CAM heatmaps
    prediction_scores/   # Evaluation CSVs

  train_sam.py           # Training script (invoked via main.py)
  evaluate.py            # Evaluation with TTA/ensemble (invoked via main.py)
  main.py                # Unified entrypoint: train / eval / gradcam
```

---

## 🚀 Usage

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train (via `main.py`)

Hybrid-Attention CNN (default configuration):

```bash
python project/main.py --cmd train \
  --data_root ./dataset \
  --model hybrid_attention_cnn \
  --img_size 256 --batch_size 32 --epochs 20 \
  --base_ch 32 --stages 1 2 3 \
  --d_model 128 --num_heads 4 --trans_layers 1 \
  --dropout_rate 0.2 \
  --use_sam
```

EfficientNet Hybrid:

```bash
python project/main.py --cmd train \
  --data_root ./dataset \
  --model efficientnet_hybrid \
  --img_size 256 --batch_size 32 --epochs 10 \
  --d_model 128 --num_heads 4 --trans_layers 1 \
  --dropout_rate 0.2 \
  --use_sam \
  --freeze_backbone --unfreeze_backbone --finetune_epochs 10
```

### SAM on/off

- `--use_sam` to enable SAM (two-step update)
- `--no_sam` to disable SAM (single-step Adam/AdamW)

Lightweight sanity run:

```bash
python project/main.py --cmd train \
  --data_root ./dataset \
  --model hybrid_attention_cnn \
  --img_size 256 --epochs 1 --batch_size 8 --no_sam \
  --max_steps_per_epoch 20 --max_val_steps 10
```

### Scaling to larger datasets / higher resolution

- Downsample feature map before Transformer to reduce tokens and memory:

```bash
--pre_transform_downsample --downsample_type avg   # or: conv
```

- Add weight decay for larger datasets (switches to AdamW):

```bash
--weight_decay 1e-4
```

Example (high-res friendly):

```bash
python project/main.py --cmd train \
  --data_root ./dataset \
  --model hybrid_attention_cnn \
  --img_size 256 --epochs 30 --batch_size 32 --use_sam \
  --pre_transform_downsample --downsample_type avg --weight_decay 1e-4 \
  --base_ch 32 --d_model 128 --num_heads 4 --trans_layers 1
```

### 3) Evaluate (supports TTA and ensemble)

Single checkpoint with TTA:

```bash
python project/main.py --cmd eval \
  --data_root ./dataset \
  --model hybrid_attention_cnn \
  --img_size 256 --batch_size 32 \
  --weights outputs/saved_models/best_weights.weights.h5 \
  --tta
```

Ensemble of checkpoints (comma separated):

```bash
python project/main.py --cmd eval \
  --data_root ./dataset \
  --model hybrid_attention_cnn \
  --img_size 256 --batch_size 32 \
  --weights outputs/saved_models/best1.weights.h5,outputs/saved_models/best2.weights.h5 \
  --tta
```

### 4) Grad-CAM heatmaps

```bash
python project/main.py --cmd gradcam \
  --data_root ./dataset \
  --model hybrid_attention_cnn \
  --img_size 256 \
  --weights outputs/saved_models/best_weights.weights.h5
```

Use `--model efficientnet_hybrid` for the EfficientNet variant.

---

## ⚙️ Quick start (your absolute paths)

Train:

```bash
python /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/main.py \
  --cmd train \
  --data_root /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/dataset \
  --model hybrid_attention_cnn --img_size 256 --epochs 20 --batch_size 32 --use_sam \
  --out_dir /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/outputs
```

Evaluate with TTA:

```bash
python /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/main.py \
  --cmd eval \
  --data_root /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/dataset \
  --model hybrid_attention_cnn --img_size 256 --batch_size 32 \
  --weights /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/outputs/saved_models/best_weights.weights.h5 \
  --tta
```

Grad-CAM:

```bash
python /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/main.py \
  --cmd gradcam \
  --data_root /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/dataset \
  --model hybrid_attention_cnn --img_size 256 \
  --weights /Users/arushreemishra/Downloads/brain_tumor/brain_tumor_classification/project/outputs/saved_models/best_weights.weights.h5
```

---

## 📝 Notes

- Images are preprocessed with EfficientNet's `preprocess_input` inside the TF pipeline; augmentation `Normalize` has been removed to avoid double-normalization.
- Models output logits (no softmax); losses use `from_logits=True`.
- Validation labels are encoded using the training split's class order to avoid mismatches.

---

## 🧮 Expected performance

Actual performance depends on split quality and training budget. As a rough guide on standard splits with TTA and SAM:

- Accuracy: ~90–98%
- Weighted F1: ~0.90–0.96
- Macro ROC-AUC: ~0.95–0.98

---

## 🖥️ Hardware guidance

- Recommended: NVIDIA GPU with ≥ 8 GB VRAM for 256×256, batch size 32
- On smaller GPUs, reduce `--batch_size` or `--img_size`

---

## 📌 Future Improvements

- Explore alternative transformer variants (Swin, ViT) and attention modules
- Hyperparameter search (LR, d_model, depth) and better regularization
- Multi-class explainability (e.g., Grad-CAM++)
