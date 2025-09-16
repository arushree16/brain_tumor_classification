import argparse, os
import numpy as np
import tensorflow as tf
from data.dataloader import list_images_and_labels
from data.augmentations import get_val_transforms, get_tta_transforms
from data.preprocessing import build_dataset
from model.hybrid_classifier_model import build_hybrid_model, build_hybrid_attention_cnn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import csv


def get_model(name, num_classes, args):
    if name == 'efficientnet_hybrid':
        return build_hybrid_model(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=num_classes,
            d_model=args.d_model,
            num_heads=args.num_heads,
            trans_layers=args.trans_layers
        )
    elif name == 'hybrid_attention_cnn':
        return build_hybrid_attention_cnn(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=num_classes,
            base_filters=args.base_ch,
            stages=tuple(args.stages),
            dropout_rate=args.dropout_rate,
            d_model=args.d_model,
            num_heads=args.num_heads,
            trans_layers=args.trans_layers
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate(args):
    _, _, classes = list_images_and_labels(os.path.join(args.data_root, 'train'))
    val_files, val_labels, _ = list_images_and_labels(os.path.join(args.data_root, 'val'))

    val_aug = get_val_transforms(args.img_size)
    val_ds = build_dataset(
        val_files, val_labels, val_aug,
        batch_size=args.batch_size, shuffle=False, img_size=args.img_size
    )

    model = get_model(args.model, len(classes), args)

    weight_paths = args.weights.split(',') if ',' in args.weights else [args.weights]
    logits_accum = None

    ys = []
    # Pre-collect all images into memory for TTA to avoid re-iterating dataset objects with numpy_function state
    xs = []
    for bx, by in val_ds:
        xs.append(bx.numpy())
        ys.append(by.numpy())
    xs = np.concatenate(xs, axis=0)
    ys = np.concatenate(ys, axis=0)

    for w in weight_paths:
        model.load_weights(w)
        # TTA
        if args.tta:
            tta_transforms = get_tta_transforms(args.img_size)
            logits_sum = np.zeros((xs.shape[0], len(classes)), dtype=np.float32)
            for t in tta_transforms:
                # Apply albumentations on CPU and preprocess consistent with training
                aug_xs = []
                for img in xs:
                    img_uint8 = ((img + 1.0) * 127.5).astype(np.uint8) if img.min() < 0 else (img * 255.0).astype(np.uint8)
                    aug_x = t(image=img_uint8)['image'].astype(np.float32)
                    # EfficientNet-style preprocess_input already applied in pipeline; here we keep zero-mean unit scaling
                    aug_xs.append(aug_x)
                aug_xs = np.stack(aug_xs, axis=0)
                logits_sum += model.predict(aug_xs, batch_size=args.batch_size, verbose=0)
            logits = logits_sum / len(tta_transforms)
        else:
            logits = model.predict(xs, batch_size=args.batch_size, verbose=0)

        logits_accum = logits if logits_accum is None else (logits_accum + logits)

    logits = logits_accum / len(weight_paths)

    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average='weighted')
    cm = confusion_matrix(ys, preds)
    report = classification_report(ys, preds)
    # Macro ROC-AUC using one-vs-rest on softmax probs
    probs = tf.nn.softmax(logits, axis=1).numpy()
    try:
        auc = roc_auc_score(ys, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    print('Accuracy:', acc)
    print('F1 Score:', f1)
    print('ROC-AUC (macro ovr):', auc)
    print(report)

    # Save predictions
    out_dir = os.path.join(args.out_dir, 'prediction_scores')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'prediction_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'true', 'pred'])
        for i, (t, p) in enumerate(zip(ys, preds)):
            writer.writerow([i, int(t), int(p)])
    print(f"Saved predictions to {csv_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--weights', required=True, help='Single path or comma-separated paths for ensemble')
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--trans_layers', type=int, default=1)
    p.add_argument('--dropout_rate', type=float, default=0.2)
    p.add_argument('--stages', nargs='+', type=int, default=[1,2,3])
    p.add_argument('--model', choices=['efficientnet_hybrid', 'hybrid_attention_cnn'], default='hybrid_attention_cnn')
    p.add_argument('--tta', action='store_true')
    args = p.parse_args()
    evaluate(args)
