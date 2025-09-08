import argparse, os
import numpy as np
import tensorflow as tf
from data.dataloader import list_images_and_labels
from data.augmentations import get_val_transforms
from data.preprocessing import build_dataset
from model.hybrid_classifier_model import build_hybrid_model
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import csv

def evaluate(args):
    _, _, classes = list_images_and_labels(os.path.join(args.data_root, 'train'))
    val_files, val_labels, _ = list_images_and_labels(os.path.join(args.data_root, 'val'))

    val_aug = get_val_transforms(args.img_size)
    val_ds = build_dataset(
        val_files, val_labels, val_aug,
        batch_size=args.batch_size, shuffle=False, img_size=args.img_size
    )

    model = build_hybrid_model(
        input_shape=(args.img_size,args.img_size,3),
        num_classes=len(classes),
        base_ch=args.base_ch,
        d_model=args.d_model,
        num_heads=args.num_heads,
        trans_layers=args.trans_layers
    )
    model.load_weights(args.weights)

    ys, preds = [], []
    for bx, by in val_ds:
        logits = model(bx, training=False).numpy()  # bx is already normalized
        p = np.argmax(logits, axis=1)
        ys.append(by.numpy())
        preds.append(p)
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)

    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average='weighted')
    cm = confusion_matrix(ys, preds)
    report = classification_report(ys, preds)
    print('Accuracy:', acc)
    print('F1 Score:', f1)
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
    p.add_argument('--weights', required=True)
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--trans_layers', type=int, default=1)
    args = p.parse_args()
    evaluate(args)
