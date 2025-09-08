import argparse, os
import numpy as np
import tensorflow as tf
from optim.sam_optimizer import SAM
from model.hybrid_classifier_model import build_hybrid_model
from data.dataloader import list_images_and_labels
from data.augmentations import get_train_transforms, get_val_transforms
from data.preprocessing import build_dataset

def train(args):
    # -----------------------------
    # Dataset
    # -----------------------------
    train_files, train_labels, classes = list_images_and_labels(os.path.join(args.data_root, 'train'))
    val_files, val_labels, _ = list_images_and_labels(os.path.join(args.data_root, 'val'))

    train_aug = get_train_transforms(args.img_size)
    val_aug = get_val_transforms(args.img_size)

    train_ds = build_dataset(
        train_files, train_labels, train_aug,
        batch_size=args.batch_size, shuffle=True, img_size=args.img_size
    )
    val_ds = build_dataset(
        val_files, val_labels, val_aug,
        batch_size=args.batch_size, shuffle=False, img_size=args.img_size
    )

    # -----------------------------
    # Model + Optimizer + SAM
    # -----------------------------
    model = build_hybrid_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(classes),
        base_ch=args.base_ch,
        d_model=args.d_model,
        num_heads=args.num_heads,
        trans_layers=args.trans_layers
    )
    model.summary()

    base_opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    sam = SAM(base_opt, rho=args.rho, adaptive=args.adaptive_sam)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # -----------------------------
    # Training Loop
    # -----------------------------
    best_val = 0.0
    ckpt_path = os.path.join(args.out_dir, 'saved_models', 'best_weights.weights.h5')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        losses = []

        # TRAIN
        for step, (bx, by) in enumerate(train_ds):
            # bx is already float32 and normalized in build_dataset
            with tf.GradientTape() as tape:
                logits = model(bx, training=True)
                loss = loss_fn(by, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            e_ws = sam.first_step(grads, model.trainable_variables)

            # Second forward-backward
            with tf.GradientTape() as tape2:
                logits2 = model(bx, training=True)
                loss2 = loss_fn(by, logits2)
            grads2 = tape2.gradient(loss2, model.trainable_variables)
            sam.second_step(grads2, model.trainable_variables, e_ws)

            losses.append(float(loss))
            if step % 50 == 0 and step > 0:
                print(f" step {step}, avg loss {np.mean(losses):.4f}")

        # VALIDATION
        acc_metric.reset_state()  # fixed method name
        for vx, vy in val_ds:
            logits = model(vx, training=False)
            acc_metric.update_state(vy, logits)
        val_acc = acc_metric.result().numpy()
        print(f" Val Acc: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            model.save_weights(ckpt_path)
            print(" Saved best weights to", ckpt_path)

    # Final save (SavedModel format)
    final_path = os.path.join(args.out_dir, 'saved_models', 'final_model.keras')
    model.save(final_path)
    print("Training complete. Best Val Acc =", best_val)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--trans_layers", type=int, default=1)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--adaptive_sam", action="store_true")
    args = p.parse_args()
    train(args)
