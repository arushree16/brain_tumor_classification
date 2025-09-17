import argparse, os
import numpy as np
import tensorflow as tf
from optim.sam_optimizer import SAM
from model.hybrid_classifier_model import build_hybrid_model, build_hybrid_attention_cnn
from data.dataloader import list_images_and_labels
from data.augmentations import get_train_transforms, get_val_transforms
from data.preprocessing import build_dataset
from glob import glob


def get_model(name, num_classes, args):
    if name == 'efficientnet_hybrid':
        return build_hybrid_model(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=num_classes,
            d_model=args.d_model,
            num_heads=args.num_heads,
            trans_layers=args.trans_layers,
            dropout_rate=args.dropout_rate,
            pretrained=True
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
            trans_layers=args.trans_layers,
            pre_transform_downsample=args.pre_transform_downsample,
            downsample_type=args.downsample_type
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train(args):
    # -----------------------------
    # Dataset
    # -----------------------------
    train_root = os.path.join(args.data_root, 'train')
    val_root = os.path.join(args.data_root, 'val')

    train_files, train_labels, classes = list_images_and_labels(train_root)

    # Build val files and labels using the SAME class mapping as train
    class_to_idx = {c: i for i, c in enumerate(classes)}
    val_files, val_labels = [], []
    for c in classes:
        c_dir = os.path.join(val_root, c)
        if not os.path.isdir(c_dir):
            # skip missing class folders in val
            continue
        fpaths = [fp for fp in glob(os.path.join(c_dir, '*')) if fp.lower().endswith((".jpg", ".jpeg", ".png"))]
        val_files.extend(fpaths)
        val_labels.extend([class_to_idx[c]] * len(fpaths))

    train_aug = get_train_transforms(args.img_size)
    val_aug = get_val_transforms(args.img_size)

    train_ds = build_dataset(train_files, train_labels, train_aug,
                             batch_size=args.batch_size, shuffle=True, img_size=args.img_size)
    val_ds = build_dataset(val_files, val_labels, val_aug,
                           batch_size=args.batch_size, shuffle=False, img_size=args.img_size)

    # -----------------------------
    # Model + Optimizer + SAM
    # -----------------------------
    model = get_model(args.model, len(classes), args)
    model.summary()

    # -----------------------------
    # Stage 1: Freeze backbone (only relevant for EfficientNet variant)
    # -----------------------------
    if args.freeze_backbone and args.model == 'efficientnet_hybrid':
        for layer in model.layers:
            if 'efficientnetb0' in layer.name:
                layer.trainable = False

    # Learning rate scheduler
    steps_per_epoch = max(1, len(train_files) // args.batch_size)
    decay_steps = steps_per_epoch * args.epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=decay_steps,
        alpha=0.0
    )
    if args.weight_decay and args.weight_decay > 0.0:
        base_opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    else:
        base_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    sam = SAM(base_opt, rho=args.rho, adaptive=args.adaptive_sam)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val = 0.0
    ckpt_path = os.path.join(args.out_dir, 'saved_models', 'best_weights.weights.h5')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        losses = []

        # TRAIN
        for step, (bx, by) in enumerate(train_ds):
            if (args.max_steps_per_epoch is not None) and (step >= args.max_steps_per_epoch):
                break
            if args.use_sam:
                with tf.GradientTape() as tape:
                    logits = model(bx, training=True)
                    loss = loss_fn(by, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                e_ws = sam.first_step(grads, model.trainable_variables)
                with tf.GradientTape() as tape2:
                    logits2 = model(bx, training=True)
                    loss2 = loss_fn(by, logits2)
                grads2 = tape2.gradient(loss2, model.trainable_variables)
                sam.second_step(grads2, model.trainable_variables, e_ws)
            else:
                with tf.GradientTape() as tape:
                    logits = model(bx, training=True)
                    loss = loss_fn(by, logits)
                grads = tape.gradient(loss, model.trainable_variables)
                base_opt.apply_gradients(zip(grads, model.trainable_variables))

            losses.append(float(loss))
            if step % 50 == 0 and step > 0:
                print(f" step {step}, avg loss {np.mean(losses):.4f}")

        # VALIDATION
        acc_metric.reset_state()
        val_losses = []
        for vstep, (vx, vy) in enumerate(val_ds):
            if (args.max_val_steps is not None) and (vstep >= args.max_val_steps):
                break
            logits = model(vx, training=False)
            val_losses.append(float(loss_fn(vy, logits)))
            acc_metric.update_state(vy, logits)
        val_acc = acc_metric.result().numpy()
        val_loss = np.mean(val_losses) if val_losses else float('nan')
        print(f" Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        # Save best weights
        if val_acc > best_val:
            best_val = val_acc
            model.save_weights(ckpt_path)
            print(" Saved best weights to", ckpt_path)

    # -----------------------------
    # Stage 2: Optional fine-tune (EfficientNet variant)
    # -----------------------------
    if args.unfreeze_backbone and args.model == 'efficientnet_hybrid':
        print("\nUnfreezing backbone for fine-tuning...")
        for layer in model.layers:
            if 'efficientnetb0' in layer.name:
                layer.trainable = True

        # Reduce LR for fine-tuning
        base_opt.learning_rate = args.lr_finetune
        for epoch in range(1, args.finetune_epochs + 1):
            print(f"\nFine-tune Epoch {epoch}/{args.finetune_epochs}")
            losses = []
            for step, (bx, by) in enumerate(train_ds):
                if (args.max_steps_per_epoch is not None) and (step >= args.max_steps_per_epoch):
                    break
                if args.use_sam:
                    with tf.GradientTape() as tape:
                        logits = model(bx, training=True)
                        loss = loss_fn(by, logits)
                    grads = tape.gradient(loss, model.trainable_variables)
                    e_ws = sam.first_step(grads, model.trainable_variables)
                    with tf.GradientTape() as tape2:
                        logits2 = model(bx, training=True)
                        loss2 = loss_fn(by, logits2)
                    grads2 = tape2.gradient(loss2, model.trainable_variables)
                    sam.second_step(grads2, model.trainable_variables, e_ws)
                else:
                    with tf.GradientTape() as tape:
                        logits = model(bx, training=True)
                        loss = loss_fn(by, logits)
                    grads = tape.gradient(loss, model.trainable_variables)
                    base_opt.apply_gradients(zip(grads, model.trainable_variables))
                losses.append(float(loss))
            acc_metric.reset_state()
            val_losses = []
            for vstep, (vx, vy) in enumerate(val_ds):
                if (args.max_val_steps is not None) and (vstep >= args.max_val_steps):
                    break
                logits = model(vx, training=False)
                val_losses.append(float(loss_fn(vy, logits)))
                acc_metric.update_state(vy, logits)
            val_acc = acc_metric.result().numpy()
            val_loss = np.mean(val_losses) if val_losses else float('nan')
            print(f" Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                model.save_weights(ckpt_path)
                print(" Saved best weights to", ckpt_path)

    # Final save
    final_path = os.path.join(args.out_dir, 'saved_models', 'final_model.keras')
    model.save(final_path)
    print("Training complete. Best Val Acc =", best_val)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=3e-4)
    p.add_argument("--rho", type=float, default=0.02)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--trans_layers", type=int, default=1)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--pre_transform_downsample", action="store_true", help="Downsample feature map before Transformer")
    p.add_argument("--downsample_type", choices=["avg","conv"], default="avg")
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay (0 to use Adam)")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--adaptive_sam", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_backbone", action="store_true")
    p.add_argument("--model", choices=["efficientnet_hybrid", "hybrid_attention_cnn"], default="hybrid_attention_cnn")
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--stages", nargs='+', type=int, default=[1,2,3])
    # Lightweight run controls
    p.add_argument("--max_steps_per_epoch", type=int, default=None, help="Limit number of training steps per epoch")
    p.add_argument("--max_val_steps", type=int, default=None, help="Limit number of validation steps")
    p.add_argument("--use_sam", dest="use_sam", action="store_true", default=False, help="Enable SAM (two-step)")
    p.add_argument("--no_sam", dest="use_sam", action="store_false", help="Disable SAM (default)")
    args = p.parse_args()
    train(args)
