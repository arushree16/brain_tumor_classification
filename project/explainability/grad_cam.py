import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, os, cv2, numpy as np, tensorflow as tf
from model.hybrid_classifier_model import build_hybrid_model, build_hybrid_attention_cnn
from data.dataloader import list_images_and_labels
from tensorflow.keras.applications.efficientnet import preprocess_input


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.SeparableConv2D):
            return layer.name
    raise ValueError('No conv layer found')


def gradcam_heatmap(model, img_tensor, class_idx=None, last_conv_name=None):
    if last_conv_name is None:
        last_conv_name = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv_outputs[0].numpy()
    pooled = pooled.numpy()
    heatmap = np.zeros(conv.shape[:2], dtype=np.float32)
    for i, w in enumerate(pooled):
        heatmap += w * conv[:,:,i]
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    hmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    hmap = np.uint8(255 * hmap)
    hmap = cv2.applyColorMap(hmap, colormap)
    overlay = cv2.addWeighted(img.astype(np.uint8), 1-alpha, hmap, alpha, 0)
    return overlay


def load_image(path, img_size, preprocess='efficientnet'):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))

    x = img_resized.astype('float32')
    if preprocess == 'efficientnet':
        x = preprocess_input(x)
    elif preprocess == 'zero_one':
        x = x / 255.0
    elif preprocess == 'minus1_1':
        x = (x / 127.5) - 1.0
    else:
        raise ValueError('Unknown preprocess mode')

    return img, x


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', default='outputs/plots')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--preprocess', choices=['efficientnet','zero_one','minus1_1'], default='efficientnet')
    p.add_argument('--model', choices=['efficientnet_hybrid', 'hybrid_attention_cnn'], default='hybrid_attention_cnn')
    p.add_argument('--base_ch', type=int, default=32)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--trans_layers', type=int, default=1)
    p.add_argument('--dropout_rate', type=float, default=0.2)
    p.add_argument('--stages', nargs='+', type=int, default=[1,2,3])
    args = p.parse_args()

    _, _, classes = list_images_and_labels(os.path.join(args.data_root, 'train'))
    model = get_model(args.model, len(classes), args)
    model.load_weights(args.weights)

    os.makedirs(args.out_dir, exist_ok=True)
    val_dir = os.path.join(args.data_root, 'val')
    for cls in classes:
        folder = os.path.join(val_dir, cls)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))][:3]
        for i, fpath in enumerate(files):
            orig, x = load_image(fpath, args.img_size, preprocess=args.preprocess)
            x_tensor = tf.expand_dims(x, 0)
            heatmap = gradcam_heatmap(model, x_tensor)
            overlay = overlay_heatmap(orig, heatmap)
            outp = os.path.join(args.out_dir, f"{cls}_{i}.png")
            cv2.imwrite(outp, overlay)
            print('wrote', outp)

if __name__ == '__main__':
    main()
