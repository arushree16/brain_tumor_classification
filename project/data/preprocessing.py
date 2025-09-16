import cv2
import numpy as np
import tensorflow as tf
import albumentations as A
from tensorflow.keras.applications.efficientnet import preprocess_input

def ensure_rgb(img):
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:  # single-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def albumentations_wrapper(aug, img_size=224):
    def _apply(img):
        img = ensure_rgb(img)
        img = cv2.resize(img, (img_size, img_size))
        if aug is not None:
            try:
                img = aug(image=img)['image']
            except Exception as e:
                print(f"[WARN] Albumentations failed: {e}")
        # EfficientNet preprocessing
        img = preprocess_input(img)
        return img.astype(np.float32)
    return _apply

def build_dataset(file_list, label_list, aug=None, batch_size=16, shuffle=True, img_size=224):
    file_ds = tf.data.Dataset.from_tensor_slices(file_list)
    label_ds = tf.data.Dataset.from_tensor_slices(label_list)
    ds = tf.data.Dataset.zip((file_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)

    def _map(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)

        if aug is not None:
            img = tf.numpy_function(
                func=albumentations_wrapper(aug, img_size),
                inp=[img],
                Tout=tf.float32
            )
        else:
            img = tf.image.resize(img, [img_size, img_size])
            img = preprocess_input(img)
            img = tf.cast(img, tf.float32)

        img.set_shape((img_size, img_size, 3))
        label.set_shape(())
        return img, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
