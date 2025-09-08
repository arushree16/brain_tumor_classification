import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

def ensure_rgb(img):
    if img is None:
        # Return a black placeholder image if read fails
        return np.zeros((224, 224, 3), dtype=np.uint8)
    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:  # single-channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# Albumentations augmentation wrapper
def albumentations_wrapper(aug, img_size=224):
    def _apply(img):
        # img is already a NumPy array from tf.numpy_function
        if img is None or img.size == 0:
            img_np = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            img_np = ensure_rgb(img)
            img_np = img_np.astype(np.uint8)

        if aug is not None:
            try:
                img_np = aug(image=img_np)['image']
            except Exception as e:
                print(f"[WARN] Albumentations failed: {e}")
                img_np = cv2.resize(img_np, (img_size, img_size))

        return img_np.astype(np.float32)
    return _apply

def build_dataset(file_list, label_list, aug, batch_size=16, shuffle=True, img_size=224):
    file_ds = tf.data.Dataset.from_tensor_slices(file_list)
    label_ds = tf.data.Dataset.from_tensor_slices(label_list)
    ds = tf.data.Dataset.zip((file_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)

    def _map(path, label):
        # Read image
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)

        # Apply albumentations with tf.numpy_function
        if aug is not None:
            img = tf.numpy_function(
                func=albumentations_wrapper(aug, img_size),
                inp=[img],
                Tout=tf.float32
            )
        else:
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32)

        img.set_shape((img_size, img_size, 3))
        label.set_shape(())
        return img, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
