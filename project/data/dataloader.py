import os
from glob import glob
import random

def list_images_and_labels(root_dir, seed=42, extensions=(".jpg", ".jpeg", ".png")):
    """
    Scan a dataset folder structured as:
        root_dir/
            class1/
                img1.png, img2.jpg ...
            class2/
                img3.png, img4.jpg ...
    Returns:
        files:  list of file paths
        labels: list of int labels
        classes: sorted list of class names
    """
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    files, labels = [], []
    for c in classes:
        pattern = os.path.join(root_dir, c, "*")
        f = [fp for fp in glob(pattern) if fp.lower().endswith(extensions)]
        files.extend(f)
        labels.extend([class_to_idx[c]] * len(f))

    if not files:
        raise ValueError(f"No image files found in {root_dir}. Check dataset path/structure.")

    # Shuffle deterministically for reproducibility
    paired = list(zip(files, labels))
    random.seed(seed)
    random.shuffle(paired)
    files, labels = zip(*paired)

    return list(files), list(labels), classes
