import albumentations as A

def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # Removed RandomRotate90 (not always meaningful in MRIs)
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ], p=0.6),
    ])

def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
    ])

def get_tta_transforms(img_size=224):
    return [
        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0)]),
        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0)]),
        A.Compose([A.Resize(img_size, img_size), A.Rotate(limit=10, p=1.0)]),
        A.Compose([A.Resize(img_size, img_size)]),
    ]
