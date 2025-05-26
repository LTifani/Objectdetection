
import os
import random
import shutil
import numpy as np
from PIL import Image
import albumentations as A
import glob

# Parameter
SOURCE_IMG_DIR = "C:\\MASTER\\Semester 2\\PROKEKT\\Oblique-instance-segmentation.v1i.yolov11\\train\\images"
SOURCE_LBL_DIR = "C:\\MASTER\\Semester 2\\PROKEKT\\Oblique-instance-segmentation.v1i.yolov11\\train\\labels"
OUTPUT_DIR = "yolo_dataset"
YOLO_IMG_SIZE = 640
AUG_PER_IMAGE = 6
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# YOLO-kompatible Augmentierungen
augmentations = A.Compose([
    A.Resize(YOLO_IMG_SIZE, YOLO_IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def prepare_dirs():
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

def collect_pairs():
    image_files = sorted(glob.glob(os.path.join(SOURCE_IMG_DIR, "*.jpg")))
    label_files = [os.path.join(SOURCE_LBL_DIR, os.path.basename(f).replace(".jpg", ".txt")) for f in image_files]
    return [(img, lbl) for img, lbl in zip(image_files, label_files) if os.path.exists(lbl)]

def split_dataset(pairs):
    random.shuffle(pairs)
    total = len(pairs)
    num_train = int(total * TRAIN_RATIO)
    num_val = int(total * VAL_RATIO)
    return pairs[:num_train], pairs[num_train:num_train+num_val], pairs[num_train+num_val:]

def process_and_augment(pairs, target_dir, augment=False):
    count = 0
    for img_path, lbl_path in pairs:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image.resize((YOLO_IMG_SIZE, YOLO_IMG_SIZE)))

        with open(lbl_path, "r") as f:
            lines = f.read().splitlines()

        bboxes = []
        class_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = map(float, parts)
            bboxes.append([x, y, w, h])
            class_labels.append(int(cls))

        # Original speichern
        base_name = os.path.basename(img_path)
        img_out = os.path.join(target_dir, base_name)
        Image.fromarray(image_np).save(img_out)
        with open(img_out.replace(".jpg", ".txt"), "w") as f:
            f.write("\n".join(lines))
        count += 1

        # Augmentieren
        if augment:
            for i in range(AUG_PER_IMAGE):
                aug = augmentations(image=image_np, bboxes=bboxes, class_labels=class_labels)
                aug_img = Image.fromarray(aug['image'])
                aug_boxes = aug['bboxes']
                aug_labels = aug['class_labels']

                aug_name = base_name.replace(".jpg", f"_aug{i}.jpg")
                aug_img_path = os.path.join(target_dir, aug_name)
                aug_txt_path = aug_img_path.replace(".jpg", ".txt")

                aug_img.save(aug_img_path)
                with open(aug_txt_path, "w") as f:
                    for cls, (x, y, w, h) in zip(aug_labels, aug_boxes):
                        f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return count

if __name__ == "__main__":
    prepare_dirs()
    pairs = collect_pairs()
    train_set, val_set, test_set = split_dataset(pairs)
    print("Train:", len(train_set), "Val:", len(val_set), "Test:", len(test_set))
    process_and_augment(train_set, TRAIN_DIR, augment=True)
    process_and_augment(val_set, VAL_DIR, augment=False)
    process_and_augment(test_set, TEST_DIR, augment=False)
    print("✅ Fertig: Alle Bilder verarbeitet und gespeichert.")
