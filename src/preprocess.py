"""
preprocess.py
1. Mengonversi anotasi COCO (plate_detection_dataset) -> format YOLO (.txt per gambar).
2. Membagi data menjadi train/val/test (80/10/10).
3. Menyalin gambar plate_text_dataset + label.csv ke folder processed untuk tahap OCR.

Struktur output:
data/processed/
├── detection/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
└── ocr/
    ├── images/
    └── labels.csv
"""

import json
import shutil
import random
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DET_SRC_IMAGES = RAW_DIR / "plate_detection_dataset" / "plate_detection_dataset" / "images"
DET_SRC_ANNOT = RAW_DIR / "plate_detection_dataset" / "plate_detection_dataset" / "annotations" / "annotations.json"

TEXT_SRC_IMAGES = RAW_DIR / "plate_text_dataset" / "plate_text_dataset" / "dataset"
TEXT_SRC_LABEL = RAW_DIR / "plate_text_dataset" / "plate_text_dataset" / "label.csv"

SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO bbox [x_min, y_min, width, height] -> YOLO [x_center, y_center, w, h] (dinormalisasi)."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def prepare_detection_data():
    print("=== Memproses data deteksi (COCO -> YOLO) ===")
    with open(DET_SRC_ANNOT, "r") as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annots_by_image = {}
    for ann in coco["annotations"]:
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images_by_id.keys())
    random.seed(SEED)
    random.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * SPLIT_RATIO["train"])
    n_val = int(n * SPLIT_RATIO["val"])
    splits = {
        "train": image_ids[:n_train],
        "val": image_ids[n_train:n_train + n_val],
        "test": image_ids[n_train + n_val:],
    }

    det_out = PROC_DIR / "detection"
    for split in splits:
        (det_out / "images" / split).mkdir(parents=True, exist_ok=True)
        (det_out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split, ids in splits.items():
        for img_id in ids:
            img_info = images_by_id[img_id]
            file_name = img_info["file_name"]
            # width/height di dataset ini tersimpan sebagai string, jadi perlu di-cast ke int
            img_w, img_h = int(img_info["width"]), int(img_info["height"])

            src_img = DET_SRC_IMAGES / file_name
            if not src_img.exists():
                continue
            shutil.copy(src_img, det_out / "images" / split / file_name)

            label_path = det_out / "labels" / split / (Path(file_name).stem + ".txt")
            lines = []
            for ann in annots_by_image.get(img_id, []):
                xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                # class 0 = "plate" (hanya 1 kelas)
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            label_path.write_text("\n".join(lines))

        print(f"  {split}: {len(ids)} gambar")

    # data.yaml untuk YOLOv8
    yaml_content = f"""path: {det_out.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['plate']
"""
    (det_out / "data.yaml").write_text(yaml_content)
    print(f"Selesai. data.yaml disimpan di {det_out / 'data.yaml'}")


def prepare_ocr_data():
    print("=== Memproses data OCR (plate_text_dataset) ===")
    ocr_out = PROC_DIR / "ocr"
    (ocr_out / "images").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TEXT_SRC_LABEL)
    df.columns = [c.strip().lower() for c in df.columns]  # normalisasi nama kolom

    copied = 0
    for _, row in df.iterrows():
        src = TEXT_SRC_IMAGES / row["filename"]
        if src.exists():
            shutil.copy(src, ocr_out / "images" / row["filename"])
            copied += 1

    df.to_csv(ocr_out / "labels.csv", index=False)
    print(f"  {copied} gambar plat disalin, label disimpan di {ocr_out / 'labels.csv'}")


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    prepare_detection_data()
    prepare_ocr_data()
    print("\nPreprocessing selesai.")
