"""
train_detection.py
Training model deteksi plat nomor menggunakan YOLOv8.

Untuk memenuhi ketentuan capstone (perbandingan minimal 2 pendekatan), jalankan
script ini dua kali dengan model berbeda:

    python src/train_detection.py --model yolov8n.pt --name plate_yolov8n
    python src/train_detection.py --model yolov8s.pt --name plate_yolov8s

Hasil kedua run bisa dibandingkan lewat runs/detect/<name>/results.csv
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "detection"
DATA_YAML = PROC_DIR / "data.yaml"


def train(model_name: str, epochs: int, imgsz: int, run_name: str):
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} tidak ditemukan. Jalankan src/preprocess.py terlebih dahulu."
        )

    model = YOLO(model_name)  # otomatis download pretrained weights (COCO)

    print(f"Training {model_name} | epochs={epochs} | imgsz={imgsz}")
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name=run_name,
        patience=15,
        project="runs/detect",
    )

    # Simpan bobot terbaik ke folder models/ agar mudah dipakai pipeline.py
    best_weights = Path("runs/detect") / run_name / "weights" / "best.pt"
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    if best_weights.exists():
        target = models_dir / f"{run_name}_best.pt"
        target.write_bytes(best_weights.read_bytes())
        print(f"Bobot terbaik disalin ke {target}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt",
                         help="yolov8n.pt (ringan) atau yolov8s.pt (lebih besar) untuk perbandingan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default=None, help="Nama run, default = nama model")
    args = parser.parse_args()

    run_name = args.name or Path(args.model).stem
    train(args.model, args.epochs, args.imgsz, run_name)
