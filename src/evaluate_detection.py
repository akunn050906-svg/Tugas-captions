"""
evaluate_detection.py
Mengevaluasi model deteksi plat pada data test: mAP@0.5, mAP@0.5:0.95, precision, recall.

Contoh:
    python src/evaluate_detection.py --weights models/plate_yolov8n_best.pt
    python src/evaluate_detection.py --weights models/plate_yolov8s_best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "processed" / "detection" / "data.yaml"


def evaluate(weights_path: str):
    model = YOLO(weights_path)
    metrics = model.val(data=str(DATA_YAML), split="test")

    print("\n=== Hasil Evaluasi ===")
    print(f"Model        : {weights_path}")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")

    return {
        "model": weights_path,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path ke file .pt hasil training")
    args = parser.parse_args()
    evaluate(args.weights)
