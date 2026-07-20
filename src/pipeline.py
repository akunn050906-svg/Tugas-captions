"""
pipeline.py
Pipeline end-to-end: deteksi lokasi plat (YOLOv8) -> crop -> baca teks (EasyOCR).

Dipakai sebagai:
1. Script CLI:  python src/pipeline.py --image contoh.jpg
2. Modul yang di-import oleh app/app.py (aplikasi Flask)
"""

import argparse
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "plate_yolov8n_best.pt"  # ganti sesuai model terbaik hasil evaluasi

_detector = None
_reader = None


def load_models(weights_path: str = None):
    """Lazy-load model agar tidak reload berulang kali (dipakai juga oleh Flask app)."""
    global _detector, _reader
    if _detector is None:
        path = weights_path or str(DEFAULT_WEIGHTS)
        _detector = YOLO(path)
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _detector, _reader


def read_plate(image_path: str, weights_path: str = None, conf: float = 0.4):
    """
    Menjalankan pipeline lengkap pada satu gambar.

    Return: dict berisi bounding box plat yang terdeteksi beserta teks hasil OCR.
    """
    detector, reader = load_models(weights_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan/tidak bisa dibaca: {image_path}")

    results = detector.predict(img, conf=conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Upscale crop karena hasil deteksi biasanya kecil (puluhan piksel),
        # terlalu kecil untuk dibaca OCR dengan akurat.
        crop_h, crop_w = crop.shape[:2]
        scale = max(1, 200 // max(crop_h, 1))  # target tinggi crop ~200px
        if scale > 1:
            crop = cv2.resize(crop, (crop_w * scale, crop_h * scale), interpolation=cv2.INTER_CUBIC)

        text_result = reader.readtext(crop, detail=0)
        plate_text = " ".join(text_result).upper().strip()

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "detection_confidence": round(confidence, 3),
            "plate_text": plate_text,
        })

    return {
        "image_path": image_path,
        "num_plates_detected": len(detections),
        "detections": detections,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path ke gambar kendaraan")
    parser.add_argument("--weights", default=None, help="Path ke bobot YOLOv8 (.pt)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold deteksi")
    args = parser.parse_args()

    output = read_plate(args.image, args.weights, args.conf)

    print(f"\nGambar: {output['image_path']}")
    print(f"Jumlah plat terdeteksi: {output['num_plates_detected']}")
    for i, det in enumerate(output["detections"], start=1):
        print(f"  [{i}] bbox={det['bbox']} conf={det['detection_confidence']} "
              f"-> teks: '{det['plate_text']}'")
