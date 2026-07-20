"""
ocr_compare.py
Membandingkan dua pendekatan OCR untuk membaca teks plat nomor:
  1. Tesseract OCR (rule-based / klasik)
  2. EasyOCR (deep learning, CRNN-based)

Metrik: Accuracy (exact match, substring match) dan Character Error Rate (CER).

Syarat sistem:
- Tesseract engine harus terinstall di OS:
    Ubuntu/Debian : sudo apt install tesseract-ocr
    Windows       : https://github.com/UB-Mannheim/tesseract/wiki
    Mac           : brew install tesseract

Jalankan:
    python src/ocr_compare.py --limit 50           # uji cepat, subset 50 gambar acak
    python src/ocr_compare.py --prefix DataTrain    # hanya evaluasi subset crop bersih
    python src/ocr_compare.py                       # full dataset
"""

import re
from pathlib import Path

import cv2
import pandas as pd
import pytesseract
import easyocr
from tqdm import tqdm

# Set path Tesseract secara eksplisit (Windows tidak selalu membaca PATH dengan benar).
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "ocr"
IMAGES_DIR = OCR_DATA_DIR / "images"
LABELS_CSV = OCR_DATA_DIR / "labels.csv"
RESULTS_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "ocr_comparison_results.csv"


def clean_text(text: str) -> str:
    """Normalisasi teks plat: uppercase, hilangkan spasi & karakter selain alfanumerik."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def char_error_rate(pred: str, gt: str) -> float:
    """CER sederhana berbasis Levenshtein distance / panjang ground truth."""
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0

    d = [[0] * (len(pred) + 1) for _ in range(len(gt) + 1)]
    for i in range(len(gt) + 1):
        d[i][0] = i
    for j in range(len(pred) + 1):
        d[0][j] = j
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(gt)][len(pred)] / len(gt)


def preprocess_for_tesseract(img):
    """Preprocessing standar: grayscale -> resize, tanpa threshold agresif yang bisa merusak teks kecil."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return gray


def run_comparison(limit: int = None, prefix: str = None):
    df = pd.read_csv(LABELS_CSV)
    df.columns = [c.strip().lower() for c in df.columns]

    if prefix:
        df = df[df["filename"].str.startswith(prefix)].reset_index(drop=True)
        print(f"Filter aktif: hanya file berawalan '{prefix}' ({len(df)} sampel).")

    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"Mode uji cepat: hanya mengevaluasi {len(df)} sampel acak.")

    print("Memuat model EasyOCR (butuh waktu pertama kali, download bobot)...")
    reader = easyocr.Reader(["en"], gpu=False)

    tess_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Mengevaluasi OCR"):
        img_path = IMAGES_DIR / row["filename"]
        gt = clean_text(str(row["label"]))

        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # --- Tesseract ---
        processed = preprocess_for_tesseract(img)
        tess_raw = pytesseract.image_to_string(processed, config=tess_config)
        tess_pred = clean_text(tess_raw)

        # --- EasyOCR ---
        easy_result = reader.readtext(img, detail=0)
        easy_pred = clean_text("".join(easy_result))

        rows.append({
            "filename": row["filename"],
            "ground_truth": gt,
            "tesseract_pred": tess_pred,
            "tesseract_correct": int(tess_pred == gt),
            "tesseract_contains_gt": int(gt in tess_pred if tess_pred else False),
            "tesseract_cer": char_error_rate(tess_pred, gt),
            "easyocr_pred": easy_pred,
            "easyocr_correct": int(easy_pred == gt),
            "easyocr_contains_gt": int(gt in easy_pred if easy_pred else False),
            "easyocr_cer": char_error_rate(easy_pred, gt),
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_CSV, index=False)

    print("\n=== Ringkasan Perbandingan OCR ===")
    print(f"Jumlah sampel dievaluasi        : {len(result_df)}")
    print(f"Tesseract - Akurasi (exact)     : {result_df['tesseract_correct'].mean():.2%}")
    print(f"Tesseract - Akurasi (substring) : {result_df['tesseract_contains_gt'].mean():.2%}")
    print(f"Tesseract - Rata-rata CER       : {result_df['tesseract_cer'].mean():.4f}")
    print(f"EasyOCR   - Akurasi (exact)     : {result_df['easyocr_correct'].mean():.2%}")
    print(f"EasyOCR   - Akurasi (substring) : {result_df['easyocr_contains_gt'].mean():.2%}")
    print(f"EasyOCR   - Rata-rata CER       : {result_df['easyocr_cer'].mean():.4f}")
    print(f"\nHasil detail disimpan di: {RESULTS_CSV}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Jumlah sampel untuk uji cepat (mis. 100). Kosongkan untuk full dataset.")
    parser.add_argument("--prefix", type=str, default=None,
                         help="Hanya evaluasi file yang nama filenya berawalan string ini (mis. DataTrain)")
    args = parser.parse_args()
    run_comparison(limit=args.limit, prefix=args.prefix)