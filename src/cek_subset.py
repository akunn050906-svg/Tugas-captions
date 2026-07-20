"""cek_subset.py - cek proporsi dan performa subset dataset OCR"""
import pandas as pd
from pathlib import Path

LABELS = Path("data/processed/ocr/labels.csv")
RESULTS = Path("data/processed/ocr_comparison_results.csv")

df = pd.read_csv(LABELS)
df.columns = [c.strip().lower() for c in df.columns]
is_clean = df["filename"].str.startswith("DataTrain")

print("=== Proporsi Dataset ===")
print("Total data                    :", len(df))
print("Subset DataTrain (crop bersih):", is_clean.sum())
print("Subset lainnya (foto lebar)   :", (~is_clean).sum())

if RESULTS.exists():
    res = pd.read_csv(RESULTS)
    res_clean = res[res["filename"].str.startswith("DataTrain")]
    res_other = res[~res["filename"].str.startswith("DataTrain")]

    print("\n=== Performa OCR: Subset DataTrain (crop bersih) ===")
    if len(res_clean) > 0:
        print(f"Jumlah sampel dievaluasi        : {len(res_clean)}")
        print(f"Tesseract - Akurasi (exact)     : {res_clean['tesseract_correct'].mean():.2%}")
        print(f"Tesseract - Akurasi (substring) : {res_clean['tesseract_contains_gt'].mean():.2%}")
        print(f"Tesseract - Rata-rata CER       : {res_clean['tesseract_cer'].mean():.4f}")
        print(f"EasyOCR   - Akurasi (exact)     : {res_clean['easyocr_correct'].mean():.2%}")
        print(f"EasyOCR   - Akurasi (substring) : {res_clean['easyocr_contains_gt'].mean():.2%}")
        print(f"EasyOCR   - Rata-rata CER       : {res_clean['easyocr_cer'].mean():.4f}")
    else:
        print("(tidak ada sampel DataTrain di hasil evaluasi ini)")

    print("\n=== Performa OCR: Subset Lainnya (foto lebar) ===")
    if len(res_other) > 0:
        print(f"Jumlah sampel dievaluasi        : {len(res_other)}")
        print(f"Tesseract - Akurasi (substring) : {res_other['tesseract_contains_gt'].mean():.2%}")
        print(f"Tesseract - Rata-rata CER       : {res_other['tesseract_cer'].mean():.4f}")
        print(f"EasyOCR   - Akurasi (substring) : {res_other['easyocr_contains_gt'].mean():.2%}")
        print(f"EasyOCR   - Rata-rata CER       : {res_other['easyocr_cer'].mean():.4f}")