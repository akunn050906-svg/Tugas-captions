"""
app.py
Aplikasi web Flask untuk demo sistem deteksi + OCR plat nomor kendaraan.
Ikuti pola deployment project sebelumnya (CurrencyVision, CloudVision) -> siap deploy ke Railway.
"""

import os
import sys
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Agar bisa import src/pipeline.py dari root project
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import read_plate  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = APP_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # maks 8MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Format file tidak didukung (gunakan jpg/png)"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        result = read_plate(str(save_path))
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {e}"}), 500

    result["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
