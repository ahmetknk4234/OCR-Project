from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# 📌 Flask API'yi Başlat
app = Flask(__name__)
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'fisler.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Receipt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    store = db.Column(db.String(100))
    date = db.Column(db.String(20))
    amount = db.Column(db.String(20))
    text = db.Column(db.Text)

# 📌 PaddleOCR Modelini Yükle
ocr = PaddleOCR(use_angle_cls=True, lang="tr")


# ✅ Eğrilik Düzeltme
def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 255))
    if coords.shape[0] == 0:
        return image  # Tam beyazsa orijinali döndür
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed


# 📌 OCR ile Metin Okuma
def ocr_read_text(image):
    result = ocr.ocr(image, cls=True)
    extracted_text = ""

    for line in result:
        if line is None:
            continue
        for word_info in line:
            if word_info is not None:
                text = word_info[1][0]
                extracted_text += f"{text}\n"

    return extracted_text


# 📌 Tarih Ayıklama
def extract_date(text):
    tarih_pattern = r"(?:TARIH|DATE)?[: ]?(\b\d{2}[./-]\d{2}[./-]\d{4}\b)"
    saatli_tarih_pattern = r"(\b\d{2}[./-]\d{2}[./-]\d{4})\d{2}[:]\d{2}"

    match = re.search(saatli_tarih_pattern, text)
    if match:
        raw_date = match.group(1)
    else:
        match = re.search(tarih_pattern, text)
        if match:
            raw_date = match.group(1)
        else:
            return "Bulunamadı"

    # Olası OCR tarih formatları
    olasi_formatlar = ["%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"]

    for fmt in olasi_formatlar:
        try:
            parsed = datetime.strptime(raw_date, fmt)
            return parsed.strftime("%Y-%m-%d")  # Standartlaştırılmış format
        except ValueError:
            continue

    return "Bulunamadı"


# 📌 Tutar Ayıklama
def extract_amount(text):
    lines = text.split("\n")

    # Sadece yıldızlı ve muhtemelen fiyat içeren satırları al
    star_lines = [
        line for line in lines
        if "*" in line and line.count("*") < 3  # kart numarası gibi satırları atla
    ]

    amounts = []
    for line in star_lines:
        # Sayıları bul
        matches = re.findall(r"([\d]+[\.,]?[\d]*)", line)
        for match in matches:
            try:
                amount = float(match.replace(",", "."))
                if 1 <= amount <= 10000:
                    amounts.append(amount)
            except ValueError:
                continue

    return max(amounts) if amounts else "Bulunamadı"




# 📌 Mağaza Listesi
store_list = [
    "Migros", "BIM", "ŞOK", "A.101", "Carrefour", "Metro", "Hakmar",
    "Teknosa", "File Market", "Torku", "Happy Center", "Özkuruyemiş",
    "Kipa", "Ucuz Market", "Tarım Kredi Kooperatif", "Bizim Market"
]


# 📌 TF-IDF ile Dinamik Mağaza Tahmini
def predict_store_name_tfidf(ocr_text):
    corpus = store_list + [ocr_text]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    best_match_index = np.argmax(cosine_similarities)
    best_score = cosine_similarities[best_match_index]
    best_store = store_list[best_match_index]

    return best_store if best_score > 0.3 else "Bilinmeyen Mağaza"


# 📌 Mağaza İsmi Ayıklama
def extract_store_name(text):
    for store in store_list:
        if store.lower() in text.lower():
            return store
    return predict_store_name_tfidf(text)


# 📌 Resim Yükleme ve OCR Çalıştırma
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ✅ Sadece eğrilik düzeltme uygulanıyor
    corrected = deskew_image(image)

    text = ocr_read_text(corrected)
    tarih = extract_date(text)
    tutar = extract_amount(text)
    magaza = extract_store_name(text)

    return jsonify({
        "success": True,
        "extracted_text": text,
        "store": magaza,
        "date": tarih,
        "amount": tutar
    })

@app.route('/save', methods=['POST'])
def save_receipt():
    data = request.get_json()

    new_receipt = Receipt(
        store=data.get("store"),
        date=data.get("date"),
        amount=data.get("amount"),
    )
    db.session.add(new_receipt)
    db.session.commit()

    return jsonify({"message": "Fiş başarıyla kaydedildi ✅"})


@app.route('/receipts', methods=['GET'])
def get_receipts():
    receipts = Receipt.query.all()
    result = [{
        "id": r.id,
        "store": r.store,
        "date": r.date,
        "amount": r.amount,
        "text": r.text
    } for r in receipts]
    return jsonify(result)

# 📌 Fiş Silme (DELETE)
@app.route('/receipt/<int:receipt_id>', methods=['DELETE'])
def delete_receipt(receipt_id):
    receipt = Receipt.query.get(receipt_id)
    if receipt is None:
        return jsonify({"error": "Fiş bulunamadı"}), 404

    db.session.delete(receipt)
    db.session.commit()
    return jsonify({"message": f"{receipt_id} ID'li fiş silindi."})


# 📌 Fiş Güncelleme (PUT)
@app.route('/receipt/<int:receipt_id>', methods=['PUT'])
def update_receipt(receipt_id):
    receipt = Receipt.query.get(receipt_id)
    if receipt is None:
        return jsonify({"error": "Fiş bulunamadı"}), 404

    data = request.get_json()
    receipt.store = data.get("store", receipt.store)
    receipt.date = data.get("date", receipt.date)
    receipt.amount = data.get("amount", receipt.amount)

    db.session.commit()
    return jsonify({"message": f"{receipt_id} ID'li fiş güncellendi."})


# 📌 Ana Sayfa Testi
@app.route('/')
def home():
    return jsonify({"message": "Flask OCR API Çalışıyor!"})

# 📌 Çalıştır
if __name__ == '__main__':
    with app.app_context():
        print("📦 Veritabanı oluşturuluyor...")
        db.create_all()
    app.run(host='0.0.0.0', port=5000)