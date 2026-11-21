
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Proje klasörleri
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

BERT_DIR = MODEL_DIR / "berturk"
QUANT_DIR = MODEL_DIR / "berturk_int8"

QUANT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# CPU kullanıyoruz: hocanın hedef ortamı CPU-only.
torch.set_num_threads(4)
device = torch.device("cpu")

print("Kullanılan cihaz:", device)


# 1. Eğitilmiş BERTurk modelini yükle
print("\nBERTurk modeli yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))

model = AutoModelForSequenceClassification.from_pretrained(
    str(BERT_DIR)
)

model = model.to(device)
model.eval()


# 2. Dynamic INT8 quantization
print("\nINT8 quantization uygulanıyor...")

quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

quantized_model.eval()


# 3. Modeli kaydet
torch.save(
    quantized_model.state_dict(),
    QUANT_DIR / "quantized_state_dict.pt"
)

tokenizer.save_pretrained(str(QUANT_DIR))

with open(
    QUANT_DIR / "quantization_info.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "method": "PyTorch Dynamic Quantization",
            "precision": "INT8 for Linear layers",
            "device": "CPU",
            "base_model": "models/berturk"
        },
        file,
        indent=2
    )

print("Quantized model kaydedildi.")


# 4. Ortak test verisini oku
test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")

texts = test_df["tweet"].fillna("").astype(str).tolist()
y_true = (test_df["label"] == "OFF").astype(int).to_numpy()

predictions = []
probabilities = []

print("\nQuantized BERTurk test ediliyor...")

# 5. Her mesajı CPU üzerinde sınıflandır
with torch.inference_mode():
    for index, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        outputs = quantized_model(**inputs)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probs).item())

        predictions.append(prediction)
        probabilities.append(float(probs[1].item()))

        if (index + 1) % 500 == 0:
            print(f"Test edilen: {index + 1}/{len(texts)}")


# 6. Sonuçları hesapla
macro_f1 = f1_score(
    y_true,
    predictions,
    average="macro"
)

print("\n=== QUANTIZED BERTURK TEST SONUÇLARI ===")
print("Macro-F1:", round(macro_f1, 4))

print(
    classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["NOT", "OFF"],
        zero_division=0
    )
)


# 7. Tahminleri kaydet
results = test_df[["id", "tweet", "label"]].copy()

results["prediction"] = [
    "OFF" if value == 1 else "NOT"
    for value in predictions
]

results["off_probability"] = probabilities

results.to_csv(
    RESULTS_DIR / "berturk_int8_predictions.csv",
    index=False
)

with open(
    RESULTS_DIR / "berturk_int8_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "macro_f1": float(macro_f1),
            "precision": "INT8",
            "test_samples": len(test_df)
        },
        file,
        indent=2
    )

print("\nKaydedildi: models/berturk_int8/")
print("Kaydedildi: results/berturk_int8_predictions.csv")
print("Kaydedildi: results/berturk_int8_metrics.json")
print("Quantized BERTurk tamamlandı!")
