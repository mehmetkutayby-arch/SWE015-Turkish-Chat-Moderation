
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocessing import normalize_text


# --------------------------------------------------
# 1. KLASÖRLER
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

BERT_DIR = MODEL_DIR / "berturk"
QUANT_DIR = MODEL_DIR / "berturk_int8"

RESULTS_DIR.mkdir(exist_ok=True)

torch.set_num_threads(4)
device = torch.device("cpu")

print("Kullanılan cihaz:", device)


# --------------------------------------------------
# 2. TEST VERİSİ
# --------------------------------------------------

test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")

texts_raw = test_df["tweet"].fillna("").astype(str).tolist()
texts_normalized = [
    normalize_text(text)
    for text in texts_raw
]

y_true = (test_df["label"] == "OFF").astype(int).to_numpy()

print("Test mesajı sayısı:", len(test_df))


# --------------------------------------------------
# 3. TOKENIZER VE FP32 MODELİ
# --------------------------------------------------

print("\nBERTurk FP32 yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))

fp32_model = AutoModelForSequenceClassification.from_pretrained(
    str(BERT_DIR)
)

fp32_model = fp32_model.to(device)
fp32_model.eval()


# --------------------------------------------------
# 4. INT8 MODELİ
# --------------------------------------------------

print("\nBERTurk INT8 yükleniyor...")

int8_model = torch.ao.quantization.quantize_dynamic(
    fp32_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

int8_state = torch.load(
    QUANT_DIR / "quantized_state_dict.pt",
    map_location="cpu",
    weights_only=True
)

int8_model.load_state_dict(int8_state)
int8_model.eval()


# --------------------------------------------------
# 5. DEĞERLENDİRME FONKSİYONU
# --------------------------------------------------

def evaluate(model, texts, name):

    predictions = []

    print(f"\nBaşlıyor: {name}")

    with torch.inference_mode():

        for index, text in enumerate(texts):

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )

            outputs = model(**inputs)

            prediction = int(
                torch.argmax(outputs.logits, dim=1).item()
            )

            predictions.append(prediction)

            if (index + 1) % 500 == 0:
                print(
                    f"{name}: {index + 1}/{len(texts)}"
                )

    score = f1_score(
        y_true,
        predictions,
        average="macro"
    )

    print(f"{name} Macro-F1: {score:.4f}")

    return score, predictions


# --------------------------------------------------
# 6. DÖRT TEST
# --------------------------------------------------

fp32_raw_score, fp32_raw_preds = evaluate(
    fp32_model,
    texts_raw,
    "FP32 RAW"
)

fp32_norm_score, fp32_norm_preds = evaluate(
    fp32_model,
    texts_normalized,
    "FP32 NORMALIZED"
)

int8_raw_score, int8_raw_preds = evaluate(
    int8_model,
    texts_raw,
    "INT8 RAW"
)

int8_norm_score, int8_norm_preds = evaluate(
    int8_model,
    texts_normalized,
    "INT8 NORMALIZED"
)


# --------------------------------------------------
# 7. SONUÇLARI GÖSTER
# --------------------------------------------------

print("\n========== KARŞILAŞTIRMA ==========")

print(f"FP32 RAW:        {fp32_raw_score:.4f}")
print(f"FP32 NORMALIZED: {fp32_norm_score:.4f}")

print(f"INT8 RAW:        {int8_raw_score:.4f}")
print(f"INT8 NORMALIZED: {int8_norm_score:.4f}")

print("===================================")


# --------------------------------------------------
# 8. SONUÇLARI AYRI DOSYALARA KAYDET
# --------------------------------------------------

metrics = {
    "test_samples": len(test_df),
    "fp32_raw_macro_f1": float(fp32_raw_score),
    "fp32_normalized_macro_f1": float(fp32_norm_score),
    "int8_raw_macro_f1": float(int8_raw_score),
    "int8_normalized_macro_f1": float(int8_norm_score)
}

with open(
    RESULTS_DIR / "normalization_comparison.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

results = test_df[["id", "tweet", "label"]].copy()

results["fp32_raw"] = fp32_raw_preds
results["fp32_normalized"] = fp32_norm_preds
results["int8_raw"] = int8_raw_preds
results["int8_normalized"] = int8_norm_preds

results.to_csv(
    RESULTS_DIR / "normalization_comparison.csv",
    index=False
)

print("\nKarşılaştırma tamamlandı!")
print("Kaydedildi: results/normalization_comparison.json")
print("Kaydedildi: results/normalization_comparison.csv")
