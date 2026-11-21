
from pathlib import Path
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from lexicon_matcher import predict_message


# Proje klasörleri
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# 1. Ortak test verisini oku
test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")

print("Test mesajları:", len(test_df))


# 2. Her mesaj için tahmin yap
predictions = []
response_times = []

print("\nLexicon Matcher test ediliyor...")

for message in test_df["tweet"].fillna(""):
    start = time.perf_counter()

    result = predict_message(message)

    elapsed_ms = (time.perf_counter() - start) * 1000

    predictions.append(result["label"])
    response_times.append(elapsed_ms)


# 3. Başarı metrikleri
y_true = test_df["label"]

macro_f1 = f1_score(
    y_true,
    predictions,
    average="macro"
)

print("\n=== LEXICON MATCHER TEST SONUÇLARI ===")
print("Macro-F1:", round(macro_f1, 4))

print(
    classification_report(
        y_true,
        predictions,
        labels=["NOT", "OFF"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_true,
        predictions,
        labels=["NOT", "OFF"]
    )
)


# 4. Ortalama tahmin süresi
average_latency = sum(response_times) / len(response_times)

print("\nOrtalama süre (ms):", round(average_latency, 4))


# 5. Tahminleri kaydet
results = test_df[["id", "tweet", "label"]].copy()

results["prediction"] = predictions
results["latency_ms"] = response_times

results.to_csv(
    RESULTS_DIR / "lexicon_predictions.csv",
    index=False
)

print("\nKaydedildi: results/lexicon_predictions.csv")
print("Lexicon değerlendirmesi tamamlandı!")
