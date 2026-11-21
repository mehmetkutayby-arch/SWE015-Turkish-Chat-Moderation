
from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from preprocessing import normalize_text


# Proje klasörleri
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# 1. Eğitim verisini oku
train_path = next(DATA_DIR.rglob("offenseval-tr-training-v1.tsv"))
train_df = pd.read_csv(train_path, sep="\t")

# 2. Daha önce hazırladığımız ortak test verisini oku
test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")

print("Eğitim mesajları:", len(train_df))
print("Test mesajları:", len(test_df))


# 3. Metinleri temizle
X_train = train_df["tweet"].fillna("").map(normalize_text)
X_test = test_df["tweet"].fillna("").map(normalize_text)

y_train = train_df["subtask_a"]
y_test = test_df["label"]


# 4. TF-IDF: Kelimeleri sayısal özelliklere dönüştür
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 5. Klasik makine öğrenmesi modelini eğit
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

print("\nModel eğitiliyor...")
model.fit(X_train_tfidf, y_train)


# 6. Ortak test verisinde değerlendir
predictions = model.predict(X_test_tfidf)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

print("\n=== TF-IDF TEST SONUÇLARI ===")
print("Macro-F1:", round(macro_f1, 4))
print(classification_report(y_test, predictions))


# 7. Eğitilen modeli kaydet
joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model
    },
    MODEL_DIR / "tfidf_model.joblib"
)

# 8. Tahminleri kaydet
results = test_df[["id", "tweet", "label"]].copy()
results["prediction"] = predictions
results.to_csv(
    RESULTS_DIR / "tfidf_predictions.csv",
    index=False
)

print("\nModel kaydedildi: models/tfidf_model.joblib")
print("Tahminler kaydedildi: results/tfidf_predictions.csv")
print("TF-IDF tamamlandı!")
