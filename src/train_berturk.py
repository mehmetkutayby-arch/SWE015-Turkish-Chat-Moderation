
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


# 1. Ayarlar
SEED = 42
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MAX_LENGTH = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_DIR = MODEL_DIR / "berturk"
CHECKPOINT_DIR = MODEL_DIR / "berturk_checkpoints"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print("CUDA kullanılabilir:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# 2. Veri setini yükle
train_path = next(DATA_DIR.rglob("offenseval-tr-training-v1.tsv"))

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")

train_df["tweet"] = train_df["tweet"].fillna("").astype(str)
test_df["tweet"] = test_df["tweet"].fillna("").astype(str)

train_df["target"] = (train_df["subtask_a"] == "OFF").astype(int)
test_df["target"] = (test_df["label"] == "OFF").astype(int)

fit_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["target"],
)

print("Eğitim:", len(fit_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# 3. BERTurk tokenizer
print("\nBERTurk tokenizer yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class ChatDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["tweet"].tolist()
        self.labels = dataframe["target"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return item


train_dataset = ChatDataset(fit_df)
val_dataset = ChatDataset(val_df)
test_dataset = ChatDataset(test_df)


# 4. BERTurk sınıflandırma modeli
print("\nBERTurk modeli yükleniyor...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "NOT", 1: "OFF"},
    label2id={"NOT": 0, "OFF": 1},
)


# 5. Macro-F1 hesaplama
def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=1)

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        )
    }


# 6. Eğitim ayarları
training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=167,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none",
    seed=SEED,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=2)
    ],
)


# 7. Modeli eğit
print("\nBERTurk fine-tuning başlıyor...")

trainer.train()


# 8. En iyi modeli kaydet
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\nModel kaydedildi:", OUTPUT_DIR)


# 9. Ortak test verisinde değerlendir
print("\nTest verisi değerlendiriliyor...")

test_output = trainer.predict(test_dataset)

logits = test_output.predictions
predictions = np.argmax(logits, axis=1)

probabilities = torch.softmax(
    torch.tensor(logits),
    dim=1,
).numpy()[:, 1]

macro_f1 = f1_score(
    test_df["target"],
    predictions,
    average="macro",
)

print("\n=== BERTURK TEST SONUÇLARI ===")
print("Macro-F1:", round(macro_f1, 4))

print(
    classification_report(
        test_df["target"],
        predictions,
        labels=[0, 1],
        target_names=["NOT", "OFF"],
        zero_division=0,
    )
)


# 10. Tahminleri kaydet
results = test_df[["id", "tweet", "label"]].copy()

results["prediction"] = [
    "OFF" if value == 1 else "NOT"
    for value in predictions
]

results["off_probability"] = probabilities

results.to_csv(
    RESULTS_DIR / "berturk_predictions.csv",
    index=False,
)

with open(
    RESULTS_DIR / "berturk_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "macro_f1": float(macro_f1),
            "model": MODEL_NAME,
            "test_samples": len(test_df),
        },
        file,
        indent=2,
    )

print("\nKaydedildi: results/berturk_predictions.csv")
print("Kaydedildi: results/berturk_metrics.json")
print("BERTurk fine-tuning tamamlandı!")
