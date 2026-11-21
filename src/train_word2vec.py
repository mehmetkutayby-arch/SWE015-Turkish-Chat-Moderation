
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import normalize_text


# 1. Ayarlar
SEED = 42
EMBEDDING_SIZE = 100
EPOCHS = 12
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)


# 2. Verileri yükle
train_path = next(DATA_DIR.rglob("offenseval-tr-training-v1.tsv"))
train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(DATA_DIR / "test_labeled.csv")

print("Eğitim verisi:", len(train_df))
print("Test verisi:", len(test_df))


# 3. Metinleri temizle ve kelimelere ayır
train_texts = [
    normalize_text(text).split()
    for text in train_df["tweet"].fillna("")
]

test_texts = [
    normalize_text(text).split()
    for text in test_df["tweet"].fillna("")
]

y_train = (train_df["subtask_a"] == "OFF").astype(int).to_numpy()
y_test = (test_df["label"] == "OFF").astype(int).to_numpy()


# 4. Word2Vec modelini sadece eğitim mesajlarıyla eğit
print("\nWord2Vec eğitiliyor...")

word2vec = Word2Vec(
    sentences=train_texts,
    vector_size=EMBEDDING_SIZE,
    window=5,
    min_count=2,
    workers=1,
    sg=1,
    epochs=10,
    seed=SEED
)

word2vec.save(str(MODEL_DIR / "word2vec.model"))


# 5. Mesajı kelime vektörlerinin ortalamasına dönüştür
def text_to_vector(tokens):
    vectors = [
        word2vec.wv[word]
        for word in tokens
        if word in word2vec.wv
    ]

    if not vectors:
        return np.zeros(EMBEDDING_SIZE, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


print("Mesaj vektörleri hazırlanıyor...")

X_train = np.array(
    [text_to_vector(tokens) for tokens in train_texts],
    dtype=np.float32
)

X_test = np.array(
    [text_to_vector(tokens) for tokens in test_texts],
    dtype=np.float32
)


# 6. Eğitim ve doğrulama verisini ayır
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train
)


def make_loader(X, y, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


train_loader = make_loader(X_fit, y_fit, shuffle=True)
val_loader = make_loader(X_val, y_val)


# 7. Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


model = FeedforwardNN().to(device)

positive_count = int(y_fit.sum())
negative_count = len(y_fit) - positive_count

loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [negative_count / positive_count],
        device=device
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 8. Sinir ağını eğit
best_val_f1 = -1.0
best_state = None

print("\nNeural Network eğitiliyor...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)

            outputs = model(features)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= 0.5).int()

            val_predictions.extend(predictions.cpu().numpy())
            val_labels.extend(labels.numpy())

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="macro"
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Validation Macro-F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }


# 9. En iyi modeli yükle
model.load_state_dict(best_state)
model.eval()

torch.save(
    {
        "model_state_dict": best_state,
        "embedding_size": EMBEDDING_SIZE,
        "model_type": "Word2Vec + FFNN"
    },
    MODEL_DIR / "word2vec_ffnn.pt"
)


# 10. Ortak test setinde değerlendir
test_tensor = torch.tensor(X_test, dtype=torch.float32)

all_predictions = []
all_probabilities = []

print("\nTest verisi değerlendiriliyor...")

with torch.no_grad():
    for batch in torch.split(test_tensor, BATCH_SIZE):
        batch = batch.to(device)

        outputs = model(batch)
        probabilities = torch.sigmoid(outputs)

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        all_predictions.extend(
            (probabilities >= 0.5).int().cpu().numpy().tolist()
        )

macro_f1 = f1_score(
    y_test,
    all_predictions,
    average="macro"
)

print("\n=== WORD2VEC + FFNN TEST SONUÇLARI ===")
print("Macro-F1:", round(macro_f1, 4))

print(
    classification_report(
        y_test,
        all_predictions,
        target_names=["NOT", "OFF"],
        zero_division=0
    )
)


# 11. Tahminleri kaydet
results = test_df[["id", "tweet", "label"]].copy()

results["prediction"] = [
    "OFF" if prediction == 1 else "NOT"
    for prediction in all_predictions
]

results["off_probability"] = all_probabilities

results.to_csv(
    RESULTS_DIR / "word2vec_predictions.csv",
    index=False
)

print("\nKaydedildi: models/word2vec.model")
print("Kaydedildi: models/word2vec_ffnn.pt")
print("Kaydedildi: results/word2vec_predictions.csv")
print("Word2Vec + FFNN tamamlandı!")
