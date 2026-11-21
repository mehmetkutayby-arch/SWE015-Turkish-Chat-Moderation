
import json
import time
import platform
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from gensim.models import Word2Vec
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

torch.set_num_threads(4)

WARMUP = 10
RUNS = 100

TEXTS = [
    "Bugün çok güzel bir gün.",
    "Bu oyunu gerçekten çok sevdim.",
    "Sen hiçbir şey bilmiyorsun.",
    "Takımımız bugün çok iyi oynadı.",
    "Bu yaptığın hiç hoş değil."
]

results = {}


# ==========================================
# HELPERS
# ==========================================

def size_mb(path):
    path = Path(path)

    if path.is_file():
        return path.stat().st_size / (1024 ** 2)

    return sum(
        f.stat().st_size
        for f in path.rglob("*")
        if f.is_file()
    ) / (1024 ** 2)


def benchmark(predict_function):
    for i in range(WARMUP):
        predict_function(TEXTS[i % len(TEXTS)])

    times = []

    for i in range(RUNS):
        text = TEXTS[i % len(TEXTS)]

        start = time.perf_counter()
        predict_function(text)
        elapsed = (time.perf_counter() - start) * 1000

        times.append(elapsed)

    return {
        "latency_ms": round(float(np.mean(times)), 4),
        "median_ms": round(float(np.median(times)), 4),
        "p95_ms": round(
            float(np.percentile(times, 95)), 4
        )
    }


def save_results():
    output = {
        "device": "CPU",
        "processor": platform.processor(),
        "cpu_threads": 4,
        "warmup_runs": WARMUP,
        "measured_runs": RUNS,
        "results": results
    }

    with open(
        RESULTS_DIR / "benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )


def show_result(name):
    r = results[name]

    print(
        f"{name}: "
        f"{r['latency_ms']:.4f} ms | "
        f"{r['size_mb']:.2f} MB"
    )


# ==========================================
# 1. TF-IDF + LOGISTIC REGRESSION
# ==========================================

print("\n[1/4] TF-IDF benchmark...")

tfidf_data = joblib.load(
    MODEL_DIR / "tfidf_model.joblib"
)

vectorizer = tfidf_data["vectorizer"]
tfidf_model = tfidf_data["model"]


def predict_tfidf(text):
    features = vectorizer.transform([text])
    return tfidf_model.predict(features)


results["TF-IDF"] = {
    **benchmark(predict_tfidf),
    "size_mb": round(
        size_mb(MODEL_DIR / "tfidf_model.joblib"), 2
    )
}

show_result("TF-IDF")
save_results()


# ==========================================
# 2. WORD2VEC + FFNN
# ==========================================

print("\n[2/4] Word2Vec benchmark...")


class FeedforwardNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


w2v = Word2Vec.load(
    str(MODEL_DIR / "word2vec.model")
)

checkpoint = torch.load(
    MODEL_DIR / "word2vec_ffnn.pt",
    map_location="cpu",
    weights_only=True
)

embedding_size = checkpoint["embedding_size"]

ffnn = FeedforwardNN(embedding_size)

ffnn.load_state_dict(
    checkpoint["model_state_dict"]
)

ffnn = ffnn.cpu().eval()


def predict_word2vec(text):
    tokens = text.lower().split()

    vectors = [
        w2v.wv[token]
        for token in tokens
        if token in w2v.wv
    ]

    if vectors:
        embedding = np.mean(
            vectors,
            axis=0
        )
    else:
        embedding = np.zeros(
            embedding_size,
            dtype=np.float32
        )

    x = torch.tensor(
        embedding,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.inference_mode():
        logits = ffnn(x)
        return torch.sigmoid(logits)


results["Word2Vec + FFNN"] = {
    **benchmark(predict_word2vec),
    "size_mb": round(
        size_mb(MODEL_DIR / "word2vec.model")
        + size_mb(MODEL_DIR / "word2vec_ffnn.pt"),
        2
    )
}

show_result("Word2Vec + FFNN")
save_results()


# ==========================================
# 3. BERTURK FP32
# ==========================================

print("\n[3/4] BERTurk FP32 benchmark...")

bert_dir = MODEL_DIR / "berturk"

tokenizer = AutoTokenizer.from_pretrained(
    str(bert_dir)
)

bert = AutoModelForSequenceClassification.from_pretrained(
    str(bert_dir)
)

bert = bert.cpu().eval()


def encode(text):
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )


def predict_bert(text):
    inputs = encode(text)

    with torch.inference_mode():
        return bert(**inputs).logits


results["BERTurk FP32"] = {
    **benchmark(predict_bert),
    "size_mb": round(
        size_mb(bert_dir / "model.safetensors"),
        2
    )
}

show_result("BERTurk FP32")
save_results()


# ==========================================
# 4. BERTURK INT8
# ==========================================

print("\n[4/4] BERTurk INT8 benchmark...")

int8_model = torch.ao.quantization.quantize_dynamic(
    bert,
    {nn.Linear},
    dtype=torch.qint8
)

int8_checkpoint = torch.load(
    MODEL_DIR
    / "berturk_int8"
    / "quantized_state_dict.pt",
    map_location="cpu",
    weights_only=True
)

int8_model.load_state_dict(int8_checkpoint)
int8_model.eval()


def predict_int8(text):
    inputs = encode(text)

    with torch.inference_mode():
        return int8_model(**inputs).logits


results["BERTurk INT8"] = {
    **benchmark(predict_int8),
    "size_mb": round(
        size_mb(
            MODEL_DIR
            / "berturk_int8"
            / "quantized_state_dict.pt"
        ),
        2
    )
}

show_result("BERTurk INT8")
save_results()


# ==========================================
# FINAL RESULTS
# ==========================================

print("\n" + "=" * 55)
print("CPU BENCHMARK RESULTS")
print("=" * 55)

for name in results:
    show_result(name)

print("=" * 55)
print("CPU:", platform.processor())
print("Runs per model:", RUNS)
print("CPU threads:", torch.get_num_threads())

print(
    "\nSaved:",
    RESULTS_DIR / "benchmark_results.json"
)

print("\nBenchmark completed successfully!")
