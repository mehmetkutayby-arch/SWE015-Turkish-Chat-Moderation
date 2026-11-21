
import time
import threading
import traceback
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import normalize_text


# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

torch.set_num_threads(4)

MODEL_NAMES = [
    "TF-IDF",
    "Word2Vec + FFNN",
    "BERTurk FP32",
    "BERTurk INT8"
]

LATENCY_BUDGET = 50.0

model_cache = {}
model_lock = threading.Lock()

LOG_COLUMNS = [
    "#",
    "Model",
    "Label",
    "Confidence %",
    "Latency (ms)",
    "Budget"
]


# ==========================================
# TEXT PREPROCESSING
# ==========================================

# Normalization is imported from src/preprocessing.py.


# ==========================================
# WORD2VEC NEURAL NETWORK
# ==========================================

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


# ==========================================
# LOAD MODELS
# ==========================================

def load_model(model_name):

    if model_name in model_cache:
        return model_cache[model_name]

    with model_lock:

        if model_name in model_cache:
            return model_cache[model_name]

        print(f"Loading model: {model_name}", flush=True)

        if model_name == "TF-IDF":

            checkpoint = joblib.load(
                MODEL_DIR / "tfidf_model.joblib"
            )

            loaded = (
                checkpoint["vectorizer"],
                checkpoint["model"]
            )

        elif model_name == "Word2Vec + FFNN":

            w2v = Word2Vec.load(
                str(MODEL_DIR / "word2vec.model")
            )

            checkpoint = torch.load(
                MODEL_DIR / "word2vec_ffnn.pt",
                map_location="cpu",
                weights_only=True
            )

            embedding_size = int(
                checkpoint["embedding_size"]
            )

            network = FeedforwardNN(embedding_size)

            network.load_state_dict(
                checkpoint["model_state_dict"]
            )

            network.eval()

            loaded = (
                w2v,
                network,
                embedding_size
            )

        elif model_name in ["BERTurk FP32", "BERTurk INT8"]:

            bert_dir = MODEL_DIR / "berturk"

            tokenizer = AutoTokenizer.from_pretrained(
                str(bert_dir)
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                str(bert_dir)
            )

            model = model.cpu().eval()

            if model_name == "BERTurk INT8":

                model = torch.ao.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},
                    dtype=torch.qint8
                )

                state = torch.load(
                    MODEL_DIR / "berturk_int8" / "quantized_state_dict.pt",
                    map_location="cpu",
                    weights_only=True
                )

                model.load_state_dict(state)
                model.eval()

            loaded = (tokenizer, model)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        model_cache[model_name] = loaded

        print(f"Model ready: {model_name}", flush=True)

        return loaded


# ==========================================
# PREDICTION
# ==========================================

def predict_message(text, model_name):

    loaded = load_model(model_name)

    start = time.perf_counter()

    if model_name == "TF-IDF":

        vectorizer, model = loaded

        features = vectorizer.transform(
            [normalize_text(text)]
        )

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        classes = list(model.classes_)
        index = classes.index(prediction)

        confidence = float(probabilities[index])

        label = (
            "OFF"
            if str(prediction).upper() in ("1", "OFF")
            else "NOT"
        )

    elif model_name == "Word2Vec + FFNN":

        w2v, network, embedding_size = loaded

        tokens = normalize_text(text).lower().split()

        vectors = [
            w2v.wv[token]
            for token in tokens
            if token in w2v.wv
        ]

        if vectors:
            embedding = np.mean(vectors, axis=0)
        else:
            embedding = np.zeros(
                embedding_size,
                dtype=np.float32
            )

        tensor = torch.tensor(
            embedding,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.inference_mode():
            logits = network(tensor)
            probability_off = float(
                torch.sigmoid(logits).item()
            )

        if probability_off >= 0.5:
            label = "OFF"
            confidence = probability_off
        else:
            label = "NOT"
            confidence = 1.0 - probability_off

    else:

        tokenizer, model = loaded

        inputs = tokenizer(
            normalize_text(text),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(
                logits,
                dim=-1
            )[0]

        index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[index].item())

        predicted_name = str(
            model.config.id2label.get(index, index)
        ).upper()

        if predicted_name in ("OFF", "OFFENSIVE"):
            label = "OFF"
        elif predicted_name in ("NOT", "CLEAN"):
            label = "NOT"
        else:
            label = "OFF" if index == 1 else "NOT"

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return label, confidence, latency_ms


# ==========================================
# MESSAGE MASKING
# ==========================================

def mask_message(text):

    return "".join(
        "█" if not char.isspace() else char
        for char in text
    )


# ==========================================
# CHAT HANDLER
# ==========================================

def send_message(message, history, logs, model_name):

    # Gradio 6 requires messages format.
    history = list(history) if history is not None else []

    # Logs are stored separately in gr.State.
    logs = list(logs) if logs is not None else []

    if not message or not message.strip():

        log_table = pd.DataFrame(
            logs,
            columns=LOG_COLUMNS
        )

        return (
            "",
            history,
            logs,
            log_table,
            "Please enter a message."
        )

    try:

        label, confidence, latency = predict_message(
            message.strip(),
            model_name
        )

        offensive = label == "OFF"

        displayed_message = (
            mask_message(message)
            if offensive
            else message
        )

        status = (
            "OFFENSIVE - Message masked"
            if offensive
            else "CLEAN - Message allowed"
        )

        budget_status = (
            "PASS"
            if latency <= LATENCY_BUDGET
            else "OVER"
        )

        response = (
            f"**{status}**\n\n"
            f"**Label:** {label}\n\n"
            f"**Confidence:** {confidence * 100:.2f}%\n\n"
            f"**Model:** {model_name}\n\n"
            f"**CPU latency:** {latency:.2f} ms\n\n"
            f"**50 ms budget:** {budget_status}"
        )

        # Correct Gradio 6 messages format.
        history.append({
            "role": "user",
            "content": displayed_message
        })

        history.append({
            "role": "assistant",
            "content": response
        })

        logs.append([
            len(logs) + 1,
            model_name,
            label,
            round(confidence * 100, 2),
            round(latency, 2),
            budget_status
        ])

        log_table = pd.DataFrame(
            logs,
            columns=LOG_COLUMNS
        )

        print(
            f"{model_name}: {label}, {latency:.2f} ms",
            flush=True
        )

        return (
            "",
            history,
            logs,
            log_table,
            status
        )

    except Exception as error:

        traceback.print_exc()

        log_table = pd.DataFrame(
            logs,
            columns=LOG_COLUMNS
        )

        return (
            message,
            history,
            logs,
            log_table,
            f"ERROR: {type(error).__name__}: {error}"
        )


# ==========================================
# CLEAR CHAT
# ==========================================

def clear_chat():

    empty_table = pd.DataFrame(
        columns=LOG_COLUMNS
    )

    return (
        "",
        [],
        [],
        empty_table,
        "Ready."
    )


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="Turkish AI Chat Moderation"
) as demo:

    gr.Markdown(
        "# Turkish AI Chat Moderation\n\n"
        "Real-time Turkish offensive language detection "
        "using four machine learning models."
    )

    log_state = gr.State([])

    with gr.Row():

        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="Live Chat",
                height=490
            )

            message_input = gr.Textbox(
                label="Your Message",
                placeholder="Type a Turkish message and press Enter...",
                lines=2
            )

            # ==========================================
            # EXAMPLE MESSAGES
            # ==========================================

            gr.Examples(
                examples=[
                    ["Bugün hava çok güzel."],
                    ["Sen tam bir aptalsın."],
                    ["Sen tam bir s4laksın."],
                    ["Orospu"]
                ],
                inputs=message_input,
                label="Example Messages (Örnek Mesajlar)"
            )

            with gr.Row():

                send_button = gr.Button(
                    "Send Message",
                    variant="primary"
                )

                clear_button = gr.Button(
                    "Clear Chat"
                )

        with gr.Column(scale=2):

            gr.Markdown("### Model Selection")

            model_selector = gr.Dropdown(
                choices=MODEL_NAMES,
                value="BERTurk INT8",
                label="Active Model",
                interactive=True
            )

            gr.Markdown(
                "**CPU latency target:** 50 ms\n\n"
                "**NOT:** Clean message\n\n"
                "**OFF:** Offensive message"
            )

            status_box = gr.Textbox(
                label="Last Result",
                value="Ready.",
                interactive=False
            )

            gr.Markdown(
                "### Per-Message CPU Latency Log"
            )

            latency_log = gr.Dataframe(
                headers=LOG_COLUMNS,
                datatype=[
                    "number",
                    "str",
                    "str",
                    "number",
                    "number",
                    "str"
                ],
                value=pd.DataFrame(columns=LOG_COLUMNS),
                interactive=False
            )

    event_inputs = [
        message_input,
        chatbot,
        log_state,
        model_selector
    ]

    event_outputs = [
        message_input,
        chatbot,
        log_state,
        latency_log,
        status_box
    ]

    send_button.click(
        fn=send_message,
        inputs=event_inputs,
        outputs=event_outputs
    )

    message_input.submit(
        fn=send_message,
        inputs=event_inputs,
        outputs=event_outputs
    )

    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=event_outputs
    )


# ==========================================
# START
# ==========================================

if __name__ == "__main__":

    print(
        "Starting Turkish AI Chat Moderation...",
        flush=True
    )

    print(
        "Gradio version:",
        gr.__version__,
        flush=True
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        inbrowser=True,
        show_error=True
    )
