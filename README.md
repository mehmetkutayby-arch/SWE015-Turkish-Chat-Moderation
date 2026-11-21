# Turkish Chat Moderation — SWE015

A Turkish offensive-language detection application built with Gradio.

## Models
- TF-IDF + Logistic Regression
- Word2Vec + Feedforward Neural Network
- Fine-tuned BERTurk (FP32)
- Quantized BERTurk (INT8)

## Requirements
- Python and pip
- Dependencies listed in requirements.txt

## Installation
1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   python -m pip install -r requirements.txt

4. Download the model files from the link below and place them in the `models/` directory, preserving the original folder structure.

   Model download: LINK_TO_BE_ADDED

## Run

   python app.py

Open the local Gradio URL printed in the terminal (configured port: 7861).

## Features
- Four selectable classification models
- Offensive (OFF) and non-offensive (NOT) labels
- Confidence scores and offensive-text masking
- Per-message CPU latency and 50 ms budget indicator
- Example Turkish messages and Clear Chat

## Dataset
OffensEval-TR: 31,277 training messages and 3,515 held-out test messages.

## Results
Benchmark and evaluation outputs are available in the `results/` folder.

## Source Code
Training, preprocessing, evaluation, quantization, and benchmarking scripts are in `src/`.
