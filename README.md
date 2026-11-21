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

4. Download the ZIP file from the link below. Extract its contents directly into the `models/` folder in the project root. After extraction, the paths should include `models/berturk/config.json`, `models/berturk_int8/quantized_state_dict.pt`, and `models/tfidf_model.joblib`.

   Model download: https://drive.google.com/file/d/19AdsyydSwQSPUT5_RIrj2bmXxVjZvmXS/view?usp=sharing

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


