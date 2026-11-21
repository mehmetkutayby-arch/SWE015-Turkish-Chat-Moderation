
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent
BERT_DIR = BASE_DIR / "models" / "berturk"
INT8_DIR = BASE_DIR / "models" / "berturk_int8"

torch.set_num_threads(4)

print("Normal BERTurk mimarisi yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))

model = AutoModelForSequenceClassification.from_pretrained(
    str(BERT_DIR)
).cpu()

model.eval()

print("INT8 mimarisi oluşturuluyor...")

model = torch.ao.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("Kaydedilmiş INT8 ağırlıkları yükleniyor...")

state_dict = torch.load(
    INT8_DIR / "quantized_state_dict.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("INT8 model başarıyla yeniden yüklendi!")

text = "Bugün çok güzel bir gün."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=128
)

with torch.inference_mode():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0]

prediction = int(torch.argmax(probabilities).item())

print("\nTest mesajı:", text)
print("Tahmin:", "OFF" if prediction == 1 else "NOT")
print("Güven:", round(float(probabilities[prediction]), 4))
print("\nINT8 yeniden yükleme testi tamamlandı!")
