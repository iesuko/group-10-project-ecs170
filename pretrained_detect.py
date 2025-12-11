import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

detection_name = "papluca/xlm-roberta-base-language-detection"
detection_tokenizer = AutoTokenizer.from_pretrained(detection_name)
detection_model = AutoModelForSequenceClassification.from_pretrained(detection_name)

print("Loaded Language Detector")
print("Label mapping:", detection_model.config.id2label)

def predict_language(text):
    """
    Detect language and return (label, confidence)
    """
    inputs = detection_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = detection_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()
        label = detection_model.config.id2label[pred_id]
    return label, confidence

texts = [
    "Hello, how are you?",
    "Bonjour, comment ça va?",
    "Hola, ¿cómo estás?"
]

for t in texts:
    lang, conf = predict_language(t)
    print(f"Text: {t}")
    print(f"Detected Language: {lang} (Confidence: {conf:.2f})\n")
