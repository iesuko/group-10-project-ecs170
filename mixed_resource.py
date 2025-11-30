from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "papluca/xlm-roberta-base-language-detection"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("Label mapping:", model.config.id2label)

def predict_language(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_id].item()
        label = model.config.id2label[predicted_id]
    return label, confidence

high_resource_texts = [
    "Hello, how are you?",  #English      
    "Guten Morgen, wie geht es dir?", #German
    "Bonjour, je suis étudiant.",  #French
    "Hola, ¿cómo estás?"  #Spanish         
]

print("\n--- High-Resource Languages Test ---\n")
for text in high_resource_texts:
    label, conf = predict_language(text)
    print(f"Text: {text}")
    print(f"Predicted language: {label} (Confidence: {conf:.2f})\n")

other_texts = [
    "Ciao, come stai?",            # Italian
    "Привет, как дела?",           # Russian
    "こんにちは、お元気ですか？",        # Japanese
    "你好，你怎么样？",                 # Chinese
    "مرحبا كيف حالك؟"                 # Arabic
]

#Out of dataset languages 
print("\n--- Other Dataset Languages Test ---\n")
for text in other_texts:
    label, conf = predict_language(text)
    print(f"Text: {text}")
    print(f"Predicted language: {label} (Confidence: {conf:.2f})\n")

low_resource_texts = [
    "Sawubona, unjani?",              # Zulu
    "Sannu, lafiya lau?",             # Hausa
    "Salve, quid agis?",              # Latin 
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",   # Punjabi 
    "Mingalaba, ne kaung la?"         # Burmese
]

print("\n--- Low-Resource Languages Test (Out-of-Dataset) ---\n")
for text in low_resource_texts:
    label, conf = predict_language(text)
    print(f"Text: {text}")
    print(f"Predicted language: {label} (Confidence: {conf:.2f})\n")