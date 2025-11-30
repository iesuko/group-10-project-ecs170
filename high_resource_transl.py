from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

gefs_model_name = "ImranzamanML/GEFS-language-detector"
gefs_tokenizer = AutoTokenizer.from_pretrained(gefs_model_name)
gefs_model = AutoModelForSequenceClassification.from_pretrained(gefs_model_name)

trans_model_name = "facebook/m2m100_418M"
trans_tokenizer = M2M100Tokenizer.from_pretrained(trans_model_name)
trans_model = M2M100ForConditionalGeneration.from_pretrained(trans_model_name)

def detect_language(text):
    inputs = gefs_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = gefs_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        label = gefs_model.config.id2label[pred_id]
        confidence = probs[0, pred_id].item()
    return label, confidence


def translate_text(text, target_lang="en"):
    src_lang, confidence = detect_language(text)
    trans_tokenizer.src_lang = src_lang
    encoded = trans_tokenizer(text, return_tensors="pt")
    generated = trans_model.generate(
        **encoded,
        forced_bos_token_id=trans_tokenizer.get_lang_id(target_lang)
    )
    translation = trans_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return src_lang, confidence, translation

test_sentences = [
    "Hello, how are you?",             # English
    "Guten Morgen, wie geht es dir?",  # German
    "Bonjour, je suis étudiant.",      # French
    "Hola, ¿cómo estás?"               # Spanish
]

print("\n--- GEFS High-Resource Languages Test ---\n")
for text in test_sentences:
    src_lang, confidence, translation = translate_text(text, target_lang="en")
    print(f"Text: {text}")
    print(f"Detected: {src_lang} (Confidence: {confidence:.2f})")
    print(f"Translation: {translation}\n")
