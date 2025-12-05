from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Loading the Language Detection Model 
detection_model_name = "papluca/xlm-roberta-base-language-detection"
detection_tokenizer = AutoTokenizer.from_pretrained(detection_model_name)
detection_model = AutoModelForSequenceClassification.from_pretrained(detection_model_name)

print("Loaded Language Detector") #Sanity Check
print("Label mapping:", detection_model.config.id2label) #Prints map of language codes 

#Function for predicting the language 
#Detects the language and resutns the labaguage label and confidence
def predict_language(text):
    """Detect language and return (label, confidence)."""
    inputs = detection_tokenizer(text, return_tensors="pt", truncation=True) #
    with torch.no_grad():
        outputs = detection_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()
        label = detection_model.config.id2label[pred_id]
    return label, confidence

# Loading the Translation Model
# M2M100 multilingual translation model 
# Supports ~100 language
translation_model_name = "facebook/m2m100_418M"
translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)

print("\nLoaded M2M100 Translation Model\n") #Sanity Check 

# Dataset (Detection)
# Maps the language detector codes to the translators codes 
# Models use different language codes 
dataset = {
    "en": "en",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "ru": "ru",
    "zh": "zh_CN",
    "ja": "ja",
    "ar": "ar",
}

def safe_translate(text, target_lang="en", min_conf=0.60):
    """
    Returns: (detected_lang, confidence, translation_or_message)
    Only translates if:
      - detection confidence >= min_conf
      - detected language is supported by translator
    """
    label, confidence = predict_language(text)

    if confidence < min_conf:
        return label, confidence, "Translation skipped (low confidence)."

    if label not in dataset:
        return label, confidence, "Translation skipped (language not supported by translation model)."

    src_lang = dataset[label]
    translation_tokenizer.src_lang = src_lang

    try:
        encoded = translation_tokenizer(text, return_tensors="pt")
        generated = translation_model.generate(
            **encoded,
            forced_bos_token_id=translation_tokenizer.get_lang_id(target_lang)
        )
        translation = translation_tokenizer.decode(generated[0], skip_special_tokens=True)
        return label, confidence, translation

    except Exception as e:
        return label, confidence, f"Translation failed (runtime error: {str(e)})"

#Test Cases 

# High Resource 
high_resource_texts = [
    "Hello, how are you?",            # English
    "Guten Morgen, wie geht es dir?", # German
    "Bonjour, je suis étudiant.",     # French
    "Hola, ¿cómo estás?"              # Spanish
]
# Mixed Resource 
other_texts = [
    "Ciao, come stai?",            # Italian
    "Привет, как дела?",           # Russian
    "こんにちは、お元気ですか？",       # Japanese
    "你好，你怎么样？",                # Chinese
    "مرحبا كيف حالك؟"                # Arabic
]
# Low Resource (Out of dataset)
low_resource_texts = [
    "Sawubona, unjani?",                 # Zulu
    "Sannu, lafiya lau?",                # Hausa
    "Salve, quid agis?",                 # Latin
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",     # Punjabi
    "Mingalaba, ne kaung la?"            # Burmese
]

#Function for running the tests 

def run_tests(name, data):
    print(f"\n--- {name} ---\n")
    for text in data:
        lang, conf, trans = safe_translate(text)
        print(f"Text: {text}")
        print(f"Detected: {lang}  (Confidence: {conf:.2f})")
        print(f"Translation result: {trans}")
        print()


run_tests("High-Resource Languages Test", high_resource_texts)
run_tests("Other Supported Languages Test", other_texts)
run_tests("Low-Resource / Out-of-Dataset Languages Test", low_resource_texts)