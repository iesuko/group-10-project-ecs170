import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
 
detection_model_name = "papluca/xlm-roberta-base-language-detection"
detection_tokenizer = AutoTokenizer.from_pretrained(detection_model_name)
detection_model = AutoModelForSequenceClassification.from_pretrained(detection_model_name)

print("Loaded Language Detector")
print("Label mapping:", detection_model.config.id2label)

def predict_language(text):
    """Detect language and return (label, confidence)."""
    inputs = detection_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = detection_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()
        label = detection_model.config.id2label[pred_id]
    return label, confidence

translation_model_name = "facebook/m2m100_418M" 
translation_tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)

print("\nLoaded M2M100 Translation Model\n")

dataset = {
    "en": "en",
    "de": "de",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "ru": "ru",
    "zh": "zh",
    "ja": "ja",
    "ar": "ar",
}

def safe_translate(text, target_lang="en", min_conf=0.60):
    """
    Translation ALWAYS runs. No confidence checks, no language filter.
    Returns: (detected_lang, confidence, translation)
    """
    label, confidence = predict_language(text)

    src_lang = dataset.get(label, "en")
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

high_resource_texts = [
    "Hello, how are you?",            
    "Guten Morgen, wie geht es dir?", 
    "Bonjour, je suis étudiant.",     
    "Hola, ¿cómo estás?"              
]

other_texts = [
    "Ciao, come stai?",            
    "Привет, как дела?",           
    "こんにちは、お元気ですか？",       
    "你好，你怎么样？",                
    "مرحبا كيف حالك؟"                
]

low_resource_texts = [
    "Sawubona, unjani?",                 
    "Sannu, lafiya lau?",                
    "Salve, quid agis?",                 
    "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",     
    "Mingalaba, ne kaung la?"            
]

def run_tests(name, data):
    print(f"\n--- {name} ---\n")
    for text in data:
        lang, conf, trans = safe_translate(text)
        print(f"Text: {text}")
        print(f"Detected: {lang}  (Confidence: {conf:.2f})")
        print(f"Translation result: {trans}")
        print()

validation_set = [
    {"text": "Hello, how are you?", "lang": "en", "translation_en": "Hello, how are you?"},
    {"text": "Guten Morgen, wie geht es dir?", "lang": "de", "translation_en": "Good morning, how are you?"},
    {"text": "Bonjour, je suis étudiant.", "lang": "fr", "translation_en": "Hello, I am a student."},
    {"text": "Hola, ¿cómo estás?", "lang": "es", "translation_en": "Hello, how are you?"},
    {"text": "Ciao, come stai?", "lang": "it", "translation_en": "Hello, how are you?"},
    {"text": "Привет, как дела?", "lang": "ru", "translation_en": "Hello, how are you?"},
    {"text": "こんにちは、お元気ですか？", "lang": "ja", "translation_en": "Hello, how are you?"},
    {"text": "你好，你怎么样？", "lang": "zh", "translation_en": "Hello, how are you?"},
    {"text": "مرحبا كيف حالك؟", "lang": "ar", "translation_en": "Hello, how are you?"},
]

def validate_model(validation_set, target_lang="en", min_conf=0.6):
    total = len(validation_set)
    correct_detection = 0
    successful_translation = 0

    print("\n--- Validation Results ---\n")
    for sample in validation_set:
        text = sample["text"]
        true_lang = sample["lang"]
        ref_translation = sample.get("translation_en")

        detected_lang, confidence, translation = safe_translate(text, target_lang=target_lang)

        detection_match = detected_lang == true_lang
        if detection_match:
            correct_detection += 1

        translation_match = False
        if ref_translation:
            translation_match = translation.strip() == ref_translation.strip()
            if translation_match:
                successful_translation += 1

        print(f"Text: {text}")
        print(f"True Language: {true_lang} | Detected: {detected_lang} | Confidence: {confidence:.2f}")
        print(f"Translation: {translation}")
        print(f"Detection Correct: {detection_match} | Translation Correct: {translation_match}")
        print("-" * 50)

    detection_accuracy = correct_detection / total
    translation_success_rate = successful_translation / total

    print(f"\nDetection Accuracy: {detection_accuracy:.2f}")
    print(f"Translation Success Rate: {translation_success_rate:.2f}")

run_tests("High-Resource Languages Test", high_resource_texts)
run_tests("Other Supported Languages Test", other_texts)
validate_model(validation_set)

