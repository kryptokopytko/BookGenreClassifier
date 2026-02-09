"""
Interactive model testing script.
Load trained models and test on custom text or sample books.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR

print("="*80)
print("📚 BOOK GENRE CLASSIFIER - MODEL TESTER")
print("="*80)

# ============================================================================
# ============================================================================

AVAILABLE_MODELS = {
    "1": ("Linear SVM", "linear_svm.pkl", "tfidf_vectorizer.pkl"),
    "2": ("Logistic Regression", "logistic_regression.pkl", "tfidf_vectorizer.pkl"),
    "3": ("Naive Bayes", "naive_bayes.pkl", "tfidf_vectorizer.pkl"),
    "4": ("Random Forest", "random_forest.pkl", "tfidf_vectorizer.pkl"),
    "5": ("KNN", "knn_model.pkl", "knn_vectorizer.pkl"),
    "6": ("Ridge Classifier", "ridge_model.pkl", None),
    "7": ("Nearest Centroid", "nearest_centroid_model.pkl", None),
}

def load_model(model_key: str):
    """Load a trained model and its vectorizer."""
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Invalid model key: {model_key}")
        return None, None

    model_name, model_file, vectorizer_file = AVAILABLE_MODELS[model_key]

    model_path = MODELS_DIR / model_file
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None

    print(f"\n📦 Loading {model_name}...")
    model = joblib.load(model_path)

    vectorizer = None
    if vectorizer_file:
        vec_path = MODELS_DIR / vectorizer_file
        if vec_path.exists():
            vectorizer = joblib.load(vec_path)
            print(f"✓ Loaded model and vectorizer")
        else:
            print(f"⚠️  Vectorizer not found: {vec_path}")
    else:
        print(f"✓ Loaded model")

    return model, vectorizer

def predict_text(text: str, model, vectorizer=None) -> Dict:
    """Make prediction on text."""
    if vectorizer:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            classes = model.classes_ if hasattr(model, 'classes_') else None
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X)[0]
            exp_scores = np.exp(scores - np.max(scores))
            proba = exp_scores / exp_scores.sum()
            classes = model.classes_ if hasattr(model, 'classes_') else None
        else:
            proba = None
            classes = None

        return {
            'prediction': pred,
            'probabilities': proba,
            'classes': classes
        }
    else:
        print("⚠️  Feature-based models need extracted features, not raw text")
        return None

def display_prediction(result: Dict, show_probabilities: bool = True):
    """Display prediction results."""
    print("\n" + "="*80)
    print("🎯 PREDICTION RESULT")
    print("="*80)

    print(f"\n**Predicted Genre:** {result['prediction']}")

    if show_probabilities and result['probabilities'] is not None:
        print(f"\n**Confidence Scores:**")

        proba_sorted = sorted(
            zip(result['classes'], result['probabilities']),
            key=lambda x: x[1],
            reverse=True
        )

        for genre, prob in proba_sorted:
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {genre:30s} {bar} {prob*100:5.2f}%")

# ============================================================================
# ============================================================================

SAMPLE_TEXTS = {
    "1": {
        "name": "Romance",
        "text": """She gazed into his eyes, her heart beating wildly in her chest.
        The moonlight cast a soft glow on his face as he leaned closer.
        "I've loved you since the moment I first saw you," he whispered,
        his voice filled with passion and longing. Their lips met in a kiss
        that seemed to last forever, sealing their eternal love."""
    },
    "2": {
        "name": "Science Fiction",
        "text": """The starship approached the alien planet, its hull gleaming in
        the light of three suns. Captain Rodriguez checked the quantum sensors
        as they entered orbit. "Anomalous readings from the surface," reported
        the AI. "Detecting advanced technology and possible artificial structures."
        They had found what they were looking for - evidence of an ancient
        alien civilization."""
    },
    "3": {
        "name": "Mystery/Crime",
        "text": """Detective Morgan examined the crime scene carefully. Blood
        spatters on the wall told a story of violence. The victim's fingerprints
        were on the murder weapon, but something didn't add up. There was
        evidence of a struggle, yet the security cameras showed no one entering
        or leaving. This case was more complicated than it appeared."""
    },
    "4": {
        "name": "Historical Fiction",
        "text": """The year was 1776, and revolution was in the air. General
        Washington reviewed his troops as they prepared to cross the Delaware.
        The British forces were strong, but the colonists' determination was
        stronger. Tomorrow would bring the battle that could change the course
        of history forever."""
    },
    "5": {
        "name": "Biography",
        "text": """Born in 1920 in rural Mississippi, she overcame poverty and
        discrimination to become one of the most influential scientists of
        her generation. Her early years were marked by hardship, but her
        brilliant mind and determination led her to win a scholarship to
        university. By age 30, she had made discoveries that would change
        medicine forever."""
    },
}

# ============================================================================
# ============================================================================

def main():
    print("\n📋 Available Models:")
    for key, (name, _, _) in AVAILABLE_MODELS.items():
        model_path = MODELS_DIR / AVAILABLE_MODELS[key][1]
        status = "✓" if model_path.exists() else "✗"
        print(f"  {key}. {status} {name}")

    print("\n" + "-"*80)
    model_key = input("\nSelect model (1-7) or 'q' to quit: ").strip()

    if model_key.lower() == 'q':
        print("\n👋 Goodbye!")
        return

    model, vectorizer = load_model(model_key)
    if model is None:
        return

    while True:
        print("\n" + "="*80)
        print("📝 TEST OPTIONS")
        print("="*80)
        print("  1. Test with sample texts")
        print("  2. Enter custom text")
        print("  3. Load text from file")
        print("  4. Switch model")
        print("  q. Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == 'q':
            print("\n👋 Goodbye!")
            break

        elif choice == '1':
            print("\n📚 Sample Texts:")
            for key, sample in SAMPLE_TEXTS.items():
                print(f"  {key}. {sample['name']}")

            sample_key = input("\nSelect sample (1-5): ").strip()

            if sample_key in SAMPLE_TEXTS:
                sample = SAMPLE_TEXTS[sample_key]
                print(f"\n📖 Testing with sample: {sample['name']}")
                print(f"\nText preview: {sample['text'][:200]}...")

                result = predict_text(sample['text'], model, vectorizer)
                if result:
                    display_prediction(result)

                    if result['prediction'] == sample['name']:
                        print("\n✅ CORRECT! Model predicted the right genre.")
                    else:
                        print(f"\n❌ INCORRECT. Expected: {sample['name']}")

        elif choice == '2':
            print("\n✍️  Enter your text (end with Ctrl+D or empty line):")
            lines = []
            try:
                while True:
                    line = input()
                    if not line:
                        break
                    lines.append(line)
            except EOFError:
                pass

            if lines:
                text = ' '.join(lines)
                result = predict_text(text, model, vectorizer)
                if result:
                    display_prediction(result)

        elif choice == '3':
            file_path = input("\n📁 Enter file path: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                print(f"\n✓ Loaded {len(text)} characters from file")
                print(f"Preview: {text[:200]}...")

                result = predict_text(text, model, vectorizer)
                if result:
                    display_prediction(result)

            except Exception as e:
                print(f"\n❌ Error loading file: {e}")

        elif choice == '4':
            main()
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
