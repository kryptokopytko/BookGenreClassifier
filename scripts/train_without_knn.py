"""
Train all models EXCEPT KNN (which is slow).
"""

import sys
from pathlib import Path
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TRAINING ALL MODELS (SKIPPING KNN)")
print("="*80)

print("\nLoading data splits...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

results = []

def train_and_evaluate(model_name, train_func):
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)

    start_time = time.time()

    try:
        result = train_func()
        elapsed = time.time() - start_time
        print(f"\n✓ {model_name} completed in {elapsed:.1f}s")

        if result:
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

        return True

    except Exception as e:
        print(f"\n✗ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_ridge():
    from src.models.ridge_model import train_ridge_model

    print("\nTraining Ridge Classifier...")
    model = train_ridge_model(
        train_file=PROCESSED_DATA_DIR / "train.csv",
        val_file=PROCESSED_DATA_DIR / "val.csv",
        save_dir=MODELS_DIR
    )

    texts_test, y_test = model.prepare_data(test_df)
    test_metrics = model.evaluate(texts_test, y_test)

    return {
        'model': 'Ridge Classifier',
        'train_acc': model.train_accuracy,
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['f1']
    }

def train_nearest_centroid():
    from src.models.nearest_centroid_model import train_nearest_centroid_model

    print("\nTraining Nearest Centroid model...")
    model = train_nearest_centroid_model(
        train_file=PROCESSED_DATA_DIR / "train.csv",
        val_file=PROCESSED_DATA_DIR / "val.csv",
        save_dir=MODELS_DIR
    )

    texts_test, y_test = model.prepare_data(test_df)
    test_metrics = model.evaluate(texts_test, y_test)

    return {
        'model': 'Nearest Centroid',
        'train_acc': model.train_accuracy,
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['f1']
    }

def train_style():
    from src.models.style_model import train_style_model

    print("\nTraining Style-based model...")
    model, scaler, metrics = train_style_model()

    return {
        'model': 'Style Features',
        'train_acc': None,
        'test_acc': metrics['accuracy'],
        'test_f1': metrics['f1']
    }

def train_tfidf_variants():
    from src.models.tfidf_model import train_tfidf_model

    results_list = []

    for algorithm in ['logistic', 'random_forest']:
        print(f"\nTraining TF-IDF + {algorithm.upper()}...")

        model = train_tfidf_model(
            train_file=PROCESSED_DATA_DIR / "train.csv",
            val_file=PROCESSED_DATA_DIR / "val.csv",
            algorithm=algorithm,
            save_dir=MODELS_DIR
        )

        texts_test, y_test = model.prepare_data(test_df)
        test_metrics = model.evaluate(texts_test, y_test)

        results_list.append({
            'model': f'TF-IDF + {algorithm.title()}',
            'train_acc': None,
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1']
        })

    return results_list

print("\n" + "="*80)
print("STARTING TRAINING SEQUENCE (NO KNN)")
print("="*80)

train_and_evaluate("Ridge Classifier", train_ridge)
train_and_evaluate("Nearest Centroid", train_nearest_centroid)
train_and_evaluate("Style-based Model", train_style)
train_and_evaluate("TF-IDF Variants", train_tfidf_variants)

print("\n" + "="*80)
print("TRAINING COMPLETE - SAVING RESULTS")
print("="*80)

if results:
    results_df = pd.DataFrame(results)
    results_file = Path('results') / 'models_without_knn.csv'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_file, index=False)

    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))

    best_idx = results_df['test_acc'].idxmax()
    best_model = results_df.iloc[best_idx]

    print("\n" + "="*80)
    print("🏆 BEST MODEL (excluding KNN)")
    print("="*80)
    print(f"Model: {best_model['model']}")
    print(f"Test Accuracy: {best_model['test_acc']:.4f}")
    print(f"Test F1: {best_model['test_f1']:.4f}")
else:
    print("\n⚠️  No models were successfully trained!")

print("\n" + "="*80)
print("DONE!")
print("="*80)
