"""
Comprehensive training script for all book genre classification models.
Trains all available models and saves results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BOOK GENRE CLASSIFIER - TRAINING ALL MODELS")
print("="*80)

print("\nLoading data splits...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

results = []

def train_and_evaluate(model_name, train_func):
    """Train a model and evaluate it."""
    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)

    start_time = time.time()

    try:
        result = train_func()

        elapsed = time.time() - start_time
        print(f"\n✓ {model_name} completed in {elapsed:.1f}s")

        if result:
            results.append(result)

        return True

    except Exception as e:
        print(f"\n✗ {model_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# ============================================================================

def train_knn():
    from src.models.knn_model import train_knn_model

    print("\nTraining K-Nearest Neighbors model...")
    model = train_knn_model(
        train_file=PROCESSED_DATA_DIR / "train.csv",
        val_file=PROCESSED_DATA_DIR / "val.csv",
        save_dir=MODELS_DIR
    )

    texts_test, y_test = model.prepare_data(test_df)
    test_metrics = model.evaluate(texts_test, y_test)

    return {
        'model': 'KNN',
        'train_acc': model.train_accuracy,
        'test_acc': test_metrics['accuracy'],
        'test_f1': test_metrics['f1']
    }

# ============================================================================
# ============================================================================

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

# ============================================================================
# ============================================================================

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

# ============================================================================
# ============================================================================

def train_style():
    from src.models.style_model import train_style_model

    print("\nTraining Style-based model...")
    model, scaler, metrics = train_style_model()

    return {
        'model': 'Style Features',
        'train_acc': None,  # Not tracked separately
        'test_acc': metrics['accuracy'],
        'test_f1': metrics['f1']
    }

# ============================================================================
# ============================================================================

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

# ============================================================================
# ============================================================================

def train_feature_models():
    features_file = PROCESSED_DATA_DIR / "features.csv"

    if not features_file.exists():
        print("\n⚠️  features.csv not found - skipping feature-based models")
        print("    Run feature extraction first to train these models")
        return []

    results_list = []

    try:
        from src.models.xgboost_model import train_from_split_files as train_xgb

        print("\nTraining XGBoost model...")
        model = train_xgb(
            features_file=features_file,
            train_file=PROCESSED_DATA_DIR / "train.csv",
            val_file=PROCESSED_DATA_DIR / "val.csv",
            save_dir=MODELS_DIR
        )

        features_df = pd.read_csv(features_file)
        test_features = features_df[features_df['book_id'].isin(test_df['book_id'])]
        X_test, y_test = model.prepare_data(test_features)
        test_metrics = model.evaluate(X_test, y_test)

        results_list.append({
            'model': 'XGBoost',
            'train_acc': model.train_accuracy,
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1']
        })
    except ImportError:
        print("⚠️  XGBoost not installed - skipping")
    except Exception as e:
        print(f"⚠️  XGBoost training failed: {e}")

    try:
        from src.models.lightgbm_model import train_from_split_files as train_lgbm

        print("\nTraining LightGBM model...")
        model = train_lgbm(
            features_file=features_file,
            train_file=PROCESSED_DATA_DIR / "train.csv",
            val_file=PROCESSED_DATA_DIR / "val.csv",
            save_dir=MODELS_DIR
        )

        features_df = pd.read_csv(features_file)
        test_features = features_df[features_df['book_id'].isin(test_df['book_id'])]
        X_test, y_test = model.prepare_data(test_features)
        test_metrics = model.evaluate(X_test, y_test)

        results_list.append({
            'model': 'LightGBM',
            'train_acc': model.train_accuracy,
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1']
        })
    except ImportError:
        print("⚠️  LightGBM not installed - skipping")
    except Exception as e:
        print(f"⚠️  LightGBM training failed: {e}")

    try:
        from src.models.feature_model import train_from_split_files as train_feat

        print("\nTraining Feature-based Random Forest...")
        model = train_feat(
            features_file=features_file,
            train_file=PROCESSED_DATA_DIR / "train.csv",
            val_file=PROCESSED_DATA_DIR / "val.csv",
            save_dir=MODELS_DIR
        )

        features_df = pd.read_csv(features_file)
        test_features = features_df[features_df['book_id'].isin(test_df['book_id'])]
        X_test, y_test = model.prepare_data(test_features)
        test_metrics = model.evaluate(X_test, y_test)

        results_list.append({
            'model': 'Feature-based RF',
            'train_acc': None,
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1']
        })
    except Exception as e:
        print(f"⚠️  Feature-based RF training failed: {e}")

    return results_list

# ============================================================================
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING SEQUENCE")
print("="*80)

train_and_evaluate("KNN Model", train_knn)
train_and_evaluate("Ridge Classifier", train_ridge)
train_and_evaluate("Nearest Centroid", train_nearest_centroid)
train_and_evaluate("Style-based Model", train_style)

print("\n" + "="*80)
print("TRAINING: TF-IDF Variants")
print("="*80)
try:
    tfidf_results = train_tfidf_variants()
    results.extend(tfidf_results)
except Exception as e:
    print(f"✗ TF-IDF variants failed: {e}")

print("\n" + "="*80)
print("TRAINING: Feature-based Models")
print("="*80)
try:
    feat_results = train_feature_models()
    results.extend(feat_results)
except Exception as e:
    print(f"✗ Feature-based models failed: {e}")

# ============================================================================
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE - SAVING RESULTS")
print("="*80)

if results:
    results_df = pd.DataFrame(results)
    results_file = MODELS_DIR.parent / 'results' / 'all_models_results.csv'
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
    print("🏆 BEST MODEL")
    print("="*80)
    print(f"Model: {best_model['model']}")
    print(f"Test Accuracy: {best_model['test_acc']:.4f}")
    print(f"Test F1: {best_model['test_f1']:.4f}")
else:
    print("\n⚠️  No models were successfully trained!")

print("\n" + "="*80)
print("DONE!")
print("="*80)
