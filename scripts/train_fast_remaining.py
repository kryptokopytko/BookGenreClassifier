"""
Fast training of remaining models using pre-trained TF-IDF vectorizer.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, RIDGE_PARAMS

MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FAST TRAINING - USING EXISTING TF-IDF VECTORIZER")
print("="*80)

# Load pre-trained TF-IDF vectorizer
print("\n📦 Loading pre-trained TF-IDF vectorizer...")
vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
if not vectorizer_path.exists():
    print(f"❌ Vectorizer not found at {vectorizer_path}")
    print("Please run train_simple.py first to create the vectorizer.")
    sys.exit(1)

vectorizer = joblib.load(vectorizer_path)
print(f"✓ Loaded vectorizer with {len(vectorizer.vocabulary_)} features")

# Load data
print("\n📂 Loading data splits...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

def load_texts(df: pd.DataFrame) -> list:
    """Load text content from file paths."""
    texts = []
    for idx, row in df.iterrows():
        try:
            text_path = Path(row['processed_path'])
            if not text_path.is_absolute():
                text_path = PROCESSED_DATA_DIR.parent / text_path
            text = text_path.read_text(encoding='utf-8')
            texts.append(text)
        except Exception as e:
            print(f"⚠️  Error loading text {idx}: {e}")
            texts.append("")
    return texts

# Load and vectorize texts ONCE
print("\n📝 Loading texts from files...")
print("  - Training texts...")
texts_train = load_texts(train_df)
y_train = train_df['genre'].values

print("  - Validation texts...")
texts_val = load_texts(val_df)
y_val = val_df['genre'].values

print("  - Test texts...")
texts_test = load_texts(test_df)
y_test = test_df['genre'].values

print(f"\n✓ Loaded {len(texts_train)} train, {len(texts_val)} val, {len(texts_test)} test texts")

# Vectorize texts using pre-trained vectorizer
print("\n🔢 Vectorizing texts (using existing vectorizer)...")
print("  - Transforming training texts...")
X_train = vectorizer.transform(texts_train)
print(f"    Shape: {X_train.shape}")

print("  - Transforming validation texts...")
X_val = vectorizer.transform(texts_val)
print(f"    Shape: {X_val.shape}")

print("  - Transforming test texts...")
X_test = vectorizer.transform(texts_test)
print(f"    Shape: {X_test.shape}")

print(f"\n✓ All texts vectorized successfully!")

results = []

# ============================================================================
# 1. RIDGE CLASSIFIER
# ============================================================================
print("\n" + "="*80)
print("1. TRAINING: RIDGE CLASSIFIER")
print("="*80)

start_time = time.time()

ridge_model = RidgeClassifier(**RIDGE_PARAMS)
print(f"Parameters: {RIDGE_PARAMS}")

print("Training...")
ridge_model.fit(X_train, y_train)

train_acc = ridge_model.score(X_train, y_train)
val_acc = ridge_model.score(X_val, y_val)
test_acc = ridge_model.score(X_test, y_test)

y_pred_test = ridge_model.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_test, average='weighted', zero_division=0
)

elapsed = time.time() - start_time

print(f"\n✓ Ridge Classifier trained in {elapsed:.1f}s")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Val Accuracy:   {val_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Test F1:        {f1:.4f}")

# Save model
ridge_save = {
    'model': ridge_model,
    'vectorizer': vectorizer,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_accuracy': test_acc
}
joblib.dump(ridge_save, MODELS_DIR / "ridge_classifier.pkl")
print(f"  Saved to: ridge_classifier.pkl")

# Save classification report
report = classification_report(y_test, y_pred_test)
report_path = Path('results') / 'classification_report_ridge.txt'
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report)
print(f"  Classification report saved to: {report_path}")

results.append({
    'model': 'Ridge Classifier',
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc,
    'test_f1': f1,
    'time_sec': elapsed
})

# ============================================================================
# 2. NEAREST CENTROID
# ============================================================================
print("\n" + "="*80)
print("2. TRAINING: NEAREST CENTROID")
print("="*80)

start_time = time.time()

nc_model = NearestCentroid(metric='euclidean')
print("Parameters: metric='euclidean'")

print("Training...")
nc_model.fit(X_train, y_train)

train_acc = nc_model.score(X_train, y_train)
val_acc = nc_model.score(X_val, y_val)
test_acc = nc_model.score(X_test, y_test)

y_pred_test = nc_model.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_test, average='weighted', zero_division=0
)

elapsed = time.time() - start_time

print(f"\n✓ Nearest Centroid trained in {elapsed:.1f}s")
print(f"  Train Accuracy: {train_acc:.4f}")
print(f"  Val Accuracy:   {val_acc:.4f}")
print(f"  Test Accuracy:  {test_acc:.4f}")
print(f"  Test F1:        {f1:.4f}")

# Save model
nc_save = {
    'model': nc_model,
    'vectorizer': vectorizer,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_accuracy': test_acc
}
joblib.dump(nc_save, MODELS_DIR / "nearest_centroid.pkl")
print(f"  Saved to: nearest_centroid.pkl")

# Save classification report
report = classification_report(y_test, y_pred_test)
report_path = Path('results') / 'classification_report_nearest_centroid.txt'
report_path.write_text(report)
print(f"  Classification report saved to: {report_path}")

results.append({
    'model': 'Nearest Centroid',
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc,
    'test_f1': f1,
    'time_sec': elapsed
})

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_file = Path('results') / 'remaining_models_results.csv'
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
print(f"Model:         {best_model['model']}")
print(f"Train Acc:     {best_model['train_acc']:.4f}")
print(f"Val Acc:       {best_model['val_acc']:.4f}")
print(f"Test Acc:      {best_model['test_acc']:.4f}")
print(f"Test F1:       {best_model['test_f1']:.4f}")
print(f"Training Time: {best_model['time_sec']:.1f}s")

print("\n" + "="*80)
print("✅ ALL TRAINING COMPLETE!")
print("="*80)
