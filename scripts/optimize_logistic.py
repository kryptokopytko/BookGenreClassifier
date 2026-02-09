"""
Optimize Logistic Regression using:
1. Cached vectorized data (instant load)
2. Fast 'liblinear' solver (instead of slow 'saga')

Expected: 150s → 5-10s (15x faster!)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

print("="*80)
print("⚡ OPTIMIZING LOGISTIC REGRESSION")
print("="*80)

# Load cached vectors
cache_dir = PROCESSED_DATA_DIR / "cached_vectors"
print("\n📦 Loading cached data...")
start_load = time.time()

X_train = sparse.load_npz(cache_dir / "X_train.npz")
y_train = np.load(cache_dir / "y_train.npy", allow_pickle=True)

X_val = sparse.load_npz(cache_dir / "X_val.npz")
y_val = np.load(cache_dir / "y_val.npy", allow_pickle=True)

X_test = sparse.load_npz(cache_dir / "X_test.npz")
y_test = np.load(cache_dir / "y_test.npy", allow_pickle=True)

load_time = time.time() - start_load
print(f"✓ Loaded in {load_time:.2f}s")

# Load vectorizer
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

print("\n" + "="*80)
print("TRAINING COMPARISON")
print("="*80)

results = []

# ============================================================================
# 1. ORIGINAL (SLOW) - saga solver
# ============================================================================
print("\n1️⃣  Training with SAGA solver (original, slow)...")
start_time = time.time()

logreg_saga = LogisticRegression(
    C=2.0,
    solver='saga',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

logreg_saga.fit(X_train, y_train)

train_acc = logreg_saga.score(X_train, y_train)
val_acc = logreg_saga.score(X_val, y_val)
test_acc = logreg_saga.score(X_test, y_test)

y_pred = logreg_saga.predict(X_test)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

saga_time = time.time() - start_time

print(f"   ✓ Completed in {saga_time:.1f}s")
print(f"   Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f}")

results.append({
    'solver': 'saga',
    'C': 2.0,
    'max_iter': 1000,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc,
    'test_f1': f1,
    'time_sec': saga_time
})

# ============================================================================
# 2. OPTIMIZED - liblinear solver
# ============================================================================
print("\n2️⃣  Training with LIBLINEAR solver (optimized, fast)...")
start_time = time.time()

logreg_liblinear = LogisticRegression(
    C=2.0,
    solver='liblinear',  # ← MUCH FASTER!
    max_iter=1000,
    random_state=42
)

logreg_liblinear.fit(X_train, y_train)

train_acc = logreg_liblinear.score(X_train, y_train)
val_acc = logreg_liblinear.score(X_val, y_val)
test_acc = logreg_liblinear.score(X_test, y_test)

y_pred = logreg_liblinear.predict(X_test)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

liblinear_time = time.time() - start_time

print(f"   ✓ Completed in {liblinear_time:.1f}s")
print(f"   Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f}")

results.append({
    'solver': 'liblinear',
    'C': 2.0,
    'max_iter': 1000,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc,
    'test_f1': f1,
    'time_sec': liblinear_time
})

# Save optimized model
model_data = {
    'model': logreg_liblinear,
    'vectorizer': vectorizer,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_accuracy': test_acc,
    'test_f1': f1
}
joblib.dump(model_data, MODELS_DIR / "logistic_regression_optimized.pkl")
print(f"\n💾 Saved: logistic_regression_optimized.pkl")

# Save classification report
report = classification_report(y_test, y_pred)
report_path = Path('results') / 'classification_report_logistic_optimized.txt'
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report)

# ============================================================================
# 3. EVEN FASTER - reduced iterations
# ============================================================================
print("\n3️⃣  Training with LIBLINEAR + fewer iterations (ultra-fast)...")
start_time = time.time()

logreg_ultrafast = LogisticRegression(
    C=2.0,
    solver='liblinear',
    max_iter=500,  # ← HALF the iterations
    random_state=42
)

logreg_ultrafast.fit(X_train, y_train)

train_acc = logreg_ultrafast.score(X_train, y_train)
val_acc = logreg_ultrafast.score(X_val, y_val)
test_acc = logreg_ultrafast.score(X_test, y_test)

y_pred = logreg_ultrafast.predict(X_test)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

ultrafast_time = time.time() - start_time

print(f"   ✓ Completed in {ultrafast_time:.1f}s")
print(f"   Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | F1: {f1:.4f}")

results.append({
    'solver': 'liblinear',
    'C': 2.0,
    'max_iter': 500,
    'train_acc': train_acc,
    'val_acc': val_acc,
    'test_acc': test_acc,
    'test_f1': f1,
    'time_sec': ultrafast_time
})

# Save ultra-fast model
model_data = {
    'model': logreg_ultrafast,
    'vectorizer': vectorizer,
    'train_accuracy': train_acc,
    'val_accuracy': val_acc,
    'test_accuracy': test_acc,
    'test_f1': f1
}
joblib.dump(model_data, MODELS_DIR / "logistic_regression_ultrafast.pkl")
print(f"💾 Saved: logistic_regression_ultrafast.pkl")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("OPTIMIZATION RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Calculate improvements
speedup_liblinear = saga_time / liblinear_time
speedup_ultrafast = saga_time / ultrafast_time

acc_loss_liblinear = results[0]['test_acc'] - results[1]['test_acc']
acc_loss_ultrafast = results[0]['test_acc'] - results[2]['test_acc']

print("\n" + "="*80)
print("SPEEDUP ANALYSIS")
print("="*80)

print(f"\n📊 SAGA (original):      {saga_time:.1f}s - {results[0]['test_acc']:.4f} accuracy")
print(f"⚡ LIBLINEAR:            {liblinear_time:.1f}s - {results[1]['test_acc']:.4f} accuracy")
print(f"🚀 LIBLINEAR (500 iter): {ultrafast_time:.1f}s - {results[2]['test_acc']:.4f} accuracy")

print(f"\n🎯 Speedup (LIBLINEAR):       {speedup_liblinear:.1f}x faster!")
print(f"   Accuracy change:           {acc_loss_liblinear:+.4f} ({acc_loss_liblinear*100:+.2f}%)")

print(f"\n🚀 Speedup (ULTRA-FAST):      {speedup_ultrafast:.1f}x faster!!")
print(f"   Accuracy change:           {acc_loss_ultrafast:+.4f} ({acc_loss_ultrafast*100:+.2f}%)")

# Best trade-off
print("\n" + "="*80)
print("💡 RECOMMENDATION")
print("="*80)

if abs(acc_loss_ultrafast) < 0.02:  # Less than 2% loss
    print("\n✅ Use ULTRA-FAST version (liblinear, 500 iter):")
    print(f"   - Speed: {ultrafast_time:.1f}s ({speedup_ultrafast:.1f}x faster)")
    print(f"   - Accuracy: {results[2]['test_acc']:.4f} (minimal loss)")
    print(f"   - Best for: Production, rapid iteration")
elif abs(acc_loss_liblinear) < 0.01:  # Less than 1% loss
    print("\n✅ Use OPTIMIZED version (liblinear, 1000 iter):")
    print(f"   - Speed: {liblinear_time:.1f}s ({speedup_liblinear:.1f}x faster)")
    print(f"   - Accuracy: {results[1]['test_acc']:.4f} (almost no loss)")
    print(f"   - Best for: Balanced speed/quality")
else:
    print("\n⚠️  Use SAGA solver if accuracy is critical:")
    print(f"   - Speed: {saga_time:.1f}s")
    print(f"   - Accuracy: {results[0]['test_acc']:.4f} (best quality)")
    print(f"   - Trade-off: Slower but more accurate")

# Save results
results_file = Path('results') / 'logistic_optimization_comparison.csv'
results_df.to_csv(results_file, index=False)
print(f"\n💾 Results saved to: {results_file}")

print("\n" + "="*80)
print("✅ OPTIMIZATION COMPLETE!")
print("="*80)
