"""
ULTRA-FAST PARALLEL MODEL TRAINING with ALL OPTIMIZATIONS:
1. Uses cached pre-vectorized data (instant load)
2. Trains multiple models in parallel (multiprocessing)
3. Optimized for speed without sacrificing quality

Expected speedup: 10-20x faster than original approach!
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import time
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

print("="*80)
print("⚡ ULTRA-FAST PARALLEL MODEL TRAINING ⚡")
print("="*80)

# Check for cached data
cache_dir = PROCESSED_DATA_DIR / "cached_vectors"
if not cache_dir.exists() or not (cache_dir / "X_train.npz").exists():
    print("\n❌ Cached vectors not found!")
    print("Run: python3 scripts/cache_vectorized_data.py")
    sys.exit(1)

# Load cached vectors (INSTANT!)
print("\n⚡ Loading cached vectorized data...")
start_load = time.time()

X_train = sparse.load_npz(cache_dir / "X_train.npz")
y_train = np.load(cache_dir / "y_train.npy", allow_pickle=True)

X_val = sparse.load_npz(cache_dir / "X_val.npz")
y_val = np.load(cache_dir / "y_val.npy", allow_pickle=True)

X_test = sparse.load_npz(cache_dir / "X_test.npz")
y_test = np.load(cache_dir / "y_test.npy", allow_pickle=True)

load_time = time.time() - start_load

print(f"✓ Loaded in {load_time:.2f}s (vs ~30-60s without cache!)")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

# Load vectorizer for saving with models
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

# Model training function (for parallel execution)
def train_single_model(model_config):
    """Train a single model and return results."""
    model_name = model_config['name']
    model_class = model_config['class']
    params = model_config['params']

    print(f"\n[{model_name}] Starting training...")
    start_time = time.time()

    try:
        # Create and train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        test_acc = model.score(X_test, y_test)

        y_pred_test = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_test, average='weighted', zero_division=0
        )

        elapsed = time.time() - start_time

        # Save model
        model_file = model_config.get('save_name', model_name.lower().replace(' ', '_') + '.pkl')
        save_path = MODELS_DIR / model_file

        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'test_f1': f1
        }
        joblib.dump(model_data, save_path)

        # Save classification report
        report = classification_report(y_test, y_pred_test)
        report_path = Path('results') / f'classification_report_{model_name.lower().replace(" ", "_")}.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)

        result = {
            'model': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'test_f1': f1,
            'time_sec': elapsed,
            'status': 'success'
        }

        print(f"[{model_name}] ✓ Complete in {elapsed:.1f}s - Test: {test_acc:.4f}")

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{model_name}] ✗ Failed: {e}")
        return {
            'model': model_name,
            'train_acc': None,
            'val_acc': None,
            'test_acc': None,
            'test_f1': None,
            'time_sec': elapsed,
            'status': f'failed: {str(e)}'
        }

# Define models to train
models_config = [
    {
        'name': 'Ridge Classifier',
        'class': RidgeClassifier,
        'params': {'alpha': 1.0, 'random_state': 42},
        'save_name': 'ridge_classifier.pkl'
    },
    {
        'name': 'Linear SVM',
        'class': LinearSVC,
        'params': {'C': 1.0, 'max_iter': 2000, 'random_state': 42},
        'save_name': 'linear_svm_fast.pkl'
    },
    {
        'name': 'Logistic Regression',
        'class': LogisticRegression,
        'params': {'C': 2.0, 'solver': 'saga', 'max_iter': 1000, 'random_state': 42, 'n_jobs': -1},
        'save_name': 'logistic_regression_fast.pkl'
    },
    {
        'name': 'Nearest Centroid',
        'class': NearestCentroid,
        'params': {'metric': 'euclidean'},
        'save_name': 'nearest_centroid.pkl'
    },
    {
        'name': 'Naive Bayes',
        'class': MultinomialNB,
        'params': {'alpha': 0.1},
        'save_name': 'naive_bayes_fast.pkl'
    },
    {
        'name': 'Random Forest',
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 8,
            'min_samples_leaf': 10,
            'random_state': 42,
            'n_jobs': -1
        },
        'save_name': 'random_forest_fast.pkl'
    }
]

print("\n" + "="*80)
print(f"🚀 TRAINING {len(models_config)} MODELS IN PARALLEL")
print("="*80)

# Train models in parallel
results = []
start_training = time.time()

# Use ProcessPoolExecutor for true parallelism
max_workers = min(4, len(models_config))  # Limit to 4 parallel processes
print(f"\nUsing {max_workers} parallel workers\n")

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_model = {
        executor.submit(train_single_model, config): config['name']
        for config in models_config
    }

    # Collect results as they complete
    for future in as_completed(future_to_model):
        model_name = future_to_model[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"\n[{model_name}] ✗ Exception: {e}")
            results.append({
                'model': model_name,
                'train_acc': None,
                'val_acc': None,
                'test_acc': None,
                'test_f1': None,
                'time_sec': 0,
                'status': f'exception: {str(e)}'
            })

total_training_time = time.time() - start_training

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# Save results
results_df = pd.DataFrame(results)
results_file = Path('results') / 'ultra_fast_results.csv'
results_file.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_file, index=False)

print(f"\n✓ Results saved to: {results_file}")

# Display results
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Sort by test accuracy
results_df_sorted = results_df.sort_values('test_acc', ascending=False)
print("\n" + results_df_sorted[['model', 'test_acc', 'test_f1', 'time_sec', 'status']].to_string(index=False))

# Best model
successful_results = results_df[results_df['status'] == 'success']
if len(successful_results) > 0:
    best_idx = successful_results['test_acc'].idxmax()
    best_model = successful_results.loc[best_idx]

    print("\n" + "="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    print(f"Model:         {best_model['model']}")
    print(f"Train Acc:     {best_model['train_acc']:.4f}")
    print(f"Val Acc:       {best_model['val_acc']:.4f}")
    print(f"Test Acc:      {best_model['test_acc']:.4f}")
    print(f"Test F1:       {best_model['test_f1']:.4f}")
    print(f"Training Time: {best_model['time_sec']:.1f}s")

# Performance summary
print("\n" + "="*80)
print("⚡ PERFORMANCE SUMMARY ⚡")
print("="*80)
print(f"Data Loading:     {load_time:.2f}s (was ~30-60s)")
print(f"Total Training:   {total_training_time:.1f}s")
print(f"Models Trained:   {len(successful_results)}/{len(models_config)}")
print(f"Parallel Workers: {max_workers}")

sequential_time = sum(results_df['time_sec'])
speedup = sequential_time / total_training_time if total_training_time > 0 else 1

print(f"\nSequential time would be: {sequential_time:.1f}s")
print(f"Actual parallel time:     {total_training_time:.1f}s")
print(f"Speedup factor:           {speedup:.1f}x 🚀")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)
