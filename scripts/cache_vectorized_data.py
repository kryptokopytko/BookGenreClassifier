"""
Cache pre-vectorized training data for ultra-fast model training.

This script:
1. Loads pre-trained TF-IDF vectorizer
2. Loads all text data
3. Vectorizes once
4. Saves vectorized matrices to disk (.npz format)

Future training scripts can load these cached vectors in <1 second
instead of re-loading and vectorizing 3700+ texts every time.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

print("="*80)
print("CACHING VECTORIZED DATA")
print("="*80)

# Create cache directory
cache_dir = PROCESSED_DATA_DIR / "cached_vectors"
cache_dir.mkdir(parents=True, exist_ok=True)

# Load TF-IDF vectorizer
print("\n📦 Loading TF-IDF vectorizer...")
vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
if not vectorizer_path.exists():
    print(f"❌ Vectorizer not found at {vectorizer_path}")
    print("Please run train_simple.py first.")
    sys.exit(1)

vectorizer = joblib.load(vectorizer_path)
print(f"✓ Loaded vectorizer: {len(vectorizer.vocabulary_)} features")

# Load data splits
print("\n📂 Loading data splits...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print(f"  Train: {len(train_df)} books")
print(f"  Val:   {len(val_df)} books")
print(f"  Test:  {len(test_df)} books")

def load_texts(df: pd.DataFrame, split_name: str) -> list:
    """Load text content from file paths."""
    print(f"\n📝 Loading {split_name} texts from files...")
    texts = []
    errors = 0

    for idx, row in df.iterrows():
        try:
            text_path = Path(row['processed_path'])
            if not text_path.is_absolute():
                text_path = PROCESSED_DATA_DIR.parent / text_path
            text = text_path.read_text(encoding='utf-8')
            texts.append(text)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️  Error loading text {idx}: {e}")
            texts.append("")

    if errors > 3:
        print(f"  ⚠️  ... and {errors - 3} more errors")

    print(f"  ✓ Loaded {len(texts)} texts ({errors} errors)")
    return texts

# Load all texts
start_time = time.time()

texts_train = load_texts(train_df, "TRAIN")
y_train = train_df['genre'].values

texts_val = load_texts(val_df, "VAL")
y_val = val_df['genre'].values

texts_test = load_texts(test_df, "TEST")
y_test = test_df['genre'].values

load_time = time.time() - start_time
print(f"\n✓ All texts loaded in {load_time:.1f}s")

# Vectorize all splits
print("\n" + "="*80)
print("VECTORIZING DATA")
print("="*80)

start_time = time.time()

print("\n🔢 Transforming TRAIN split...")
X_train = vectorizer.transform(texts_train)
print(f"  Shape: {X_train.shape}")
print(f"  Sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}%")

print("\n🔢 Transforming VAL split...")
X_val = vectorizer.transform(texts_val)
print(f"  Shape: {X_val.shape}")

print("\n🔢 Transforming TEST split...")
X_test = vectorizer.transform(texts_test)
print(f"  Shape: {X_test.shape}")

vectorize_time = time.time() - start_time
print(f"\n✓ All splits vectorized in {vectorize_time:.1f}s")

# Save cached vectors
print("\n" + "="*80)
print("SAVING CACHED VECTORS")
print("="*80)

print("\n💾 Saving TRAIN vectors...")
sparse.save_npz(cache_dir / "X_train.npz", X_train)
np.save(cache_dir / "y_train.npy", y_train)
train_size = (cache_dir / "X_train.npz").stat().st_size / 1024**2
print(f"  ✓ Saved X_train.npz ({train_size:.1f} MB)")
print(f"  ✓ Saved y_train.npy")

print("\n💾 Saving VAL vectors...")
sparse.save_npz(cache_dir / "X_val.npz", X_val)
np.save(cache_dir / "y_val.npy", y_val)
val_size = (cache_dir / "X_val.npz").stat().st_size / 1024**2
print(f"  ✓ Saved X_val.npz ({val_size:.1f} MB)")
print(f"  ✓ Saved y_val.npy")

print("\n💾 Saving TEST vectors...")
sparse.save_npz(cache_dir / "X_test.npz", X_test)
np.save(cache_dir / "y_test.npy", y_test)
test_size = (cache_dir / "X_test.npz").stat().st_size / 1024**2
print(f"  ✓ Saved X_test.npz ({test_size:.1f} MB)")
print(f"  ✓ Saved y_test.npy")

total_size = train_size + val_size + test_size

# Save metadata
metadata = {
    'vectorizer_path': str(vectorizer_path),
    'n_features': len(vectorizer.vocabulary_),
    'train_samples': len(y_train),
    'val_samples': len(y_val),
    'test_samples': len(y_test),
    'created_at': pd.Timestamp.now().isoformat(),
    'load_time_sec': load_time,
    'vectorize_time_sec': vectorize_time,
    'total_size_mb': total_size
}

import json
metadata_file = cache_dir / "metadata.json"
metadata_file.write_text(json.dumps(metadata, indent=2))
print(f"\n💾 Saved metadata.json")

print("\n" + "="*80)
print("CACHE CREATION COMPLETE!")
print("="*80)

print(f"\n📊 Summary:")
print(f"  Cache Location: {cache_dir}")
print(f"  Total Size:     {total_size:.1f} MB")
print(f"  Samples:        {len(y_train)} train, {len(y_val)} val, {len(y_test)} test")
print(f"  Features:       {len(vectorizer.vocabulary_)}")
print(f"  Time Saved:     ~{load_time + vectorize_time:.1f}s per training run")

print("\n✅ Future training scripts can now load cached vectors in <1 second!")
print(f"   Use: sparse.load_npz('{cache_dir}/X_train.npz')")

print("\n" + "="*80)
