"""
Generate TF-IDF vectorizer for book genre classification.

This script creates and saves a TF-IDF vectorizer trained on the training set.
The vectorizer is used by other scripts (cache_vectorized_data.py, train_ultra_fast.py, etc.)

Usage:
  python3 scripts/generate_vectorizer.py

Output:
  - models_saved/tfidf_vectorizer.pkl
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

print("="*80)
print("TF-IDF VECTORIZER GENERATION")
print("="*80)

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
print("\n📂 Loading training data...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
print(f"✓ Loaded {len(train_df)} training samples")

# Load text content
def load_text(row):
    """Load text content from processed file."""
    genre = row['genre'].replace('/', '_')
    filename = f"{row['book_id']}.txt"
    path = PROCESSED_DATA_DIR / genre / filename
    try:
        if path.exists():
            return path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  ⚠️  Error loading {path}: {e}")
    return ""

print("\n📝 Loading text content from files...")
start_time = time.time()
X_train = train_df.apply(load_text, axis=1).values
load_time = time.time() - start_time

# Count empty texts
empty_count = sum(1 for text in X_train if not text)
if empty_count > 0:
    print(f"  ⚠️  Warning: {empty_count} texts are empty")
print(f"✓ Loaded {len(X_train)} texts in {load_time:.1f}s")

# Create TF-IDF vectorizer
print("\n" + "="*80)
print("CREATING TF-IDF VECTORIZER")
print("="*80)

print("\n⚙️  Vectorizer parameters:")
print("  - max_features: 5000 (top 5000 most frequent terms)")
print("  - ngram_range: (1, 2) (unigrams and bigrams)")
print("  - Default: lowercase=True, stop_words=None")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

print("\n🔄 Training vectorizer on training set...")
start_time = time.time()
vectorizer.fit(X_train)
fit_time = time.time() - start_time

print(f"✓ Vectorizer trained in {fit_time:.1f}s")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} terms")

# Test transformation
print("\n🧪 Testing transformation...")
X_train_transformed = vectorizer.transform(X_train[:10])
print(f"✓ Test transformation successful")
print(f"  Output shape: {X_train_transformed.shape}")
print(f"  Sparsity: {100 * (1 - X_train_transformed.nnz / (X_train_transformed.shape[0] * X_train_transformed.shape[1])):.2f}%")

# Save vectorizer
print("\n" + "="*80)
print("SAVING VECTORIZER")
print("="*80)

vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_path)

file_size = vectorizer_path.stat().st_size / 1024**2
print(f"\n✓ Vectorizer saved to: {vectorizer_path}")
print(f"  File size: {file_size:.2f} MB")

print("\n" + "="*80)
print("✅ VECTORIZER GENERATION COMPLETE!")
print("="*80)

print("\n📋 Next steps:")
print("  1. Cache vectorized data:")
print("     python3 scripts/cache_vectorized_data.py")
print("\n  2. Train models:")
print("     python3 scripts/train_ultra_fast.py")

print("\n" + "="*80)
