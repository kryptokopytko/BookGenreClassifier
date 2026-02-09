import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print("="*60)
print("MODEL TESTING AND OPTIMIZATION")
print("="*60)
print(f"\nDataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

def load_text(row):
    genre = row['genre'].replace('/', '_')
    filename = f"{row['book_id']}.txt"
    path = PROCESSED_DATA_DIR / genre / filename
    try:
        if path.exists():
            return path.read_text(encoding='utf-8')
    except:
        pass
    return ""

print("\nLoading texts...")
X_train = train_df.apply(load_text, axis=1).values
X_val = val_df.apply(load_text, axis=1).values
X_test = test_df.apply(load_text, axis=1).values

y_train = train_df['genre'].values
y_val = val_df['genre'].values
y_test = test_df['genre'].values

print("\n" + "="*60)
print("TESTING EXISTING MODELS")
print("="*60)

vectorizer_path = MODELS_DIR / 'tfidf_vectorizer.pkl'
if vectorizer_path.exists():
    vectorizer = joblib.load(vectorizer_path)
    X_train_tfidf = vectorizer.transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Loaded existing vectorizer")
else:
    print("Creating new TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    joblib.dump(vectorizer, vectorizer_path)

for model_name in ['logistic_regression', 'linear_svm', 'naive_bayes', 'random_forest']:
    model_path = MODELS_DIR / f'{model_name}.pkl'
    if model_path.exists():
        print(f"\n{model_name.upper()}:")
        model = joblib.load(model_path)

        if 'forest' in model_name:
            features_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
            features_df = features_df.drop_duplicates(subset=['book_id'], keep='first')

            test_features = test_df.merge(features_df, on='book_id', how='left')
            feature_cols = [c for c in features_df.columns if c not in ['book_id', 'title', 'author', 'genre']]
            X_test_feat = test_features[feature_cols].fillna(0).values
            y_pred = model.predict(X_test_feat)
        else:
            y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        print(f"  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

print("\n" + "="*60)
print("SVM HYPERPARAMETER OPTIMIZATION")
print("="*60)

print("\nTesting different C values...")
c_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
best_c = 1.0
best_val_acc = 0

results = []
for c in c_values:
    svm = LinearSVC(C=c, max_iter=2000, random_state=42)
    svm.fit(X_train_tfidf, y_train)

    val_acc = accuracy_score(y_val, svm.predict(X_val_tfidf))
    test_acc = accuracy_score(y_test, svm.predict(X_test_tfidf))

    results.append({'C': c, 'val_acc': val_acc, 'test_acc': test_acc})
    print(f"  C={c:5.2f}: Val {val_acc:.4f}, Test {test_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_c = c

print(f"\nBest C: {best_c} (Val Acc: {best_val_acc:.4f})")

print("\nTraining final optimized SVM...")
best_svm = LinearSVC(C=best_c, max_iter=2000, random_state=42)
best_svm.fit(X_train_tfidf, y_train)

y_pred_test = best_svm.predict(X_test_tfidf)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n{'='*60}")
print(f"FINAL RESULTS - Optimized Linear SVM (C={best_c})")
print(f"{'='*60}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

print("\nPer-Genre Results:")
print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test)
genres = sorted(set(y_test))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres)
plt.title(f'Confusion Matrix - Optimized SVM (C={best_c})')
plt.ylabel('True Genre')
plt.xlabel('Predicted Genre')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'optimized_svm_confusion_matrix.png', dpi=150)
print(f"\nConfusion matrix saved to: {RESULTS_DIR / 'optimized_svm_confusion_matrix.png'}")

joblib.dump(best_svm, MODELS_DIR / 'linear_svm_optimized.pkl')
print(f"Optimized model saved to: {MODELS_DIR / 'linear_svm_optimized.pkl'}")

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_DIR / 'svm_optimization_results.csv', index=False)
print(f"Optimization results saved to: {RESULTS_DIR / 'svm_optimization_results.csv'}")

print("\nTesting and optimization complete!")
