import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("COMPREHENSIVE MODEL TESTING")
print("="*80)

print("\nLoading data...")
train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print(f"   Train: {len(train_df)} samples")
print(f"   Val:   {len(val_df)} samples")
print(f"   Test:  {len(test_df)} samples")

def load_text(row):
    """Load text from processed file."""
    genre = row['genre'].replace('/', '_')
    filename = f"{row['book_id']}.txt"
    path = PROCESSED_DATA_DIR / genre / filename
    try:
        if path.exists():
            return path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"   Error loading {path}: {e}")
    return ""

print("\nLoading texts...")
X_test_texts = test_df.apply(load_text, axis=1).values
y_test = test_df['genre'].values

print(f"   Loaded {len(X_test_texts)} test texts")

print("\nLoading features...")
features_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
features_df = features_df.drop_duplicates(subset=['book_id'], keep='first')
test_features = test_df.merge(features_df, on='book_id', how='left')
feature_cols = [c for c in features_df.columns if c not in ['book_id', 'title', 'author', 'genre']]
X_test_feat = test_features[feature_cols].fillna(0).values

print(f"   Loaded {len(feature_cols)} features")

tfidf_vectorizer = None
if (MODELS_DIR / 'tfidf_vectorizer.pkl').exists():
    print(f"\nLoading shared TF-IDF vectorizer...")
    tfidf_vectorizer = joblib.load(MODELS_DIR / 'tfidf_vectorizer.pkl')
    X_test_tfidf = tfidf_vectorizer.transform(X_test_texts)
    print(f"   Transformed to TF-IDF: {X_test_tfidf.shape}")

models_to_test = [
    ('Logistic Regression', 'logistic_regression.pkl', 'tfidf', None),
    ('Linear SVM', 'linear_svm.pkl', 'tfidf', None),
    ('Naive Bayes', 'naive_bayes.pkl', 'tfidf', None),

    ('Random Forest', 'random_forest.pkl', 'features', None),
]

if (MODELS_DIR / 'linear_svm_optimized.pkl').exists():
    models_to_test.append(('Linear SVM (Optimized)', 'linear_svm_optimized.pkl', 'tfidf', None))

if (MODELS_DIR / 'knn_model.pkl').exists():
    models_to_test.append(('KNN', 'knn_model.pkl', 'text', 'knn'))
if (MODELS_DIR / 'ridge_model.pkl').exists():
    models_to_test.append(('Ridge Classifier', 'ridge_model.pkl', 'text', 'ridge'))
if (MODELS_DIR / 'tfidf_svm_model.pkl').exists():
    models_to_test.append(('TF-IDF SVM', 'tfidf_svm_model.pkl', 'text', 'tfidf'))

results = []

print("\n" + "="*80)
print("TESTING MODELS")
print("="*80)

for model_name, model_file, input_type, model_type in models_to_test:
    model_path = MODELS_DIR / model_file

    if not model_path.exists():
        print(f"\n❌ {model_name}: Model file not found - {model_file}")
        continue

    print(f"\n🔍 Testing {model_name}...")
    print(f"   Model file: {model_file}")

    try:
        if model_type == 'knn':
            from src.models.knn_model import KNNModel
            model = KNNModel()
            model.load(MODELS_DIR)
            y_pred = model.predict(list(X_test_texts))
        elif model_type == 'ridge':
            from src.models.ridge_model import RidgeModel
            model = RidgeModel.load(MODELS_DIR)
            y_pred = model.predict(list(X_test_texts))
        elif model_type == 'tfidf':
            from src.models.tfidf_model import TFIDFModel
            model = TFIDFModel(algorithm='svm')
            model.load(MODELS_DIR)
            y_pred = model.predict(list(X_test_texts))
        else:
            model = joblib.load(model_path)

            if input_type == 'tfidf':
                if tfidf_vectorizer is None:
                    print(f"   Warning: TF-IDF vectorizer not found")
                    continue
                y_pred = model.predict(X_test_tfidf)
            elif input_type == 'text':
                vectorizer_file = model_file.replace('_model.pkl', '_vectorizer.pkl').replace('.pkl', '_vectorizer.pkl')
                if not vectorizer_file.endswith('_vectorizer.pkl'):
                    vectorizer_file = model_file.replace('.pkl', '') + '_vectorizer.pkl'

                vectorizer_path = MODELS_DIR / vectorizer_file
                if not vectorizer_path.exists():
                    vectorizer_path = MODELS_DIR / 'vectorizer.pkl'

                if vectorizer_path.exists():
                    vectorizer = joblib.load(vectorizer_path)
                    X_test_transformed = vectorizer.transform(X_test_texts)
                    y_pred = model.predict(X_test_transformed)
                else:
                    print(f"   Warning: Vectorizer not found for {model_name}")
                    continue
            else:
                y_pred = model.predict(X_test_feat)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0, labels=sorted(np.unique(y_test))
        )

        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")

        results.append({
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        })

        cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test)))

        plt.figure(figsize=(12, 10))
        genres = sorted(np.unique(y_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=genres, yticklabels=genres,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.2%}',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_file = RESULTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved confusion matrix to: {cm_file}")

        report = classification_report(y_test, y_pred, zero_division=0)
        report_file = RESULTS_DIR / f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(report)
        print(f"   Saved classification report to: {report_file}")

    except Exception as e:
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

if results:
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    results_df = pd.DataFrame([{
        'Model': r['model'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1 Score': r['f1']
    } for r in results])

    results_df = results_df.sort_values('F1 Score', ascending=False)

    print("\n" + results_df.to_string(index=False))

    results_file = RESULTS_DIR / "all_models_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    plt.figure(figsize=(14, 6))

    x = np.arange(len(results_df))
    width = 0.2

    plt.bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar(x - 0.5*width, results_df['Precision'], width, label='Precision', alpha=0.8)
    plt.bar(x + 0.5*width, results_df['Recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + 1.5*width, results_df['F1 Score'], width, label='F1 Score', alpha=0.8)

    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, results_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    comparison_file = RESULTS_DIR / "model_comparison.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {comparison_file}")

    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
else:
    print("\nNo models were successfully tested.")
