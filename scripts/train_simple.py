import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")

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

print("Loading texts...")
X_train = train_df.apply(load_text, axis=1).values
X_val = val_df.apply(load_text, axis=1).values
X_test = test_df.apply(load_text, axis=1).values

y_train = train_df['genre'].values
y_val = val_df['genre'].values
y_test = test_df['genre'].values

print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

results = []

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
y_pred_train = lr.predict(X_train_tfidf)
y_pred_val = lr.predict(X_val_tfidf)
y_pred_test = lr.predict(X_test_tfidf)
results.append({
    'model': 'Logistic Regression',
    'train_acc': accuracy_score(y_train, y_pred_train),
    'val_acc': accuracy_score(y_val, y_pred_val),
    'test_acc': accuracy_score(y_test, y_pred_test)
})
print(f"  Train: {results[-1]['train_acc']:.4f}, Val: {results[-1]['val_acc']:.4f}, Test: {results[-1]['test_acc']:.4f}")
joblib.dump(lr, MODELS_DIR / 'logistic_regression.pkl')

print("\nTraining Linear SVM...")
svm = LinearSVC(max_iter=1000, random_state=42)
svm.fit(X_train_tfidf, y_train)
y_pred_train = svm.predict(X_train_tfidf)
y_pred_val = svm.predict(X_val_tfidf)
y_pred_test = svm.predict(X_test_tfidf)
results.append({
    'model': 'Linear SVM',
    'train_acc': accuracy_score(y_train, y_pred_train),
    'val_acc': accuracy_score(y_val, y_pred_val),
    'test_acc': accuracy_score(y_test, y_pred_test)
})
print(f"  Train: {results[-1]['train_acc']:.4f}, Val: {results[-1]['val_acc']:.4f}, Test: {results[-1]['test_acc']:.4f}")
joblib.dump(svm, MODELS_DIR / 'linear_svm.pkl')

print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_train = nb.predict(X_train_tfidf)
y_pred_val = nb.predict(X_val_tfidf)
y_pred_test = nb.predict(X_test_tfidf)
results.append({
    'model': 'Naive Bayes',
    'train_acc': accuracy_score(y_train, y_pred_train),
    'val_acc': accuracy_score(y_val, y_pred_val),
    'test_acc': accuracy_score(y_test, y_pred_test)
})
print(f"  Train: {results[-1]['train_acc']:.4f}, Val: {results[-1]['val_acc']:.4f}, Test: {results[-1]['test_acc']:.4f}")
joblib.dump(nb, MODELS_DIR / 'naive_bayes.pkl')

print("\nTraining Random Forest...")
features_df = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
features_df = features_df.drop_duplicates(subset=['book_id'], keep='first')

train_features = train_df.merge(features_df, on='book_id', how='left')
val_features = val_df.merge(features_df, on='book_id', how='left')
test_features = test_df.merge(features_df, on='book_id', how='left')

feature_cols = [c for c in features_df.columns if c not in ['book_id', 'title', 'author', 'genre']]
X_train_feat = train_features[feature_cols].fillna(0).values
X_val_feat = val_features[feature_cols].fillna(0).values
X_test_feat = test_features[feature_cols].fillna(0).values

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_feat, y_train)
y_pred_train = rf.predict(X_train_feat)
y_pred_val = rf.predict(X_val_feat)
y_pred_test = rf.predict(X_test_feat)
results.append({
    'model': 'Random Forest',
    'train_acc': accuracy_score(y_train, y_pred_train),
    'val_acc': accuracy_score(y_val, y_pred_val),
    'test_acc': accuracy_score(y_test, y_pred_test)
})
print(f"  Train: {results[-1]['train_acc']:.4f}, Val: {results[-1]['val_acc']:.4f}, Test: {results[-1]['test_acc']:.4f}")
joblib.dump(rf, MODELS_DIR / 'random_forest.pkl')

results_df = pd.DataFrame(results)
results_df.to_csv(MODELS_DIR.parent / 'results' / 'model_results.csv', index=False)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(results_df.to_string(index=False))
print(f"\nModels saved to: {MODELS_DIR}")
print(f"Results saved to: {MODELS_DIR.parent / 'results' / 'model_results.csv'}")
