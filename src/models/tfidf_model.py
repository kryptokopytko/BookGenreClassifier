from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords

from ..utils.logger import get_logger
from ..utils.config import (
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS, SVM_PARAMS,
    PROCESSED_DATA_DIR, MODELS_DIR
)

logger = get_logger(__name__)

class TFIDFModel:

    def __init__(
        self,
        algorithm: str = 'svm',
        max_features: int = TFIDF_MAX_FEATURES,
        use_lemmas: bool = False,
        remove_stopwords: bool = True
    ):
        self.algorithm = algorithm
        self.max_features = max_features
        self.use_lemmas = use_lemmas
        self.remove_stopwords = remove_stopwords

        try:
            self.stopwords = set(stopwords.words('english')) if remove_stopwords else None
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english')) if remove_stopwords else None

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words=list(self.stopwords) if self.stopwords else None,
            lowercase=True
        )

        self.model = self._create_classifier(algorithm)
        self.genre_labels = None

    def _create_classifier(self, algorithm: str):
        if algorithm == 'logistic':
            return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        elif algorithm == 'random_forest':
            return RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        elif algorithm == 'svm':

            svm_params = {k: v for k, v in SVM_PARAMS.items() if k != 'kernel'}
            return LinearSVC(**svm_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def load_texts(self, df: pd.DataFrame, text_column: str = 'processed_path') -> List[str]:
        texts = []
        for idx, row in df.iterrows():
            try:

                if text_column == 'processed_path':
                    text_path = Path(row[text_column])
                    if not text_path.is_absolute():
                        text_path = PROCESSED_DATA_DIR.parent / text_path
                    text = text_path.read_text(encoding='utf-8')
                else:
                    text = row[text_column]

                texts.append(text)

            except Exception as e:
                logger.error(f"Error loading text for book {row.get('book_id', idx)}: {e}")
                texts.append("")

        return texts

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'processed_path'
    ) -> Tuple[List[str], np.ndarray]:

        texts = self.load_texts(df, text_column)
        labels = df['genre'].values

        logger.info(f"Loaded {len(texts)} texts")

        return texts, labels

    def train(self, texts_train: List[str], y_train: np.ndarray):
        logger.info(f"Max features: {self.max_features}, Ngrams: {TFIDF_NGRAM_RANGE}")

        self.genre_labels = sorted(np.unique(y_train))

        logger.info("Vectorizing texts...")
        X_train = self.vectorizer.fit_transform(texts_train)
        logger.info(f"TF-IDF shape: {X_train.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        logger.info("Training classifier...")
        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_acc:.4f}")

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):

            scores = self.model.decision_function(X)

            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
        else:
            raise ValueError(f"Algorithm {self.algorithm} does not support probability prediction")

    def evaluate(self, texts_test: List[str], y_test: np.ndarray) -> Dict[str, float]:

        y_pred = self.predict(texts_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")

        return metrics

    def get_top_features_per_genre(self, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        feature_names = self.vectorizer.get_feature_names_out()
        top_features = {}

        if self.algorithm == 'logistic':

            for i, genre in enumerate(self.genre_labels):
                coefficients = self.model.coef_[i]
                top_indices = coefficients.argsort()[-n:][::-1]
                top_features[genre] = [
                    (feature_names[idx], coefficients[idx])
                    for idx in top_indices
                ]

        elif self.algorithm == 'random_forest':

            importances = self.model.feature_importances_
            top_indices = importances.argsort()[-n:][::-1]
            top_features['all'] = [
                (feature_names[idx], importances[idx])
                for idx in top_indices
            ]

        elif self.algorithm == 'svm':

            for i, genre in enumerate(self.genre_labels):
                coefficients = self.model.coef_[i]
                top_indices = coefficients.argsort()[-n:][::-1]
                top_features[genre] = [
                    (feature_names[idx], coefficients[idx])
                    for idx in top_indices
                ]

        return top_features

    def save(self, output_dir: Path = MODELS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"tfidf_{self.algorithm}"

        joblib.dump(self.model, output_dir / f"{prefix}_model.pkl")
        joblib.dump(self.vectorizer, output_dir / f"{prefix}_vectorizer.pkl")
        joblib.dump(self.genre_labels, output_dir / f"{prefix}_labels.pkl")

        config = {
            'algorithm': self.algorithm,
            'max_features': self.max_features,
            'use_lemmas': self.use_lemmas,
            'remove_stopwords': self.remove_stopwords
        }
        joblib.dump(config, output_dir / f"{prefix}_config.pkl")

        logger.info(f"Model saved to {output_dir}")

    def load(self, input_dir: Path = MODELS_DIR):
        prefix = f"tfidf_{self.algorithm}"

        self.model = joblib.load(input_dir / f"{prefix}_model.pkl")
        self.vectorizer = joblib.load(input_dir / f"{prefix}_vectorizer.pkl")
        self.genre_labels = joblib.load(input_dir / f"{prefix}_labels.pkl")

        config = joblib.load(input_dir / f"{prefix}_config.pkl")
        self.max_features = config['max_features']
        self.use_lemmas = config['use_lemmas']
        self.remove_stopwords = config['remove_stopwords']

        logger.info(f"Model loaded from {input_dir}")

def train_tfidf_model(
    train_file: Path,
    val_file: Path,
    algorithm: str = 'svm',
    max_features: int = TFIDF_MAX_FEATURES,
    save_dir: Path = MODELS_DIR
) -> TFIDFModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    model = TFIDFModel(algorithm=algorithm, max_features=max_features)

    texts_train, y_train = model.prepare_data(train_df)
    texts_val, y_val = model.prepare_data(val_df)

    model.train(texts_train, y_train)

    val_metrics = model.evaluate(texts_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nTop features per genre:")
    top_features = model.get_top_features_per_genre(n=10)
    for genre, features in top_features.items():
        logger.info(f"\n{genre}:")
        for feature, score in features[:5]:
            logger.info(f"  {feature}: {score:.4f}")

    model.save(save_dir)

    return model
