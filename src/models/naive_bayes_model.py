from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords

from ..utils.logger import get_logger
from ..utils.config import (
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
    PROCESSED_DATA_DIR, MODELS_DIR
)

logger = get_logger(__name__)

class NaiveBayesModel:

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        alpha: float = 1.0,
        remove_stopwords: bool = True
    ):
        self.max_features = max_features
        self.alpha = alpha
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

        self.model = MultinomialNB(alpha=alpha)
        self.label_encoder = LabelEncoder()
        self.genre_labels = None

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
        logger.info(f"Alpha (smoothing): {self.alpha}")

        self.genre_labels = sorted(np.unique(y_train))

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        logger.info("Vectorizing texts...")
        X_train = self.vectorizer.fit_transform(texts_train)
        logger.info(f"TF-IDF shape: {X_train.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        logger.info("Training classifier...")
        self.model.fit(X_train, y_train_encoded)

        train_acc = self.model.score(X_train, y_train_encoded)
        logger.info(f"Training accuracy: {train_acc:.4f}")

        logger.info("\nClass priors (log probabilities):")
        for i, genre in enumerate(self.label_encoder.classes_):
            logger.info(f"  {genre}: {np.exp(self.model.class_log_prior_[i]):.4f}")

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict_log_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_log_proba(X)

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

        for i, genre in enumerate(self.label_encoder.classes_):

            log_probs = self.model.feature_log_prob_[i]

            top_indices = log_probs.argsort()[-n:][::-1]

            top_features[genre] = [
                (feature_names[idx], log_probs[idx])
                for idx in top_indices
            ]

        return top_features

    def save(self, output_dir: Path = MODELS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / "naive_bayes_model.pkl")
        joblib.dump(self.vectorizer, output_dir / "naive_bayes_vectorizer.pkl")
        joblib.dump(self.label_encoder, output_dir / "naive_bayes_label_encoder.pkl")
        joblib.dump(self.genre_labels, output_dir / "naive_bayes_labels.pkl")

        config = {
            'max_features': self.max_features,
            'alpha': self.alpha,
            'remove_stopwords': self.remove_stopwords
        }
        joblib.dump(config, output_dir / "naive_bayes_config.pkl")

        logger.info(f"Model saved to {output_dir}")

    def load(self, input_dir: Path = MODELS_DIR):

        self.model = joblib.load(input_dir / "naive_bayes_model.pkl")
        self.vectorizer = joblib.load(input_dir / "naive_bayes_vectorizer.pkl")
        self.label_encoder = joblib.load(input_dir / "naive_bayes_label_encoder.pkl")
        self.genre_labels = joblib.load(input_dir / "naive_bayes_labels.pkl")

        config = joblib.load(input_dir / "naive_bayes_config.pkl")
        self.max_features = config['max_features']
        self.alpha = config['alpha']
        self.remove_stopwords = config['remove_stopwords']

        logger.info(f"Model loaded from {input_dir}")

def train_naive_bayes_model(
    train_file: Path,
    val_file: Path,
    max_features: int = TFIDF_MAX_FEATURES,
    alpha: float = 1.0,
    save_dir: Path = MODELS_DIR
) -> NaiveBayesModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    model = NaiveBayesModel(max_features=max_features, alpha=alpha)

    texts_train, y_train = model.prepare_data(train_df)
    texts_val, y_val = model.prepare_data(val_df)

    model.train(texts_train, y_train)

    val_metrics = model.evaluate(texts_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nTop 10 words per genre:")
    top_features = model.get_top_features_per_genre(n=10)
    for genre, features in top_features.items():
        words = [w for w, p in features[:10]]
        logger.info(f"{genre}: {', '.join(words)}")

    model.save(save_dir)

    return model
