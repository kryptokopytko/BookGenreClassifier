from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.corpus import stopwords

from ..utils.logger import get_logger
from ..utils.config import (
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
    PROCESSED_DATA_DIR, MODELS_DIR, KNN_PARAMS
)

logger = get_logger(__name__)

class KNNModel:

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        remove_stopwords: bool = True,
        use_config_params: bool = True
    ):
        self.max_features = max_features
        self.remove_stopwords = remove_stopwords

        if use_config_params:
            self.n_neighbors = KNN_PARAMS.get('n_neighbors', 20)
            self.metric = KNN_PARAMS.get('metric', 'cosine')
            self.weights = KNN_PARAMS.get('weights', 'distance')
        else:

            self.n_neighbors = 5
            self.metric = 'cosine'
            self.weights = 'distance'

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

        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weights=self.weights,
            n_jobs=-1
        )

        self.label_encoder = LabelEncoder()
        self.genre_labels = None
        self.train_accuracy = None

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
        logger.info(f"Max features: {self.max_features}, Metric: {self.metric}, Weights: {self.weights}")

        self.genre_labels = sorted(np.unique(y_train))

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        logger.info("Vectorizing texts...")
        X_train = self.vectorizer.fit_transform(texts_train)
        logger.info(f"TF-IDF shape: {X_train.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        logger.info("Training KNN classifier...")
        self.model.fit(X_train, y_train_encoded)

        train_acc = self.model.score(X_train, y_train_encoded)
        self.train_accuracy = train_acc
        logger.info(f"Training accuracy: {train_acc:.4f}")

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def get_neighbors(self, text: str, n_neighbors: int = None) -> List[Tuple[int, float]]:
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        X = self.vectorizer.transform([text])
        distances, indices = self.model.kneighbors(X, n_neighbors=n_neighbors)

        return list(zip(indices[0], distances[0]))

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

    def save(self, output_dir: Path = MODELS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / "knn_model.pkl")
        joblib.dump(self.vectorizer, output_dir / "knn_vectorizer.pkl")
        joblib.dump(self.label_encoder, output_dir / "knn_label_encoder.pkl")
        joblib.dump(self.genre_labels, output_dir / "knn_labels.pkl")

        config = {
            'n_neighbors': self.n_neighbors,
            'max_features': self.max_features,
            'metric': self.metric,
            'weights': self.weights,
            'remove_stopwords': self.remove_stopwords
        }
        joblib.dump(config, output_dir / "knn_config.pkl")

        logger.info(f"Model saved to {output_dir}")

    def load(self, input_dir: Path = MODELS_DIR):

        self.model = joblib.load(input_dir / "knn_model.pkl")
        self.vectorizer = joblib.load(input_dir / "knn_vectorizer.pkl")
        self.label_encoder = joblib.load(input_dir / "knn_label_encoder.pkl")
        self.genre_labels = joblib.load(input_dir / "knn_labels.pkl")

        config = joblib.load(input_dir / "knn_config.pkl")
        self.n_neighbors = config['n_neighbors']
        self.max_features = config['max_features']
        self.metric = config['metric']
        self.weights = config['weights']
        self.remove_stopwords = config['remove_stopwords']

        logger.info(f"Model loaded from {input_dir}")

def train_knn_model(
    train_file: Path,
    val_file: Path,
    n_neighbors: int = None,
    max_features: int = TFIDF_MAX_FEATURES,
    metric: str = None,
    save_dir: Path = MODELS_DIR
) -> KNNModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    if n_neighbors is None and metric is None:

        model = KNNModel(
            max_features=max_features,
            use_config_params=True
        )
    else:

        model = KNNModel(
            max_features=max_features,
            use_config_params=False
        )
        if n_neighbors is not None:
            model.n_neighbors = n_neighbors
            model.model.n_neighbors = n_neighbors
        if metric is not None:
            model.metric = metric
            model.model.metric = metric

    texts_train, y_train = model.prepare_data(train_df)
    texts_val, y_val = model.prepare_data(val_df)

    model.train(texts_train, y_train)

    val_metrics = model.evaluate(texts_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    model.save(save_dir)

    return model
