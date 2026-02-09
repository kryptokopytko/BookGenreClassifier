from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logger import get_logger
from ..utils.config import (
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
    PROCESSED_DATA_DIR, MODELS_DIR, NEAREST_CENTROID_PARAMS
)

logger = get_logger(__name__)

class NearestCentroidModel:

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        use_lemmas: bool = False,
        remove_stopwords: bool = True,
        use_config_params: bool = True
    ):
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

        if use_config_params:
            self.model = NearestCentroid(**NEAREST_CENTROID_PARAMS)
        else:
            self.model = NearestCentroid(metric='cosine')

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
        logger.info(f"Max features: {self.max_features}, Metric: {self.model.metric}")

        self.genre_labels = sorted(np.unique(y_train))
        logger.info(f"Computing centroids for {len(self.genre_labels)} genres")

        logger.info("Vectorizing texts...")
        X_train = self.vectorizer.fit_transform(texts_train)
        logger.info(f"TF-IDF shape: {X_train.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        logger.info("Computing genre centroids...")
        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        self.train_accuracy = train_acc
        logger.info(f"Training accuracy: {train_acc:.4f}")

        logger.info(f"Centroids computed: shape={self.model.centroids_.shape}")

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.model.predict(texts)

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

        return metrics

    def get_centroid_top_features(self, n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        feature_names = self.vectorizer.get_feature_names_out()
        centroids = self.model.centroids_

        top_features = {}
        for i, genre in enumerate(self.genre_labels):

            centroid = centroids[i]
            top_indices = np.argsort(centroid)[-n:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            top_scores = [centroid[idx] for idx in top_indices]
            top_features[genre] = list(zip(top_words, top_scores))

        return top_features

    def save(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file = save_dir / "nearest_centroid_model.pkl"
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'genre_labels': self.genre_labels,
            'max_features': self.max_features,
            'train_accuracy': self.train_accuracy
        }, model_file)

        logger.info(f"Model saved to {model_file}")

        try:
            top_features = self.get_centroid_top_features(n=20)
            centroid_features_file = save_dir / "nearest_centroid_top_features.txt"

            with open(centroid_features_file, 'w') as f:
                f.write("Top Features per Genre Centroid\n")
                f.write("="*60 + "\n\n")

                for genre, features in top_features.items():
                    f.write(f"{genre}:\n")
                    for word, score in features:
                        f.write(f"  {word}: {score:.4f}\n")
                    f.write("\n")

            logger.info(f"Centroid features saved to {centroid_features_file}")
        except Exception as e:
            logger.warning(f"Could not save centroid features: {e}")

    @classmethod
    def load(cls, save_dir: Path):
        model_file = save_dir / "nearest_centroid_model.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        data = joblib.load(model_file)

        model = cls(max_features=data['max_features'], use_config_params=False)
        model.model = data['model']
        model.vectorizer = data['vectorizer']
        model.genre_labels = data['genre_labels']
        model.train_accuracy = data.get('train_accuracy')

        logger.info(f"Model loaded from {model_file}")
        return model

def train_nearest_centroid_model(
    train_file: Path,
    val_file: Path,
    max_features: int = TFIDF_MAX_FEATURES,
    save_dir: Path = MODELS_DIR
) -> NearestCentroidModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    model = NearestCentroidModel(max_features=max_features, use_config_params=True)

    texts_train, y_train = model.prepare_data(train_df)
    texts_val, y_val = model.prepare_data(val_df)

    model.train(texts_train, y_train)

    val_metrics = model.evaluate(texts_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nTop words per genre centroid:")
    top_features = model.get_centroid_top_features(n=10)
    for genre, features in top_features.items():
        words = [word for word, score in features]
        logger.info(f"  {genre}: {', '.join(words[:10])}")

    model.save(save_dir)

    return model
