from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logger import get_logger
from ..utils.config import (
    LOGISTIC_REGRESSION_PARAMS, PROCESSED_DATA_DIR, MODELS_DIR
)
from ..features.vocabulary_analyzer import VocabularyAnalyzer

logger = get_logger(__name__)

class BaselineKeywordModel:

    def __init__(self, keywords_dict: Dict[str, List[str]] = None):
        self.analyzer = VocabularyAnalyzer()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        self.genre_labels = None

    def set_keywords(self, keywords_dict: Dict[str, List[str]]):
        self.keywords_dict = keywords_dict
        logger.info(f"Keywords set for {len(keywords_dict)} genres")

    def extract_features(self, text: str) -> np.ndarray:
        if not self.keywords_dict:
            raise ValueError("Keywords dictionary not set")

        features = self.analyzer.create_keyword_features(
            text,
            self.keywords_dict,
            normalize=True
        )

        genre_order = sorted(self.keywords_dict.keys())
        feature_values = [
            features.get(f'keyword_{genre.replace("/", "_")}', 0)
            for genre in genre_order
        ]

        return np.array(feature_values)

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'processed_path'
    ) -> Tuple[np.ndarray, np.ndarray]:

        X = []
        y = []

        for idx, row in df.iterrows():
            try:

                if text_column == 'processed_path':
                    text_path = Path(row[text_column])
                    if not text_path.is_absolute():
                        text_path = PROCESSED_DATA_DIR.parent / text_path
                    text = text_path.read_text(encoding='utf-8')
                else:
                    text = row[text_column]

                features = self.extract_features(text)

                X.append(features)
                y.append(row['genre'])

            except Exception as e:
                logger.error(f"Error processing book {row.get('book_id', idx)}: {e}")
                continue

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):

        self.genre_labels = sorted(np.unique(y_train))

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, y_train)

        train_acc = self.model.score(X_train_scaled, y_train)
        logger.info(f"Training accuracy: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:

        y_pred = self.predict(X_test)

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

    def get_feature_importance(self) -> pd.DataFrame:
        genre_order = sorted(self.keywords_dict.keys())
        feature_names = [f'keyword_{genre.replace("/", "_")}' for genre in genre_order]

        coefficients = self.model.coef_

        importance_data = []

        for i, genre in enumerate(self.genre_labels):
            for j, feature in enumerate(feature_names):
                importance_data.append({
                    'genre': genre,
                    'feature': feature,
                    'coefficient': coefficients[i, j]
                })

        df = pd.DataFrame(importance_data)
        df = df.sort_values('coefficient', key=abs, ascending=False)

        return df

    def save(self, output_dir: Path = MODELS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / "baseline_model.pkl")
        joblib.dump(self.scaler, output_dir / "baseline_scaler.pkl")
        joblib.dump(self.keywords_dict, output_dir / "baseline_keywords.pkl")
        joblib.dump(self.genre_labels, output_dir / "baseline_labels.pkl")

        logger.info(f"Model saved to {output_dir}")

    def load(self, input_dir: Path = MODELS_DIR):

        self.model = joblib.load(input_dir / "baseline_model.pkl")
        self.scaler = joblib.load(input_dir / "baseline_scaler.pkl")
        self.keywords_dict = joblib.load(input_dir / "baseline_keywords.pkl")
        self.genre_labels = joblib.load(input_dir / "baseline_labels.pkl")

        logger.info(f"Model loaded from {input_dir}")

def train_baseline_model(
    train_file: Path,
    val_file: Path,
    keywords_dict: Dict[str, List[str]],
    save_dir: Path = MODELS_DIR
) -> BaselineKeywordModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    model = BaselineKeywordModel(keywords_dict)

    X_train, y_train = model.prepare_data(train_df)
    X_val, y_val = model.prepare_data(val_df)

    model.train(X_train, y_train)

    val_metrics = model.evaluate(X_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    model.save(save_dir)

    return model
