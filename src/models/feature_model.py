from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logger import get_logger
from ..utils.config import RANDOM_FOREST_PARAMS, MODELS_DIR

logger = get_logger(__name__)

class FeatureBasedModel:

    def __init__(self, feature_columns: List[str] = None):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        self.genre_labels = None

    def set_feature_columns(self, columns: List[str]):
        self.feature_columns = columns
        logger.info(f"Using {len(columns)} features")

    def prepare_data(
        self,
        df: pd.DataFrame,
        auto_select_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.feature_columns is None and auto_select_features:

            metadata_cols = ['book_id', 'title', 'author', 'genre', 'analysis_path']
            feature_cols = [col for col in df.columns
                          if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])]
            self.feature_columns = feature_cols
            logger.info(f"Auto-selected {len(feature_cols)} feature columns")

        X = df[self.feature_columns].values
        y = df['genre'].values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")

        return X, y

    def train(self, X_train: np.ndarray, y_train: np.ndarray):

        self.genre_labels = sorted(np.unique(y_train))
        logger.info(f"Training on {len(self.genre_labels)} genres")

        X_train_scaled = self.scaler.fit_transform(X_train)

        self.model.fit(X_train_scaled, y_train)

        train_acc = self.model.score(X_train_scaled, y_train)
        logger.info(f"Training accuracy: {train_acc:.4f}")

        if hasattr(self.model, 'oob_score_'):
            logger.info(f"OOB score: {self.model.oob_score_:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

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

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save(self, output_dir: Path = MODELS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, output_dir / "feature_model.pkl")
        joblib.dump(self.scaler, output_dir / "feature_scaler.pkl")
        joblib.dump(self.feature_columns, output_dir / "feature_columns.pkl")
        joblib.dump(self.genre_labels, output_dir / "feature_labels.pkl")

        logger.info(f"Model saved to {output_dir}")

    def load(self, input_dir: Path = MODELS_DIR):

        self.model = joblib.load(input_dir / "feature_model.pkl")
        self.scaler = joblib.load(input_dir / "feature_scaler.pkl")
        self.feature_columns = joblib.load(input_dir / "feature_columns.pkl")
        self.genre_labels = joblib.load(input_dir / "feature_labels.pkl")

        logger.info(f"Model loaded from {input_dir}")

def train_feature_model(
    features_file: Path,
    train_ids: List[str],
    val_ids: List[str],
    save_dir: Path = MODELS_DIR,
    feature_columns: List[str] = None
) -> FeatureBasedModel:
    df = pd.read_csv(features_file)

    train_df = df[df['book_id'].isin(train_ids)]
    val_df = df[df['book_id'].isin(val_ids)]

    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    model = FeatureBasedModel(feature_columns)

    X_train, y_train = model.prepare_data(train_df)
    X_val, y_val = model.prepare_data(val_df)

    model.train(X_train, y_train)

    val_metrics = model.evaluate(X_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nTop 10 Most Important Features:")
    importance_df = model.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    model.save(save_dir)

    return model

def train_from_split_files(
    features_file: Path,
    train_file: Path,
    val_file: Path,
    save_dir: Path = MODELS_DIR
) -> FeatureBasedModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    train_ids = train_df['book_id'].tolist()
    val_ids = val_df['book_id'].tolist()

    return train_feature_model(features_file, train_ids, val_ids, save_dir)
