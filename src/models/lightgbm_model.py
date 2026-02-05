from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR, RANDOM_STATE, LIGHTGBM_PARAMS, LIGHTGBM_EARLY_STOPPING_ROUNDS

logger = get_logger(__name__)

class LightGBMModel:

    def __init__(
        self,
        feature_columns: List[str] = None,
        use_config_params: bool = True
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        self.feature_columns = feature_columns

        if use_config_params:
            self.params = LIGHTGBM_PARAMS.copy()
            self.early_stopping_rounds = LIGHTGBM_EARLY_STOPPING_ROUNDS
        else:

            self.params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'min_child_samples': 20,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
            self.early_stopping_rounds = None

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.genre_labels = None
        self.train_accuracy = None
        self.best_iteration = None

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

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        logger.info(f"Parameters: {self.params}")

        self.genre_labels = sorted(np.unique(y_train))
        logger.info(f"Training on {len(self.genre_labels)} genres")

        y_train_encoded = self.label_encoder.fit_transform(y_train)

        X_train_scaled = self.scaler.fit_transform(X_train)

        model_params = self.params.copy()
        model_params['objective'] = 'multiclass'
        model_params['num_class'] = len(self.genre_labels)

        self.model = lgb.LGBMClassifier(**model_params)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val_encoded)]
            logger.info(f"Using validation set with {len(X_val)} samples for early stopping")

        if eval_set and self.early_stopping_rounds:
            self.model.fit(
                X_train_scaled,
                y_train_encoded,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
            self.best_iteration = self.model.best_iteration_
            logger.info(f"Early stopping at iteration {self.best_iteration}")
        else:
            self.model.fit(
                X_train_scaled,
                y_train_encoded
            )

        train_acc = self.model.score(X_train_scaled, y_train_encoded)
        self.train_accuracy = train_acc
        logger.info(f"Training accuracy: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)

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

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet")

        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance

    def save(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file = save_dir / "lightgbm_model.pkl"
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'genre_labels': self.genre_labels,
            'params': self.params,
            'train_accuracy': self.train_accuracy,
            'best_iteration': self.best_iteration
        }, model_file)

        logger.info(f"Model saved to {model_file}")

    @classmethod
    def load(cls, save_dir: Path):
        model_file = save_dir / "lightgbm_model.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        data = joblib.load(model_file)

        model = cls(feature_columns=data['feature_columns'], use_config_params=False)
        model.model = data['model']
        model.scaler = data['scaler']
        model.label_encoder = data['label_encoder']
        model.genre_labels = data['genre_labels']
        model.params = data.get('params', {})
        model.train_accuracy = data.get('train_accuracy')
        model.best_iteration = data.get('best_iteration')

        logger.info(f"Model loaded from {model_file}")
        return model

def train_from_split_files(
    features_file: Path,
    train_file: Path,
    val_file: Path,
    save_dir: Path = MODELS_DIR
) -> LightGBMModel:
    features_df = pd.read_csv(features_file)
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    logger.info(f"Features: {len(features_df)} books")
    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    train_features = features_df[features_df['book_id'].isin(train_df['book_id'])]
    val_features = features_df[features_df['book_id'].isin(val_df['book_id'])]

    logger.info(f"Train features matched: {len(train_features)} books")
    logger.info(f"Val features matched: {len(val_features)} books")

    model = LightGBMModel(use_config_params=True)

    X_train, y_train = model.prepare_data(train_features)
    X_val, y_val = model.prepare_data(val_features)

    model.train(X_train, y_train, X_val, y_val)

    val_metrics = model.evaluate(X_val, y_val)

    logger.info("\nValidation Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    model.save(save_dir)

    return model
