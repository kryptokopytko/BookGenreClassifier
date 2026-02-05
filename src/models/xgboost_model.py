from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR, RANDOM_STATE, XGBOOST_PARAMS, XGBOOST_EARLY_STOPPING_ROUNDS

logger = get_logger(__name__)

class XGBoostModel:

    def __init__(
        self,
        feature_columns: List[str] = None,
        use_config_params: bool = True
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.feature_columns = feature_columns

        if use_config_params:
            self.params = XGBOOST_PARAMS.copy()
            self.early_stopping_rounds = XGBOOST_EARLY_STOPPING_ROUNDS
        else:

            self.params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
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
        model_params['objective'] = 'multi:softprob'
        model_params['num_class'] = len(self.genre_labels)

        self.model = xgb.XGBClassifier(**model_params)

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
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )
            self.best_iteration = self.model.best_iteration
            logger.info(f"Early stopping at iteration {self.best_iteration}")
        else:
            self.model.fit(
                X_train_scaled,
                y_train_encoded,
                verbose=False
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

        self.model.save_model(str(output_dir / "xgboost_model.json"))
        joblib.dump(self.scaler, output_dir / "xgboost_scaler.pkl")
        joblib.dump(self.label_encoder, output_dir / "xgboost_label_encoder.pkl")
        joblib.dump(self.feature_columns, output_dir / "xgboost_feature_columns.pkl")
        joblib.dump(self.genre_labels, output_dir / "xgboost_labels.pkl")

        logger.info(f"Model saved to {output_dir}")

    def load(self, input_dir: Path = MODELS_DIR):

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(input_dir / "xgboost_model.json"))

        self.scaler = joblib.load(input_dir / "xgboost_scaler.pkl")
        self.label_encoder = joblib.load(input_dir / "xgboost_label_encoder.pkl")
        self.feature_columns = joblib.load(input_dir / "xgboost_feature_columns.pkl")
        self.genre_labels = joblib.load(input_dir / "xgboost_labels.pkl")

        logger.info(f"Model loaded from {input_dir}")

def train_xgboost_model(
    features_file: Path,
    train_ids: List[str],
    val_ids: List[str],
    save_dir: Path = MODELS_DIR,
    feature_columns: List[str] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> XGBoostModel:
    df = pd.read_csv(features_file)

    train_df = df[df['book_id'].isin(train_ids)]
    val_df = df[df['book_id'].isin(val_ids)]

    logger.info(f"Training set: {len(train_df)} books")
    logger.info(f"Validation set: {len(val_df)} books")

    model = XGBoostModel(
        feature_columns=feature_columns,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

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
) -> XGBoostModel:
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    train_ids = train_df['book_id'].tolist()
    val_ids = val_df['book_id'].tolist()

    return train_xgboost_model(features_file, train_ids, val_ids, save_dir)
