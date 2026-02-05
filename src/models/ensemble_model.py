import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

class EnsembleModel:

    def __init__(self, model_names: List[str], voting: str = 'soft'):
        self.model_names = model_names
        self.voting = voting
        self.models = {}
        self.vectorizers = {}
        self.ensemble = None

    def load_models(self):
        for model_name in self.model_names:
            model_path = MODELS_DIR / f"{model_name}_model.pkl"
            vectorizer_path = MODELS_DIR / f"{model_name}_vectorizer.pkl"

            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"Loaded model: {model_name}")

                if vectorizer_path.exists():
                    with open(vectorizer_path, 'rb') as f:
                        self.vectorizers[model_name] = pickle.load(f)
                    logger.info(f"Loaded vectorizer for: {model_name}")
            else:
                logger.warning(f"Model not found: {model_name}")

    def predict(self, texts: List[str]) -> np.ndarray:
        predictions = []
        for model_name in self.models:
            model = self.models[model_name]

            if model_name in self.vectorizers:
                vectorizer = self.vectorizers[model_name]
                X = vectorizer.transform(texts)
            else:

                X = texts

            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if self.voting == 'hard':

            from scipy.stats import mode
            final_pred = mode(predictions, axis=0)[0].flatten()
        else:

            proba_predictions = []
            for model_name in self.models:
                model = self.models[model_name]
                if model_name in self.vectorizers:
                    vectorizer = self.vectorizers[model_name]
                    X = vectorizer.transform(texts)
                else:
                    X = texts

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    proba_predictions.append(proba)

            if proba_predictions:
                avg_proba = np.mean(proba_predictions, axis=0)
                final_pred = self.models[self.model_names[0]].classes_[np.argmax(avg_proba, axis=1)]
            else:

                from scipy.stats import mode
                final_pred = mode(predictions, axis=0)[0].flatten()

        return final_pred

    def evaluate(self, texts: List[str], y_true: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(texts)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def train_ensemble_model(model_names: List[str] = None, voting: str = 'soft'):
    if model_names is None:
        model_names = ['tfidf_logistic', 'tfidf_random_forest', 'naive_bayes']

    logger.info(f"Creating ensemble with models: {model_names}")
    logger.info(f"Voting strategy: {voting}")

    test_file = PROCESSED_DATA_DIR / "test.csv"
    test_df = pd.read_csv(test_file)

    logger.info(f"Loading texts for {len(test_df)} books")
    texts = []
    for _, row in test_df.iterrows():
        file_path = Path(row['processed_path'])
        if not file_path.is_absolute():
            file_path = PROCESSED_DATA_DIR.parent / file_path
        try:
            text = file_path.read_text(encoding='utf-8')
            texts.append(text)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            texts.append("")

    y_true = test_df['genre'].values

    ensemble = EnsembleModel(model_names, voting)
    ensemble.load_models()

    logger.info("Evaluating ensemble...")
    metrics = ensemble.evaluate(texts, y_true)

    logger.info(f"\nEnsemble Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1: {metrics['f1']:.4f}")

    return ensemble, metrics
