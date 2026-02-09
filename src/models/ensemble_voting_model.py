from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR, PROCESSED_DATA_DIR

logger = get_logger(__name__)

class EnsembleVotingModel:

    def __init__(
        self,
        models: List[Tuple[str, any]],
        voting: str = 'soft',
        weights: List[float] = None
    ):
        self.voting = voting
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.genre_labels = None
        self.train_accuracy = None

        if abs(sum(self.weights) - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights)}")

        if models:
            _, first_model = models[0]
            if hasattr(first_model, 'genre_labels'):
                self.genre_labels = first_model.genre_labels

    def predict(self, data: any) -> np.ndarray:

        all_probas = []

        for (name, model), weight in zip(self.models, self.weights):
            try:

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(data)
                elif hasattr(model, 'decision_function'):

                    scores = model.decision_function(data)

                    proba = self._softmax(scores)
                else:

                    preds = model.predict(data)
                    proba = self._predictions_to_proba(preds)

                all_probas.append(proba * weight)

            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {e}")
                continue

            if not all_probas:
                raise ValueError("No valid predictions from any model")

            avg_proba = np.sum(all_probas, axis=0)

            predictions = self.genre_labels[np.argmax(avg_proba, axis=1)]

        else:

            all_preds = []

            for (name, model), weight in zip(self.models, self.weights):
                try:
                    pred = model.predict(data)

                    repeat_count = max(1, int(weight * 10))
                    all_preds.extend([pred] * repeat_count)
                except Exception as e:
                    logger.warning(f"Error getting predictions from {name}: {e}")
                    continue

            if not all_preds:
                raise ValueError("No valid predictions from any model")

            all_preds = np.array(all_preds)
            from scipy.stats import mode
            predictions = mode(all_preds, axis=0)[0].flatten()

        return predictions

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _predictions_to_proba(self, predictions: np.ndarray) -> np.ndarray:
        n_samples = len(predictions)
        n_classes = len(self.genre_labels)
        proba = np.zeros((n_samples, n_classes))

        for i, pred in enumerate(predictions):
            class_idx = np.where(self.genre_labels == pred)[0][0]
            proba[i, class_idx] = 1.0

        return proba

    def evaluate(self, data: any, y_true: np.ndarray) -> Dict[str, float]:

        y_pred = self.predict(data)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics

    def save(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)

        config_file = save_dir / "ensemble_voting_config.pkl"
        joblib.dump({
            'model_names': [name for name, _ in self.models],
            'voting': self.voting,
            'weights': self.weights,
            'genre_labels': self.genre_labels,
            'train_accuracy': self.train_accuracy
        }, config_file)

        logger.info(f"Ensemble config saved to {config_file}")

    def prepare_data(self, df: pd.DataFrame):
        texts = []
        for idx, row in df.iterrows():
            try:
                text_path = Path(row['processed_path'])
                if not text_path.is_absolute():
                    text_path = PROCESSED_DATA_DIR.parent / text_path
                text = text_path.read_text(encoding='utf-8')
                texts.append(text)
            except Exception as e:
                logger.error(f"Error loading text: {e}")
                texts.append("")

        labels = df['genre'].values
        return texts, labels

def create_ensemble_from_trained_models(
    models: List[Tuple[str, str, any]],
    voting: str = 'soft',
    weights: List[float] = None
) -> EnsembleVotingModel:

    ensemble = EnsembleVotingModel(
        models=models,
        voting=voting,
        weights=weights
    )

    logger.info(f"Created ensemble with {len(models)} models:")
    for (name, _), weight in zip(models, ensemble.weights):
        logger.info(f"  - {name} (weight: {weight:.2f})")
    logger.info(f"Voting method: {voting}")

    return ensemble

def train_ensemble_model(
    model_dict: Dict[str, any],
    train_file: Path,
    val_file: Path,
    voting: str = 'soft',
    weights: List[float] = None,
    save_dir: Path = MODELS_DIR
) -> EnsembleVotingModel:
    logger.info("CREATING ENSEMBLE MODEL")
    logger.info("="*60)

    models = [(name, model) for name, model in model_dict.items()]

    ensemble = EnsembleVotingModel(
        models=models,
        voting=voting,
        weights=weights
    )

    logger.info(f"\nEnsemble configuration:")
    logger.info(f"  Models: {len(models)}")
    logger.info(f"  Voting: {voting}")
    logger.info(f"  Weights: {ensemble.weights}")

    for (name, _), weight in zip(models, ensemble.weights):
        logger.info(f"    - {name}: {weight:.2f}")

    val_df = pd.read_csv(val_file)
    logger.info(f"\nValidation set: {len(val_df)} books")

    logger.info("\nEnsemble created successfully!")
    logger.info("Note: Evaluation requires preparing data for each model type separately")

    ensemble.save(save_dir)

    return ensemble
