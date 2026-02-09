import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

from ..utils.logger import get_logger
from ..utils.config import RESULTS_DIR, GENRES

logger = get_logger(__name__)

model_names = ["baseline_model"]

class ModelEvaluator:

    def __init__(self, genre_labels: List[str]):
        self.genre_labels = genre_labels

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=self.genre_labels, zero_division=0
        )

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=self.genre_labels)

        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': dict(zip(self.genre_labels, precision)),
            'recall_per_class': dict(zip(self.genre_labels, recall)),
            'f1_per_class': dict(zip(self.genre_labels, f1)),
            'support_per_class': dict(zip(self.genre_labels, support)),
            'confusion_matrix': cm
        }

        return metrics

    def print_metrics(self, metrics: Dict[str, Any]):
        logger.info("EVALUATION METRICS")
        logger.info("="*60)

        logger.info(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
        logger.info(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
        logger.info(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")

        logger.info("\nPer-Class Metrics:")
        logger.info("-" * 60)
        logger.info(f"{'Genre':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        logger.info("-" * 60)

        for genre in self.genre_labels:
            precision = metrics['precision_per_class'][genre]
            recall = metrics['recall_per_class'][genre]
            f1 = metrics['f1_per_class'][genre]
            support = metrics['support_per_class'][genre]

            logger.info(f"{genre:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

        logger.info("-" * 60)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        output_file: Path = None,
        normalize: bool = False,
        title: str = "Confusion Matrix"
    ):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.genre_labels,
            yticklabels=self.genre_labels,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )

        plt.title(title, fontsize=16, pad=20)
        plt.ylabel('True Genre', fontsize=12)
        plt.xlabel('Predicted Genre', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_file}")

        plt.close()

    def plot_per_class_metrics(
        self,
        metrics: Dict[str, Any],
        output_file: Path = None,
        title: str = "Per-Class Performance"
    ):
        precision = [metrics['precision_per_class'][g] for g in GENRES]
        recall = [metrics['recall_per_class'][g] for g in GENRES]
        f1 = [metrics['f1_per_class'][g] for g in GENRES]

        x = np.arange(len(GENRES))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(GENRES, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to {output_file}")

        plt.close()

    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        output_file: Path = None
    ):
        metrics_to_compare = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        data = {metric: [] for metric in metrics_to_compare}

        for model_name in model_names:
            for metric in metrics_to_compare:
                data[metric].append(results[model_name][metric])

        x = np.arange(len(model_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(metrics_to_compare):
            ax.bar(x + i * width, data[metric], width, label=metric.replace('_', ' ').title())

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=16, pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {output_file}")

        plt.close()

    def save_results(
        self,
        metrics: Dict[str, Any],
        output_file: Path,
        model_name: str = "Model",
        train_accuracy: float = None,
        val_accuracy: float = None
    ):
        overall_metrics = {
            'model': model_name,
            'accuracy': metrics['accuracy'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted'],
            'f1_weighted': metrics['f1_weighted']
        }

        if train_accuracy is not None:
            overall_metrics['train_accuracy'] = train_accuracy
        if val_accuracy is not None:
            overall_metrics['val_accuracy'] = val_accuracy

        if train_accuracy is not None:
            overall_metrics['train_test_gap'] = train_accuracy - metrics['accuracy']

        per_class_data = []
        for genre in self.genre_labels:
            per_class_data.append({
                'model': model_name,
                'genre': genre,
                'precision': metrics['precision_per_class'][genre],
                'recall': metrics['recall_per_class'][genre],
                'f1': metrics['f1_per_class'][genre],
                'support': metrics['support_per_class'][genre]
            })

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        overall_df = pd.DataFrame([overall_metrics])
        overall_file = output_file.parent / f"{output_file.stem}_overall.csv"
        overall_df.to_csv(overall_file, index=False)
        logger.info(f"Overall metrics saved to {overall_file}")

        per_class_df = pd.DataFrame(per_class_data)
        per_class_file = output_file.parent / f"{output_file.stem}_per_class.csv"
        per_class_df.to_csv(per_class_file, index=False)
        logger.info(f"Per-class metrics saved to {per_class_file}")

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genre_labels: List[str],
    model_name: str = "Model",
    output_dir: Path = RESULTS_DIR,
    save_plots: bool = True,
    train_accuracy: float = None,
    val_accuracy: float = None
) -> Dict[str, Any]:
    evaluator = ModelEvaluator(genre_labels)

    metrics = evaluator.calculate_metrics(y_true, y_pred)

    evaluator.print_metrics(metrics)

    if save_plots:
        output_dir = Path(output_dir) / model_name.replace(' ', '_').lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluator.plot_confusion_matrix(
            metrics['confusion_matrix'],
            output_file=output_dir / "confusion_matrix.png",
            title=f"Confusion Matrix - {model_name}"
        )

        evaluator.plot_confusion_matrix(
            metrics['confusion_matrix'],
            output_file=output_dir / "confusion_matrix_normalized.png",
            normalize=True,
            title=f"Normalized Confusion Matrix - {model_name}"
        )

        evaluator.plot_per_class_metrics(
            metrics,
            output_file=output_dir / "per_class_metrics.png",
            title=f"Per-Class Performance - {model_name}"
        )

        evaluator.save_results(
            metrics,
            output_file=output_dir / "results.csv",
            model_name=model_name,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy
        )

    return metrics
