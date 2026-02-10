import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger
from ..utils.config import RESULTS_DIR

logger = get_logger(__name__)

class DimensionalityVisualizer:

    def __init__(self):
        self.scaler = StandardScaler()

    def tsne_projection(
        self,
        X: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42
    ) -> np.ndarray:
        logger.info(f"Input shape: {X.shape}")

        X_scaled = self.scaler.fit_transform(X)

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            verbose=1
        )

        X_tsne = tsne.fit_transform(X_scaled)

        logger.info(f"t-SNE projection complete. Output shape: {X_tsne.shape}")

        return X_tsne

    def pca_projection(
        self,
        X: np.ndarray,
        n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Input shape: {X.shape}")

        X_scaled = self.scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        logger.info(f"PCA projection complete. Output shape: {X_pca.shape}")
        logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")

        return X_pca, pca.explained_variance_ratio_

    def plot_2d_projection(
        self,
        X_proj: np.ndarray,
        labels: np.ndarray,
        title: str = "2D Projection",
        output_file: Optional[Path] = None,
        method: str = "t-SNE"
    ):
        unique_labels = sorted(np.unique(labels))
        n_labels = len(unique_labels)

        palette = sns.color_palette("husl", n_labels)
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(14, 10))

        for label in unique_labels:
            mask = labels == label
            plt.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                c=[color_map[label]],
                label=label,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )

        plt.xlabel(f'{method} Dimension 1', fontsize=12)
        plt.ylabel(f'{method} Dimension 2', fontsize=12)
        plt.title(title, fontsize=16, pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")

        plt.close()

    def plot_3d_projection(
        self,
        X_proj: np.ndarray,
        labels: np.ndarray,
        title: str = "3D Projection",
        output_file: Optional[Path] = None,
        method: str = "t-SNE"
    ):

        unique_labels = sorted(np.unique(labels))
        n_labels = len(unique_labels)

        palette = sns.color_palette("husl", n_labels)
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                X_proj[mask, 2],
                c=[color_map[label]],
                label=label,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
        ax.set_zlabel(f'{method} Dimension 3', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")

        plt.close()

    def plot_pca_variance(
        self,
        explained_variance: np.ndarray,
        output_file: Optional[Path] = None
    ):
        cumulative_variance = np.cumsum(explained_variance)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.bar(range(1, n_components + 1), explained_variance, alpha=0.7)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Variance Explained by Each Component', fontsize=14)
        ax1.grid(alpha=0.3)

        ax2.plot(range(1, n_components + 1), cumulative_variance, 'o-', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")

        plt.close()

def visualize_with_tsne(
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: Path = RESULTS_DIR / "tsne",
    perplexity: float = 30.0,
    n_components: int = 2
):
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = DimensionalityVisualizer()

    X_tsne = visualizer.tsne_projection(
        X,
        n_components=n_components,
        perplexity=perplexity
    )

    if n_components >= 2:
        visualizer.plot_2d_projection(
            X_tsne,
            labels,
            title=f"t-SNE 2D Projection (perplexity={perplexity})",
            output_file=output_dir / "tsne_2d.png",
            method="t-SNE"
        )

    if n_components == 3:
        visualizer.plot_3d_projection(
            X_tsne,
            labels,
            title=f"t-SNE 3D Projection (perplexity={perplexity})",
            output_file=output_dir / "tsne_3d.png",
            method="t-SNE"
        )

    logger.info(f"t-SNE visualization complete. Saved to {output_dir}")

def visualize_with_pca(
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: Path = RESULTS_DIR / "pca",
    n_components: int = 10
):
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = DimensionalityVisualizer()

    X_pca, explained_variance = visualizer.pca_projection(X, n_components=n_components)

    visualizer.plot_2d_projection(
        X_pca[:, :2],
        labels,
        title="PCA 2D Projection",
        output_file=output_dir / "pca_2d.png",
        method="PCA"
    )

    if n_components >= 3:
        visualizer.plot_3d_projection(
            X_pca[:, :3],
            labels,
            title="PCA 3D Projection",
            output_file=output_dir / "pca_3d.png",
            method="PCA"
        )

    visualizer.plot_pca_variance(
        explained_variance,
        output_file=output_dir / "pca_variance.png"
    )

    logger.info(f"PCA visualization complete. Saved to {output_dir}")

    return X_pca, explained_variance
