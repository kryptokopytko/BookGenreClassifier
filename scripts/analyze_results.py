"""
Advanced Results Analysis & Visualization

Creates comprehensive visualizations:
1. Per-genre accuracy comparison
2. Most confused genre pairs
3. Precision/Recall/F1 per genre
4. Performance heatmaps
5. Error analysis

Usage:
    python3 scripts/analyze_results.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("📊 ADVANCED RESULTS ANALYSIS & VISUALIZATION")
print("="*80)

# Load test data
print("\n📂 Loading test data...")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
print(f"   Test set: {len(test_df)} books")

# Load texts
def load_text(row):
    """Load text from processed file."""
    genre = row['genre'].replace('/', '_')
    filename = f"{row['book_id']}.txt"
    path = PROCESSED_DATA_DIR / genre / filename
    try:
        if path.exists():
            return path.read_text(encoding='utf-8')
    except Exception as e:
        pass
    return ""

print("\n📝 Loading test texts...")
X_test_texts = test_df.apply(load_text, axis=1).values
y_test = test_df['genre'].values
genres = sorted(np.unique(y_test))
print(f"   Loaded {len(X_test_texts)} texts")
print(f"   Genres: {len(genres)}")

# Load best model (Linear SVM)
print("\n🏆 Loading best model (Linear SVM)...")
model_path = MODELS_DIR / "linear_svm_fast.pkl"
model_data = joblib.load(model_path)
model = model_data['model']
vectorizer = model_data['vectorizer']

# Transform and predict
print("   Transforming texts...")
X_test_tfidf = vectorizer.transform(X_test_texts)
y_pred = model.predict(X_test_tfidf)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=genres)

print("\n" + "="*80)
print("📈 CREATING VISUALIZATIONS")
print("="*80)

# ============================================================================
# 1. PER-GENRE ACCURACY BAR CHART
# ============================================================================
print("\n📊 1. Per-genre accuracy...")

genre_accuracies = []
genre_counts = []
for i, genre in enumerate(genres):
    # Get indices for this genre
    genre_mask = y_test == genre
    genre_true = y_test[genre_mask]
    genre_pred = y_pred[genre_mask]

    # Calculate accuracy
    accuracy = np.mean(genre_true == genre_pred)
    count = len(genre_true)

    genre_accuracies.append(accuracy)
    genre_counts.append(count)

# Sort by accuracy
sorted_indices = np.argsort(genre_accuracies)[::-1]
sorted_genres = [genres[i] for i in sorted_indices]
sorted_accuracies = [genre_accuracies[i] for i in sorted_indices]
sorted_counts = [genre_counts[i] for i in sorted_indices]

# Create bar chart
fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.RdYlGn(np.array(sorted_accuracies))
bars = ax.barh(range(len(sorted_genres)), sorted_accuracies, color=colors, alpha=0.8)

# Add value labels
for i, (acc, count) in enumerate(zip(sorted_accuracies, sorted_counts)):
    ax.text(acc + 0.01, i, f'{acc:.1%} ({count} books)',
            va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(sorted_genres)))
ax.set_yticklabels(sorted_genres, fontsize=11)
ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Per-Genre Classification Accuracy (Linear SVM)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1.0)
ax.axvline(x=np.mean(sorted_accuracies), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {np.mean(sorted_accuracies):.1%}')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "per_genre_accuracy.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: per_genre_accuracy.png")

# ============================================================================
# 2. MOST CONFUSED GENRE PAIRS
# ============================================================================
print("\n🔀 2. Most confused genre pairs...")

# Find off-diagonal elements (misclassifications)
confusion_pairs = []
for i in range(len(genres)):
    for j in range(len(genres)):
        if i != j:  # Exclude diagonal
            confusion_pairs.append({
                'true': genres[i],
                'predicted': genres[j],
                'count': cm[i, j],
                'pair': f"{genres[i]} → {genres[j]}"
            })

confusion_df = pd.DataFrame(confusion_pairs)
confusion_df = confusion_df.sort_values('count', ascending=False).head(15)

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(confusion_df)))
bars = ax.barh(range(len(confusion_df)), confusion_df['count'], color=colors, alpha=0.8)

# Add value labels
for i, count in enumerate(confusion_df['count']):
    ax.text(count + 0.5, i, str(int(count)), va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(confusion_df)))
ax.set_yticklabels(confusion_df['pair'], fontsize=10)
ax.set_xlabel('Number of Misclassifications', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Most Confused Genre Pairs', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confused_pairs.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: confused_pairs.png")

# ============================================================================
# 3. PRECISION, RECALL, F1 PER GENRE
# ============================================================================
print("\n📐 3. Precision/Recall/F1 per genre...")

from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=genres, zero_division=0
)

# Create DataFrame
metrics_df = pd.DataFrame({
    'Genre': genres,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

# Sort by F1-score
metrics_df = metrics_df.sort_values('F1-Score', ascending=True)

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(metrics_df))
width = 0.25

bars1 = ax.barh(x - width, metrics_df['Precision'], width,
                label='Precision', color='#3498db', alpha=0.8)
bars2 = ax.barh(x, metrics_df['Recall'], width,
                label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax.barh(x + width, metrics_df['F1-Score'], width,
                label='F1-Score', color='#e74c3c', alpha=0.8)

ax.set_yticks(x)
ax.set_yticklabels(metrics_df['Genre'], fontsize=11)
ax.set_xlabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Precision, Recall, and F1-Score per Genre',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 1.0)
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "metrics_per_genre.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: metrics_per_genre.png")

# ============================================================================
# 4. CONFUSION MATRIX HEATMAP (NORMALIZED)
# ============================================================================
print("\n🔥 4. Normalized confusion matrix heatmap...")

# Normalize by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Proportion', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

# Set ticks
ax.set_xticks(np.arange(len(genres)))
ax.set_yticks(np.arange(len(genres)))
ax.set_xticklabels(genres, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(genres, fontsize=10)

# Add text annotations
for i in range(len(genres)):
    for j in range(len(genres)):
        value = cm_normalized[i, j]
        count = cm[i, j]
        if value > 0.01:  # Only show if > 1%
            text = ax.text(j, i, f'{value:.0%}\n({count})',
                          ha="center", va="center",
                          color="white" if value > 0.5 else "black",
                          fontsize=8)

ax.set_xlabel('Predicted Genre', fontsize=12, fontweight='bold')
ax.set_ylabel('True Genre', fontsize=12, fontweight='bold')
ax.set_title('Normalized Confusion Matrix (Linear SVM)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_heatmap.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: confusion_heatmap.png")

# ============================================================================
# 5. ACCURACY vs SAMPLE SIZE SCATTER
# ============================================================================
print("\n📉 5. Accuracy vs sample size...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create scatter plot
scatter = ax.scatter(genre_counts, genre_accuracies,
                    s=200, alpha=0.6, c=genre_accuracies,
                    cmap='RdYlGn', edgecolors='black', linewidth=1.5)

# Add genre labels
for i, genre in enumerate(genres):
    ax.annotate(genre, (genre_counts[i], genre_accuracies[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=9, fontweight='bold')

# Add trend line
z = np.polyfit(genre_counts, genre_accuracies, 1)
p = np.poly1d(z)
ax.plot(genre_counts, p(genre_counts), "r--", alpha=0.5, linewidth=2,
        label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')

ax.set_xlabel('Test Set Size (number of books)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Genre Accuracy vs Test Set Size', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1.0)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_vs_size.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: accuracy_vs_size.png")

# ============================================================================
# 6. SUMMARY STATISTICS TABLE
# ============================================================================
print("\n📋 6. Summary statistics table...")

summary_text = f"""
{'='*80}
GENRE CLASSIFICATION ANALYSIS SUMMARY
{'='*80}

Model: Linear SVM (C=10.0)
Overall Test Accuracy: {np.mean(y_test == y_pred):.2%}
Total Test Samples: {len(y_test)}

{'='*80}
PER-GENRE PERFORMANCE
{'='*80}

{'Genre':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Count':>8}
{'-'*80}
"""

for i, genre in enumerate(sorted_genres):
    idx = genres.index(genre)
    summary_text += f"{genre:<30} {sorted_accuracies[i]:>9.1%} {precision[idx]:>9.1%} {recall[idx]:>9.1%} {f1[idx]:>9.1%} {sorted_counts[i]:>8}\n"

summary_text += f"\n{'='*80}\n"
summary_text += f"{'Best Genre:':<30} {sorted_genres[0]} ({sorted_accuracies[0]:.1%})\n"
summary_text += f"{'Worst Genre:':<30} {sorted_genres[-1]} ({sorted_accuracies[-1]:.1%})\n"
summary_text += f"{'Average Accuracy:':<30} {np.mean(sorted_accuracies):.1%}\n"
summary_text += f"{'Std Dev:':<30} {np.std(sorted_accuracies):.1%}\n"

summary_text += f"\n{'='*80}\n"
summary_text += f"TOP 5 MOST CONFUSED PAIRS\n"
summary_text += f"{'='*80}\n"

for i, row in confusion_df.head(5).iterrows():
    summary_text += f"{row['pair']:<40} {int(row['count']):>5} misclassifications\n"

summary_text += f"\n{'='*80}\n"

# Save summary
summary_file = RESULTS_DIR / "analysis_summary.txt"
summary_file.write_text(summary_text)
print(f"   ✓ Saved: analysis_summary.txt")

# Print to console
print("\n" + summary_text)

# ============================================================================
# 7. SAVE DETAILED CSV
# ============================================================================
print("\n💾 7. Saving detailed metrics CSV...")

detailed_metrics = pd.DataFrame({
    'Genre': genres,
    'Accuracy': genre_accuracies,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Test_Samples': support,
    'Correct': [cm[i, i] for i in range(len(genres))],
    'Incorrect': [np.sum(cm[i, :]) - cm[i, i] for i in range(len(genres))]
})

detailed_metrics = detailed_metrics.sort_values('F1-Score', ascending=False)
detailed_metrics.to_csv(RESULTS_DIR / "detailed_metrics_per_genre.csv", index=False)
print(f"   ✓ Saved: detailed_metrics_per_genre.csv")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print(f"\n📁 All visualizations saved to: {RESULTS_DIR}")
print("\nGenerated files:")
print("  1. per_genre_accuracy.png        - Bar chart of accuracy per genre")
print("  2. confused_pairs.png            - Top 15 most confused genre pairs")
print("  3. metrics_per_genre.png         - Precision/Recall/F1 comparison")
print("  4. confusion_heatmap.png         - Normalized confusion matrix")
print("  5. accuracy_vs_size.png          - Accuracy vs test set size")
print("  6. analysis_summary.txt          - Text summary with statistics")
print("  7. detailed_metrics_per_genre.csv - Detailed metrics spreadsheet")
print("\n" + "="*80)
