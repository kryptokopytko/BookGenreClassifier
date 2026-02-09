"""
Comprehensive visualization and reporting script for model results.
Generates plots and a detailed markdown report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("GENERATING VISUALIZATIONS AND REPORT")
print("="*80)

results_file = RESULTS_DIR / "all_models_results.csv"
if not results_file.exists():
    print(f"❌ Results file not found: {results_file}")
    sys.exit(1)

results_df = pd.read_csv(results_file)

if 'test_acc' not in results_df.columns:
    results_df.rename(columns={
        'Model': 'model',
        'Accuracy': 'test_acc',
        'F1 Score': 'test_f1',
        'Precision': 'precision',
        'Recall': 'recall'
    }, inplace=True)

    if 'train_acc' not in results_df.columns:
        results_df['train_acc'] = np.nan

print(f"\n✓ Loaded results for {len(results_df)} models")

# ============================================================================
# ============================================================================

print("\n📊 Creating model comparison chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

results_sorted = results_df.sort_values('test_acc', ascending=True)

ax1.barh(results_sorted['model'], results_sorted['test_acc'], color='steelblue')
ax1.set_xlabel('Test Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)

for i, v in enumerate(results_sorted['test_acc']):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

ax2.barh(results_sorted['model'], results_sorted['test_f1'], color='coral')
ax2.set_xlabel('Test F1 Score', fontsize=12, fontweight='bold')
ax2.set_title('Model Test F1 Score Comparison', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

for i, v in enumerate(results_sorted['test_f1']):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
comparison_file = RESULTS_DIR / "model_comparison.png"
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {comparison_file}")
plt.close()

# ============================================================================
# ============================================================================

print("\n📊 Creating overfitting analysis chart...")

models_with_train = results_df[results_df['train_acc'].notna()].copy()
overfitting_models = pd.DataFrame()

if len(models_with_train) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(models_with_train))
    width = 0.35

    bars1 = ax.bar(x - width/2, models_with_train['train_acc'], width,
                   label='Train Accuracy', color='lightgreen', alpha=0.8)
    bars2 = ax.bar(x + width/2, models_with_train['test_acc'], width,
                   label='Test Accuracy', color='lightcoral', alpha=0.8)

    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Train vs Test Accuracy (Overfitting Detection)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_with_train['model'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    for i, (train, test) in enumerate(zip(models_with_train['train_acc'],
                                           models_with_train['test_acc'])):
        gap = train - test
        if gap > 0.15:  # Significant overfitting
            ax.text(i, train + 0.03, f'⚠️ {gap:.2f}',
                   ha='center', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    overfitting_file = RESULTS_DIR / "overfitting_analysis.png"
    plt.savefig(overfitting_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {overfitting_file}")
    plt.close()

# ============================================================================
# ============================================================================

print("\n📊 Creating top models comparison...")

top_n = min(5, len(results_df))
top_models = results_df.nlargest(top_n, 'test_acc')

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['test_acc', 'test_f1']
metric_names = ['Accuracy', 'F1 Score']
x = np.arange(len(top_models))
width = 0.35

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    offset = width * (i - 0.5)
    ax.bar(x + offset, top_models[metric], width, label=name, alpha=0.8)

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Top {top_n} Models - Detailed Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_models['model'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
top_models_file = RESULTS_DIR / "top_models_comparison.png"
plt.savefig(top_models_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {top_models_file}")
plt.close()

# ============================================================================
# ============================================================================

print("\n📊 Creating performance heatmap...")

heatmap_data = results_df[['model', 'test_acc', 'test_f1']].set_index('model')
heatmap_data.columns = ['Test Accuracy', 'Test F1']

fig, ax = plt.subplots(figsize=(8, max(6, len(results_df) * 0.4)))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, cbar_kws={'label': 'Score'},
            linewidths=0.5, ax=ax)
ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

heatmap_file = RESULTS_DIR / "performance_heatmap.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {heatmap_file}")
plt.close()

# ============================================================================
# ============================================================================

print("\n📝 Generating markdown report...")

report = []
report.append("# 📊 Book Genre Classifier - Model Results Report")
report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"\n**Total Models Trained:** {len(results_df)}")
report.append("\n---\n")

report.append("## 🎯 Executive Summary\n")

best_model = results_df.loc[results_df['test_acc'].idxmax()]
worst_model = results_df.loc[results_df['test_acc'].idxmin()]

report.append(f"### Best Model: **{best_model['model']}**")
report.append(f"- **Test Accuracy:** {best_model['test_acc']:.4f} ({best_model['test_acc']*100:.2f}%)")
report.append(f"- **Test F1 Score:** {best_model['test_f1']:.4f}")
if pd.notna(best_model['train_acc']):
    overfitting = best_model['train_acc'] - best_model['test_acc']
    report.append(f"- **Overfitting Gap:** {overfitting:.4f} {'⚠️ HIGH' if overfitting > 0.15 else '✓ OK'}")

report.append(f"\n### Worst Model: **{worst_model['model']}**")
report.append(f"- **Test Accuracy:** {worst_model['test_acc']:.4f} ({worst_model['test_acc']*100:.2f}%)")
report.append(f"- **Test F1 Score:** {worst_model['test_f1']:.4f}")

improvement = (best_model['test_acc'] - worst_model['test_acc']) / worst_model['test_acc'] * 100
report.append(f"\n**Performance Range:** {improvement:.1f}% improvement from worst to best model")

report.append("\n---\n")

report.append("## 📈 Model Comparison\n")
report.append("![Model Comparison](model_comparison.png)\n")
report.append("*Figure 1: Test accuracy and F1 score comparison across all models*\n")

if len(models_with_train) > 0:
    report.append("## 🔍 Overfitting Analysis\n")
    report.append("![Overfitting Analysis](overfitting_analysis.png)\n")
    report.append("*Figure 2: Train vs Test accuracy - large gaps indicate overfitting*\n")

    overfitting_models = models_with_train[
        (models_with_train['train_acc'] - models_with_train['test_acc']) > 0.15
    ]

    if len(overfitting_models) > 0:
        report.append("### ⚠️ Models with Significant Overfitting (>15% gap):\n")
        for _, row in overfitting_models.iterrows():
            gap = row['train_acc'] - row['test_acc']
            report.append(f"- **{row['model']}**: Train {row['train_acc']:.3f} → Test {row['test_acc']:.3f} (gap: {gap:.3f})")
        report.append("\n**Recommendation:** These models need regularization (lower max_depth, increase min_samples_split, add dropout)\n")

report.append("## 🏆 Top 5 Models\n")
report.append("![Top Models](top_models_comparison.png)\n")
report.append("*Figure 3: Detailed metrics for the best performing models*\n")

report.append("| Rank | Model | Test Accuracy | Test F1 | Status |\n")
report.append("|------|-------|---------------|---------|--------|\n")

for i, (_, row) in enumerate(top_models.iterrows(), 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐"
    status = "✓" if pd.isna(row['train_acc']) or (row['train_acc'] - row['test_acc']) < 0.15 else "⚠️"
    report.append(f"| {emoji} {i} | {row['model']} | {row['test_acc']:.4f} | {row['test_f1']:.4f} | {status} |\n")

report.append("\n## 🌡️ Performance Heatmap\n")
report.append("![Performance Heatmap](performance_heatmap.png)\n")
report.append("*Figure 4: Heatmap visualization of model performance metrics*\n")

report.append("## 📋 Detailed Results\n")
report.append("| Model | Train Acc | Test Acc | Test F1 | Overfitting Gap |\n")
report.append("|-------|-----------|----------|---------|----------------|\n")

for _, row in results_df.sort_values('test_acc', ascending=False).iterrows():
    train_acc = f"{row['train_acc']:.4f}" if pd.notna(row['train_acc']) else "N/A"
    gap = ""
    if pd.notna(row['train_acc']):
        gap_val = row['train_acc'] - row['test_acc']
        gap = f"{gap_val:.4f}"
        if gap_val > 0.15:
            gap += " ⚠️"

    report.append(f"| {row['model']} | {train_acc} | {row['test_acc']:.4f} | {row['test_f1']:.4f} | {gap} |\n")

report.append("\n## 💡 Recommendations\n")

report.append("### For Production Use:\n")
report.append(f"1. **Primary Model:** Use **{best_model['model']}** (highest accuracy)")
report.append(f"2. **Ensemble:** Combine top 3-5 models with voting for robustness")

top_3 = results_df.nlargest(3, 'test_acc')
if top_3['test_acc'].std() < 0.05:
    report.append("3. **Ensemble Strategy:** Top models have similar performance - use weighted voting")
else:
    report.append("3. **Ensemble Strategy:** Models show diversity - simple majority voting may work well")

report.append("\n### To Improve Performance:\n")

if len(overfitting_models) > 0:
    report.append("- **Fix Overfitting:** Reduce model complexity, add regularization, increase training data")

if best_model['test_acc'] - results_df['test_acc'].median() > 0.1:
    report.append(f"- **Model Architecture:** {best_model['model']} works significantly better - investigate why")

report.append("- **Data Augmentation:** Generate synthetic samples for underrepresented genres")
report.append("- **Feature Engineering:** Add more domain-specific features")
report.append("- **Hyperparameter Tuning:** Run grid search on top 3 models")
report.append("- **Cross-Validation:** Verify results with k-fold cross-validation")

report.append("\n### Model-Specific Insights:\n")

text_models = ['KNN', 'Ridge', 'Nearest Centroid', 'TF-IDF']
text_results = results_df[results_df['model'].str.contains('|'.join(text_models), case=False, na=False)]

if len(text_results) > 0:
    avg_text = text_results['test_acc'].mean()
    report.append(f"- **Text-based models** average: {avg_text:.4f}")

feature_models = ['XGBoost', 'LightGBM', 'Random Forest', 'Feature']
feature_results = results_df[results_df['model'].str.contains('|'.join(feature_models), case=False, na=False)]

if len(feature_results) > 0:
    avg_feature = feature_results['test_acc'].mean()
    report.append(f"- **Feature-based models** average: {avg_feature:.4f}")

    if len(text_results) > 0 and avg_feature > avg_text:
        report.append(f"  - Feature-based models outperform text-based by {(avg_feature - avg_text)*100:.1f}%")

report.append("\n## 📊 Dataset Information\n")

try:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    report.append(f"- **Training samples:** {len(train_df)}")
    report.append(f"- **Validation samples:** {len(val_df)}")
    report.append(f"- **Test samples:** {len(test_df)}")
    report.append(f"- **Total samples:** {len(train_df) + len(val_df) + len(test_df)}")

    if 'genre' in train_df.columns:
        report.append(f"\n### Genre Distribution (Training Set):\n")
        genre_counts = train_df['genre'].value_counts()
        for genre, count in genre_counts.items():
            pct = count / len(train_df) * 100
            report.append(f"- **{genre}:** {count} ({pct:.1f}%)")

        balance_ratio = genre_counts.max() / genre_counts.min()
        if balance_ratio > 2:
            report.append(f"\n⚠️ **Class Imbalance Detected:** {balance_ratio:.1f}x difference between most and least common genres")
            report.append("   - Consider class weights or oversampling minority classes")

except Exception as e:
    report.append(f"*Could not load dataset info: {e}*")

report.append("\n---")
report.append("\n*Report generated automatically by `visualize_results.py`*")
report.append(f"\n*Models saved in: `{MODELS_DIR}`*")
report.append(f"\n*Results saved in: `{RESULTS_DIR}`*")

report_file = RESULTS_DIR / "MODEL_RESULTS.md"
with open(report_file, 'w') as f:
    f.write('\n'.join(report))

print(f"\n✓ Saved report: {report_file}")

print("\n" + "="*80)
print("✅ VISUALIZATION AND REPORT COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {comparison_file}")
if len(models_with_train) > 0:
    print(f"  - {overfitting_file}")
print(f"  - {top_models_file}")
print(f"  - {heatmap_file}")
print(f"  - {report_file}")
print(f"\n📖 View full report: {report_file}")
