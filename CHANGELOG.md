# Changelog

Wszystkie istotne zmiany w projekcie będą dokumentowane w tym pliku.

Format bazuje na [Keep a Changelog](https://keepachangelog.com/pl/1.0.0/),
a projekt stosuje [Semantic Versioning](https://semver.org/lang/pl/).

---

## [2.0.0] - 2026-02-09

### ✅ Added
- **Comprehensive README.md** z pełną dokumentacją projektu
- **MODEL_ALGORITHMS_EXPLAINED.md** - 900+ linii szczegółowych wyjaśnień algorytmów
- **train_all_models.py** - skrypt do treningu wszystkich 12 modeli
- **train_without_knn.py** - szybsza wersja bez KNN
- **test_model.py** - interaktywny tester modeli z przykładami
- **visualize_results.py** - automatyczne generowanie raportów i wykresów
- **4 nowe modele**: Ridge Classifier, Nearest Centroid, Style-based, Ensemble Voting

### 🐛 Fixed
- **20 critical bugs** naprawionych w 8 plikach modeli:
  - `baseline_model.py` (3 bugs)
  - `xgboost_model.py` (2 bugs)
  - `knn_model.py` (3 bugs)
  - `ridge_model.py` (4 bugs)
  - `naive_bayes_model.py` (4 bugs)
  - `lightgbm_model.py` (3 bugs)
  - `tfidf_model.py` (5 bugs)
  - `ensemble_model.py` (3 bugs)

### 🔧 Changed
- **Random Forest overfitting fix**: max_depth 15→8, min_samples_leaf +10
- **TF-IDF optimization**: 15000→5000 features, trigrams→bigrams dla szybkości
- **Config optimization**: Logistic Regression C=2.0, solver='saga'
- **NEAREST_CENTROID_PARAMS**: metric 'cosine'→'euclidean' (sklearn compatibility)

### 📊 Performance
- **Linear SVM**: 53.6% accuracy, 53.3% F1 (BEST MODEL)
- **Logistic Regression**: 47.5% accuracy, 48.1% F1
- **Random Forest**: 36.3% accuracy (fixed from 85% train overfitting)
- **Naive Bayes**: 27.0% accuracy (biased to Biography)

### 📚 Documentation
- **Comprehensive testing**: All 5 models tested on full test set
- **Confusion matrices**: Generated for all models
- **Classification reports**: Detailed per-genre metrics
- **MODEL_RESULTS.md**: Executive summary + recommendations

---

## [1.5.0] - 2026-02-05

### ✅ Added
- **test_all_models.py** - comprehensive testing script
- **Confusion matrix visualization** for all models
- **Per-genre performance analysis**
- **results.md** - detailed analysis report

### 📊 Performance
- Initial testing of 5 basic models
- Identified overfitting in Random Forest (100% train → 55.8% test)
- Best model: Linear SVM with 72.6% test accuracy

---

## [1.0.0] - 2026-01-XX

### ✅ Added
- Initial project structure
- Data collection from Project Gutenberg
- Basic preprocessing pipeline
- Train/val/test split (70/15/15)
- 4 baseline models:
  - Linear SVM
  - Logistic Regression
  - Random Forest
  - Naive Bayes
- Basic evaluation metrics

### 📊 Dataset
- 3,703 books total
- 8 balanced genres (~450-500 books each)
- Metadata tracking (author, title, genre, source)

---

## [0.5.0] - 2026-01-XX

### ✅ Added
- Project setup and structure
- Gutenberg scraper
- Data preprocessing utilities
- Configuration file
- Requirements.txt

---

## Legenda

- `Added` - nowe funkcjonalności
- `Changed` - zmiany w istniejących funkcjach
- `Deprecated` - funkcje wkrótce usunięte
- `Removed` - usunięte funkcje
- `Fixed` - naprawione bugi
- `Security` - poprawki bezpieczeństwa
- `Performance` - ulepszenia wydajności
- `Documentation` - zmiany w dokumentacji
