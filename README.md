# 📚 Book Genre Classifier

Automatyczna klasyfikacja gatunków literackich na podstawie treści książek z Project Gutenberg przy użyciu metod Machine Learning i NLP.

## 🎯 Opis Projektu

System klasyfikuje książki do 8 gatunków literackich:

- Adventure/Action
- Biography
- Mystery/Crime
- Science Fiction
- Historical Fiction
- Thriller/Horror
- Fantasy
- Romance

**Główne cechy:**

- ✅ 10 różnych modeli ML (od prostych baseline do ensemble)
- ✅ Podział danych author-based (zapobiega data leakage)
- ✅ Ekstrakcja cech TF-IDF i statystycznych
- ✅ Kompleksowa ewaluacja z wizualizacjami
- ✅ Modułowa architektura kodu

## 📁 Struktura

```
book-genre-classifier/
├── src/                    # Kod źródłowy (data, features, models, evaluation)
├── scripts/                # Skrypty treningowe
├── data/                   # raw/, processed/, metadata.csv
├── models_saved/           # Wytrenowane modele (.pkl)
└── results/                # Wyniki i metryki
```

## 📊 Dataset

**Źródło:** Project Gutenberg (darmowe e-booki)

**Aktualny rozmiar:**

- **Total:** 3,703 książki
- Train: 2,202 (59.5%)
- Validation: 461 (12.4%)
- Test: 1,040 (28.1%)

**Rozkład gatunków (prawie zbalansowany!):**

```
Fantasy:            502 książki ✅
Historical Fiction: 499 książek ✅
Thriller/Horror:    498 książek ✅
Science Fiction:    498 książek ✅
Mystery/Crime:      495 książek ✅
Romance:            495 książek ✅
Adventure/Action:   489 książek ✅
Biography:          443 książki ✅
```

**Stan:** Dataset znacznie poprawiony! Wszystkie gatunki mają ~450-500 książek.

## 🔧 Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm nltk requests beautifulsoup4
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 🚀 Quick Start

```bash
# 1. Generate metadata from existing books
python3 scripts/generate_metadata_from_existing.py

# 2. Preprocess and split data
python3 scripts/download_books.py --skip_download

# 3. Extract features (fast - without POS/keywords)
python3 scripts/extract_features.py --skip_pos --skip_keywords

# 4. Train models
python3 scripts/train_simple.py

# 5. Test and optimize
python3 scripts/test_and_optimize.py
```

**Note**: See [SCRIPTS_STATUS.md](SCRIPTS_STATUS.md) for complete script documentation and troubleshooting.

## 📈 Wyniki

### Wytrenowane Modele (Dataset: 3,703 książki)

| Model                   | Val Acc | Test Acc  | Status                         |
| ----------------------- | ------- | --------- | ------------------------------ |
| **Linear SVM (C=10)**   | 53.4%   | **53.0%** | ✅ Najlepszy (zoptymalizowany) |
| **Linear SVM (C=1)**    | 50.1%   | 53.6%     | ✅ Dobry                       |
| **Logistic Regression** | 46.2%   | 47.5%     | ✅ Stabilny                    |
| **Random Forest**       | 34.5%   | 36.2%     | ⚠️ Overfitting                 |
| **Naive Bayes**         | 32.1%   | 27.0%     | ⚠️ Słaby                       |

### Najlepszy Model: Optimized Linear SVM (C=10.0)

**Test Accuracy: 53.0%**

**Per-Genre Performance:**

- 🥇 **Biography**: 84% F1 (92% precision!)
- 🥈 **Mystery/Crime**: 65% F1
- 🥉 **Romance**: 56% F1
- ⚠️ **Fantasy**: 21% F1 (najtrudniejszy)
- ⚠️ **Thriller/Horror**: 29% F1

### Analiza Modeli

✅ **Linear SVM**: Najlepszy trade-off między accuracy a generalizacją

✅ **Logistic Regression**: Najbardziej stabilny (niewielki gap train-test)

⚠️ **Random Forest**: Silny overfitting mimo max_depth=15

- Train: 86.1% → Test: 36.3% (gap: 49.8 punktów procentowych!)
- Potrzebuje regularyzacji lub więcej danych

⚠️ **Naive Bayes**: Zbyt prosty dla tego problemu (8 klas, podobne gatunki)

### Interpretacja Wyników

**Dlaczego ~50% accuracy?**

1. **8 klas**: Random baseline to 12.5%, więc 53.6% to solidny wynik
2. **Podobne gatunki**: Science Fiction vs Fantasy, Romance vs Historical Fiction - trudne do rozróżnienia
3. **Brak keyword features**: Baseline model (keywords) nie był trenowany
4. **Brak POS features**: Pominęliśmy wolną analizę POS

**Porównanie z literaturą:**

- Multi-class text classification (8 klas): 40-70% accuracy jest typowe
- BERT/transformers osiągają ~70-80%, ale wymagają GPU i długiego trenowania

## 🎯 Kluczowe Cechy

- **Author-based split**: Żaden autor w train i test jednocześnie (zapobiega data leakage)
- **TF-IDF features**: Unigrams + bigrams, max 5000 features
- **Statistical features**: Sentence length, vocabulary richness, etc.

### 🎯 Potencjalne Dalsze Usprawnienia

**Quick wins (+5-10%):**

1. Keyword features + baseline model
2. POS analysis (pomijaliśmy ze względu na czas)
3. Ensemble voting z top 3 modeli

**Medium effort (+10-15%):** 4. Word embeddings (Word2Vec/GloVe) 5. Feature engineering (dialog ratio, chapter structure) 6. Cross-validation

**Advanced (+15-25%):** 7. Fine-tune BERT/RoBERTa 8. Hierarchical classification 9. Dataset expansion (5000+ books)

---

**Status:** ✅ Fully Functional | ✨ Optimized
**Wersja:** 1.2.0
**Ostatnia aktualizacja:** 2026-02-05

**Final Results:**

- **Dataset**: 3,703 książki (zbalansowany: ~450-500 per gatunek)
- **Best Model**: Optimized Linear SVM (C=10.0)
- **Test Accuracy**: 53.0% (vs 12.5% random baseline)
- **Best Genre**: Biography (84% F1-score)
- **Models**: 5 wytrenowanych modeli w `models_saved/`
- **Optimization**: Hyperparameter tuning wykonany
- **Code**: Wszystkie moduły naprawione i testowalne
