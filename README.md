# 📚 Book Genre Classifier

Automatyczna klasyfikacja gatunków literackich przy użyciu Machine Learning i NLP.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Spis Treści

- [Opis Projektu](#opis-projektu)
- [Dane](#dane)
- [Modele](#modele)
- [Wyniki](#wyniki)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Struktura Projektu](#struktura-projektu)
- [Dokumentacja](#dokumentacja)

---

## 🎯 Opis Projektu

Celem projektu jest automatyczne przypisywanie książkom gatunków literackich na podstawie ich treści. Projekt wykorzystuje różne algorytmy uczenia maszynowego i metody przetwarzania języka naturalnego (NLP).

### Kluczowe Funkcje

- ✅ **8 gatunków literackich** - zbalansowany dataset
- ✅ **12 różnych modeli ML** - od prostych (Naive Bayes) do zaawansowanych (XGBoost)
- ✅ **Pełna analiza i wizualizacje** - confusion matrices, reports, wykresy
- ✅ **Interaktywny tester** - testuj modele na własnych tekstach
- ✅ **Szczegółowa dokumentacja** - opis wszystkich algorytmów

---

## 📊 Dane

### Źródło
**Project Gutenberg** - kolekcja darmowych e-booków

### Rozmiar Datasetu
- **Całość**: 3,703 książek (zbalansowane: ~450-500 per gatunek)
- **Train**: 3,080 książek (70%)
- **Validation**: 660 książek (15%)
- **Test**: 660 książek (15%)

### Gatunki
1. 🗺️ **Adventure** - Przygodowe
2. 📖 **Biography** - Biografie
3. 🔍 **Mystery/Crime** - Kryminały i tajemnice
4. 🚀 **Science Fiction** - Science Fiction
5. 🏛️ **Historical Fiction** - Powieści historyczne
6. 😱 **Thriller/Horror** - Thrillery i horrory
7. 🧙 **Fantasy** - Fantasy
8. 💕 **Romance** - Romanse

### Preprocessing
- Usuwanie nagłówków Project Gutenberg
- Tokenizacja i normalizacja
- Podział train/val/test **po autorach** (zapobiega data leakage)
- Filtrowanie zbyt krótkich/długich książek (3K - 500K znaków)

---

## 🤖 Modele

### Modele Oparte na Tekście (TF-IDF)

| Model | Accuracy | F1 Score | Status | Opis |
|-------|----------|----------|--------|------|
| **Linear SVM** | 53.6% | 53.3% | ⭐ BEST | Najlepszy model - wykorzystuje hiperpłaszczyzny |
| **Linear SVM (Opt)** | 53.0% | 53.2% | ✅ | Zoptymalizowana wersja |
| **Logistic Regression** | 47.5% | 48.1% | ✅ | Szybki baseline z prawdopodobieństwami |
| **Random Forest** | 36.3% | 37.0% | ⚠️  | Overfitting (85% train → 36% test) |
| **Naive Bayes** | 27.0% | 17.9% | ❌ | Słaby - bias do Biography |

### Modele w Treningu
- 🔄 **Ridge Classifier** - L2 regularization
- 🔄 **Nearest Centroid** - Distance-based, bardzo szybki
- 🔄 **KNN** - K-nearest neighbors (wolny na dużych danych)
- 🔄 **Style-based** - Analiza stylu pisania

### Modele Zaawansowane (Feature-based)
- 🚀 **XGBoost** - Gradient boosting
- ⚡ **LightGBM** - Szybki gradient boosting
- 🌳 **Feature-based RF** - Random Forest na extracted features

---

## 🎯 Wyniki

### Top Model: Linear SVM

**Test Set Performance:**
- **Accuracy**: 53.6% (4.3x lepsze niż random baseline - 12.5%)
- **Weighted F1**: 53.3%
- **Training Time**: ~2 minuty
- **Prediction Speed**: <1ms per książkę

### Performance per Genre

| Gatunek | Precision | Recall | F1 Score | Trudność |
|---------|-----------|--------|----------|----------|
| Biography | 87% | 75% | 80% | 🟢 Łatwy |
| Mystery/Crime | 67% | 76% | 71% | 🟢 Średni |
| Romance | 57% | 52% | 54% | 🟡 Średni |
| Historical Fiction | 43% | 52% | 47% | 🟡 Trudny |
| Science Fiction | 47% | 43% | 45% | 🟡 Trudny |
| Adventure | 39% | 45% | 42% | 🟡 Trudny |
| Fantasy | 22% | 36% | 27% | 🔴 Bardzo trudny |
| Thriller/Horror | 23% | 29% | 25% | 🔴 Bardzo trudny |

### Kluczowe Insights

✅ **Biography** jest najłatwiejszym gatunkiem (wyrazista leksyka: "born", "died", "early life")

⚠️ **Fantasy/SciFi** są bardzo mylone (39 książek - overlap w tematyce)

⚠️ **Thriller/Horror** najtrudniejszy (25% F1) - potrzeba więcej danych treningowych

❌ **Random Forest** ma silny overfitting (85% train → 36% test)

---

## 🚀 Instalacja

### Wymagania
- Python 3.9+
- 8GB+ RAM (dla treningu wszystkich modeli)
- ~5GB miejsca na dysku (książki + modele)

### Szybka Instalacja

```bash
# Clone repository
git clone https://github.com/your-username/book-genre-classifier.git
cd book-genre-classifier

# Install dependencies
pip install -r requirements.txt

# Download data (optional - already processed)
python3 scripts/download_books.py --skip_download

# Train models (or use pre-trained)
python3 scripts/train_simple.py
```

---

## 💻 Użycie

### 1. Trenowanie Modeli

```bash
# Szybki trening (4 podstawowe modele)
python3 scripts/train_simple.py

# Pełny trening (wszystkie modele) - UWAGA: długo trwa!
python3 scripts/train_all_models.py

# Trening bez KNN (szybciej)
python3 scripts/train_without_knn.py
```

### 2. Testowanie Modeli

```bash
# Kompletny test + wizualizacje
python3 scripts/test_all_models.py

# Test i optymalizacja
python3 scripts/test_and_optimize.py
```

### 3. Interaktywny Tester

```bash
# Uruchom interaktywny tester
python3 scripts/test_model.py

# Wybierz model (1-7)
# Testuj na przykładowych tekstach lub własnych!
```

**Przykład użycia:**
```
📋 Available Models:
  1. ✓ Linear SVM
  2. ✓ Logistic Regression
  3. ✓ Naive Bayes
  4. ✓ Random Forest

Select model (1-7): 1

TEST OPTIONS:
  1. Test with sample texts
  2. Enter custom text
  3. Load text from file

Your choice: 2

Enter your text (end with empty line):
The detective examined the blood-stained knife carefully.
The victim had been dead for hours. Who could have done this?

🎯 PREDICTION RESULT
**Predicted Genre:** Mystery/Crime

**Confidence Scores:**
  Mystery/Crime                  ████████████████████░░░░░░░░░░░░░░░░░░░░  78.23%
  Thriller/Horror                ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  12.45%
  Adventure                      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   5.67%
```

### 4. Generowanie Raportów

```bash
# Wygeneruj pełny raport z wykresami
python3 scripts/visualize_results.py

# Wynik: results/MODEL_RESULTS.md + 4 PNG charts
```

---

## 📁 Struktura Projektu

```
book-genre-classifier/
│
├── src/                          # Kod źródłowy
│   ├── data/                     # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── gutenberg_scraper.py  # Scraping z Project Gutenberg
│   │   └── preprocessing.py      # Czyszczenie i tokenizacja
│   │
│   ├── features/                 # Feature extraction
│   │   ├── __init__.py
│   │   ├── text_features.py      # TF-IDF, word counts, etc.
│   │   └── style_features.py     # Sentence length, vocab richness
│   │
│   ├── models/                   # Implementacje modeli
│   │   ├── __init__.py
│   │   ├── baseline_model.py     # Linear SVM baseline
│   │   ├── tfidf_model.py        # TF-IDF + classifiers
│   │   ├── knn_model.py          # K-Nearest Neighbors
│   │   ├── ridge_model.py        # Ridge Classifier
│   │   ├── naive_bayes_model.py  # Multinomial Naive Bayes
│   │   ├── nearest_centroid_model.py  # Nearest Centroid
│   │   ├── style_model.py        # Style-based RF
│   │   ├── xgboost_model.py      # XGBoost
│   │   ├── lightgbm_model.py     # LightGBM
│   │   ├── feature_model.py      # Feature-based RF
│   │   └── ensemble_voting_model.py  # Ensemble voting
│   │
│   └── utils/                    # Narzędzia
│       ├── __init__.py
│       └── config.py             # Konfiguracja i parametry
│
├── scripts/                      # Skrypty wykonawcze
│   ├── download_books.py         # Pobieranie i preprocessing
│   ├── extract_features.py       # Ekstrakcja cech
│   ├── train_simple.py           # Trening podstawowych modeli
│   ├── train_all_models.py       # Trening wszystkich modeli
│   ├── train_without_knn.py      # Trening bez KNN (szybciej)
│   ├── test_all_models.py        # Testing + wizualizacje
│   ├── test_and_optimize.py      # Testing + optymalizacja
│   ├── test_model.py             # Interaktywny tester
│   └── visualize_results.py      # Generowanie raportów
│
├── data/                         # Dane
│   ├── raw/                      # Surowe książki (.txt)
│   ├── processed/                # Przetworzone dane
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── features.csv
│   └── metadata.csv              # Metadane książek
│
├── models_saved/                 # Wytrenowane modele
│   ├── linear_svm.pkl            # ⭐ Najlepszy model
│   ├── linear_svm_optimized.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer
│
├── results/                      # Wyniki i raporty
│   ├── all_models_results.csv    # Tabela wyników
│   ├── MODEL_RESULTS.md          # Kompletny raport
│   ├── model_comparison.png      # Wykres porównawczy
│   ├── confusion_matrix_*.png    # Confusion matrices
│   └── classification_report_*.txt  # Classification reports
│
├── docs/                         # Dokumentacja
│   └── MODEL_ALGORITHMS_EXPLAINED.md  # 📚 Szczegółowy opis algorytmów
│
├── README.md                     # Ten plik
└── requirements.txt              # Zależności Python
```

---

## 📚 Dokumentacja

### Główne Dokumenty

- **[MODEL_ALGORITHMS_EXPLAINED.md](docs/MODEL_ALGORITHMS_EXPLAINED.md)** - Szczegółowy opis wszystkich 12 algorytmów
  - Jak działają (z diagramami)
  - Matematyka za nimi
  - Zalety i wady
  - Kiedy którego użyć
  - Praktyczne wskazówki

- **[MODEL_RESULTS.md](results/MODEL_RESULTS.md)** - Kompletny raport z wyników
  - Executive summary
  - Porównanie wszystkich modeli
  - Analiza overfittingu
  - Szczegółowe metryki per gatunek
  - Rekomendacje

- **[MEMORY.md](.claude/projects/-home-kasia-ML-book-genre-classifier/memory/MEMORY.md)** - Session memory (wewnętrzne)
  - Status projektu
  - Kluczowe fakty
  - Naprawione bugi
  - Quick commands

### Kluczowe Skrypty

| Skrypt | Opis | Czas |
|--------|------|------|
| `train_simple.py` | Trening 4 podstawowych modeli | ~5 min |
| `train_all_models.py` | Trening wszystkich 12 modeli | ~30 min |
| `test_all_models.py` | Test + confusion matrices | ~2 min |
| `test_model.py` | Interaktywny tester | Interactive |
| `visualize_results.py` | Generowanie raportów | ~30 sek |

---

## 🔬 Metodologia

### Feature Engineering

**TF-IDF Features (5000 dimensi):**
- Unigrams i bigrams (1-2)
- Min document frequency: 3
- Max document frequency: 85%
- L2 normalization

**Style Features (26 cech):**
- Średnia długość zdania/słowa
- Bogactwo słownictwa
- Stosunek dialogów
- Liczba rozdziałów
- Interpunkcja i emoticons

### Algorytmy

**Proste i Szybkie:**
- Naive Bayes (NAJSZYBSZY)
- Logistic Regression (dobry baseline)
- Nearest Centroid (szybkie predykcje)

**Mocne i Dokładne:**
- Linear SVM (NAJLEPSZY)
- Random Forest (z regularizacją)
- XGBoost (state-of-the-art)

**Eksperymentalne:**
- KNN (wolny ale ciekawy)
- Style-based (analiza stylu)
- Ensemble (łączenie modeli)

---

## 📈 Poprawa Wyników

### Aktualne Wyzwania

1. **Fantasy/SciFi Confusion** (39 książek mylonych)
   - Rozwiązanie: Dodać genre-specific keywords, character names, world-building features

2. **Thriller słabo klasyfikowany** (25% F1)
   - Rozwiązanie: Zebrać więcej danych treningowych, użyć sentiment analysis

3. **Random Forest overfitting** (85% train → 36% test)
   - Rozwiązanie: ✅ Naprawione (max_depth=8, min_samples_leaf=10)

### Planowane Ulepszenia

- [ ] Word2Vec/FastText embeddings (semantic similarity)
- [ ] BERT fine-tuning (state-of-the-art NLP)
- [ ] Character-level features (names, places, items)
- [ ] Sentiment analysis features
- [ ] Book-specific metadata (publication year, length)
- [ ] Active learning (hard examples)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Project Gutenberg** - za darmowe książki
- **scikit-learn** - za excellent ML library
- **Anthropic Claude** - za pomoc w development

---

## 📧 Contact

- **GitHub**: [your-username](https://github.com/your-username)
- **Email**: your.email@example.com
- **Project Link**: [https://github.com/your-username/book-genre-classifier](https://github.com/your-username/book-genre-classifier)

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{book-genre-classifier-2026,
  author = {Your Name},
  title = {Book Genre Classifier: ML-based Literary Genre Classification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/book-genre-classifier}
}
```

---

**Made with ❤️ and Python 🐍**
