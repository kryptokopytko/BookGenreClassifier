# 📚 Klasyfikacja gaunków literackich

## Opis Projektu

Celem projektu jest automatyczne przypisywanie książkom gatunków literackich przy użyciu metod uczenia maszynowego i NLP (przetwarzanie języka naturalnego).

## Struktura projektu

```
book-genre-classifier/
├── src/                    # Kod źródłowy (data, features, models, evaluation)
├── scripts/                # Skrypty treningowe
├── data/                   # raw/, processed/, metadata.csv
├── models_saved/           # Wytrenowane modele (.pkl)
└── results/                # Wyniki i metryki
```

## Dane

**Źródło:** Project Gutenberg (darmowe e-booki)

**Gatunki książek**
- Adventure,
- Biographies,
- Poetry,
- Romance,
- Science-Fiction & Fantasy,
- Crime, Thrillers & Mystery,
- Children & Young Adult Reading,
- Engineering & Technology,
- History - Other,
- Politics,
- Cooking & Drinking

**Rozmiar danych:**
- **Całość:**  4400 książek, po 400 dla każdego gatunku.
- Zbiór treningowy:  70%
- Zbiór walidacyjny:  15%
- Zbiór testowy:  15%

Dane zostały podzielone po autorach aby zapobiec data leakage.

### Modele
- **Linear SVM**
- **Logistic Regression**
- **Random Forest**
- **Naive Bayes**

## Komendy

```bash
# instalacja wymaganych bibliotek
python3 scripts/install_requirenments.py

# pobranie danych, preprocessing i podział danych
# pominięcie wybranego kroku: --skip_download --skip_preprocessing --skip_splitting
python3 scripts/download_books.py

# wyodrębnienie features
python3 scripts/extract_features.py --skip_pos --skip_keywords

# trenowanie modeli
python3 scripts/train_simple.py

# testowanie modeli
python3 scripts/test_and_optimize.py
```

## Wyniki

### Wytrenowane Modele (Dataset: 3,703 książki)

| Model                   | Test Acc  | Precision | Recall | F1 Score |
| ----------------------- | --------- | --------- | ------ | -------- |
| **Linear SVM (C=10)**   | %     |           |        |          |
| **Linear SVM (C=1)**    | %     |           |        |          |
| **Logistic Regression** | %     |           |        |          |
| **Random Forest**       | %     |           |        |          |
| **Naive Bayes**         | %     |           |        |          |
