# 📚 Klasyfikacja gaunków literackich

## Opis Projektu

Celem projektu jest automatyczne przypisywanie książkom gatunków literackich przy użyciu metod uczenia maszynowego i NLP (przetwarzanie języka naturalnego).

## Struktura projektu

```
book-genre-classifier/
├── src/                    # Kod źródłowy
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

- **Całość:** 4400 książek, po 400 dla każdego gatunku.
- Zbiór treningowy: 70%
- Zbiór walidacyjny: 15%
- Zbiór testowy: 15%

Dane zostały podzielone po autorach aby zapobiec data leakage.

## Komendy

```bash
# instalacja wymaganych bibliotek
python3 install_requirenments.py

# pobranie danych, preprocessing i podział danych
# pominięcie wybranego kroku: --skip_download --skip_preprocessing --skip_splitting
python3 src/download_books.py

# wyodrębnienie features
python3 src/features/extract_features.py # opcjonalnie: --skip_keywords --skip_statistical

python3 src/generate_vectorizer.py
python3 src/cache_vectorized_data.py

# trenowanie i testowanie modeli
python3 src/train_ultra_fast.py
```