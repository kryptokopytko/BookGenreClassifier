# 📚 Przewodnik po Algorytmach Klasyfikacji - Jak Działają Modele

---

## 📖 Spis Treści

1. [Wprowadzenie](#wprowadzenie)
2. [Modele Oparte na Tekście (TF-IDF)](#modele-oparte-na-tekście-tf-idf)
   - Linear SVM
   - Logistic Regression
   - Naive Bayes
   - Ridge Classifier
   - K-Nearest Neighbors (KNN)
   - Nearest Centroid
3. [Modele Oparte na Cechach](#modele-oparte-na-cechach)
   - Random Forest
   - XGBoost
   - LightGBM
4. [Modele Hybrydowe](#modele-hybrydowe)
   - Style-based Model
   - Baseline Keyword Model
5. [Ensemble Models](#ensemble-models)
6. [Porównanie Algorytmów](#porównanie-algorytmów)
7. [Kiedy Którego Użyć](#kiedy-którego-użyć)

---

## Wprowadzenie

W tym projekcie używamy różnych algorytmów uczenia maszynowego do klasyfikacji gatunków książek. Każdy algorytm ma inne podejście do problemu i działa lepiej w różnych sytuacjach.

### Podstawowe Pojęcia

**TF-IDF (Term Frequency-Inverse Document Frequency)**

- Sposób reprezentacji tekstu jako liczb
- **TF**: Jak często słowo pojawia się w dokumencie
- **IDF**: Jak rzadkie jest słowo w całym korpusie
- Ważne słowa mają wysokie wartości, częste słowa ("the", "is") niskie

**Feature Engineering**

- Wyciąganie użytecznych informacji z danych
- Np. liczba słów, średnia długość zdania, stosunek dialogów

**Overfitting**

- Model "nauczył się na pamięć" dane treningowe
- Działa świetnie na treningu, źle na testach
- Rozwiązanie: regularyzacja, mniej parametrów

---

## Modele Oparte na Tekście (TF-IDF)

### 1. 🎯 Linear SVM (Support Vector Machine)

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Reprezentuj teksty jako    │
│  wektory TF-IDF (15000 wymiarów)    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Znajdź hiperpłaszczyznę    │
│  która najlepiej separuje klasy     │
│                                      │
│    Romance  │  Mystery               │
│      •      │    •                   │
│    •  •     │  •   •                │
│      •    [GRANICA]  •              │
│              │    •                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Nowy tekst → po której     │
│  stronie granicy leży?              │
└─────────────────────────────────────┘
```

**Matematyka (uproszczona):**

- SVM szuka granicy (hiperpłaszczyzny) która maksymalizuje margines między klasami
- Dla punktu x, decyzja: `sign(w·x + b)`
- `w` = wektor wag (nauczony)
- `b` = bias (nauczony)

**Zalety:**

- ✅ Świetnie działa na wysokowymiarowych danych (tekst)
- ✅ Odporny na overfitting
- ✅ Szybkie predykcje
- ✅ Teoretyczne podstawy (maksymalizacja marginesu)

**Wady:**

- ❌ Długi czas treningu dla dużych zbiorów
- ❌ Trudno interpretować (nie wiadomo "dlaczego")
- ❌ Potrzebuje dobrego skalowania danych

**Kiedy użyć:**

- Dane tekstowe z TF-IDF
- Zależy Ci na accuracy
- Masz więcej cech niż sampli

**W naszym projekcie:**

- **Accuracy: 72.6%** (najlepszy model!)
- Używa 15000 cech TF-IDF
- Kernel: liniowy (najszybszy dla tekstu)

---

### 2. 📊 Logistic Regression

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: TF-IDF representation      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Dla każdej klasy,         │
│  oblicz prawdopodobieństwo:         │
│                                      │
│  P(Romance|text) = σ(w₁·x + b₁)    │
│  P(Mystery|text) = σ(w₂·x + b₂)    │
│  ...                                │
│                                      │
│  σ = sigmoid (0-1)                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Wybierz klasę z highest   │
│  prawdopodobieństwem                │
└─────────────────────────────────────┘
```

**Matematyka:**

- Sigmoid function: `σ(z) = 1/(1 + e^(-z))`
- Dla multi-class: softmax
- `P(class_i|x) = exp(w_i·x) / Σ exp(w_j·x)`

**Zalety:**

- ✅ Daje prawdopodobieństwa (nie tylko klasy)
- ✅ Szybki trening i predykcja
- ✅ Łatwo interpretować wagi
- ✅ Regularizacja (L1/L2) zapobiega overfittingowi

**Wady:**

- ❌ Zakłada liniową separowalność
- ❌ Może być zbyt prosty dla złożonych wzorców

**Kiedy użyć:**

- Potrzebujesz prawdopodobieństw
- Chcesz zrozumieć, które słowa są ważne
- Baseline model (zawsze zacznij od tego!)

**W naszym projekcie:**

- **Accuracy: 65.5%**
- Regularizacja: C=2.0 (mniej restrykcyjna)
- Solver: SAGA (dobry dla dużych danych)

---

### 3. 🎲 Naive Bayes (MultinomialNB)

**Jak działa:**

```
┌─────────────────────────────────────┐
│  Bayes Theorem:                     │
│  P(Genre|Words) =                   │
│    P(Words|Genre) × P(Genre)        │
│    ─────────────────────────        │
│         P(Words)                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  "Naive" assumption:                │
│  Słowa są niezależne!               │
│                                      │
│  P(w₁,w₂,...|Genre) =              │
│    P(w₁|Genre) × P(w₂|Genre) × ... │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Dla nowego tekstu:                 │
│  Oblicz P(Genre|words) dla każdego │
│  gatunku → wybierz max             │
└─────────────────────────────────────┘
```

**Intuicja:**

- Model "pamięta" jakie słowa występują w każdym gatunku
- Np. w Romance: "love"=często, "murder"=rzadko
- Nowy tekst z "love" → prawdopodobnie Romance

**Zalety:**

- ✅ BARDZO szybki (zarówno trening jak predykcja)
- ✅ Działa dobrze na małych zbiorach
- ✅ Prosty i interpretowalny
- ✅ Naturalnie obsługuje multi-class

**Wady:**

- ❌ Naiwne założenie (słowa NIE są niezależne!)
- ❌ Wrażliwy na dane spoza treningu (smoothing pomaga)
- ❌ Nie uczy się interakcji między słowami

**Kiedy użyć:**

- Mały zbiór danych
- Potrzebujesz SZYBKOŚCI
- Baseline model
- Spam detection, sentiment analysis

**W naszym projekcie:**

- **Accuracy: 59.4%**
- Alpha=1.0 (Laplace smoothing)
- Dobry jako szybki baseline

---

### 4. 📏 Ridge Classifier

**Jak działa:**

```
┌─────────────────────────────────────┐
│  To Logistic Regression ale z       │
│  regularizacją L2 (Ridge)           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Minimize: Loss + α × ||w||²       │
│                                      │
│  ||w||² = suma kwadratów wag       │
│  α = siła regularyzacji            │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Efekt: Małe wagi → mniej          │
│  overfittingu → lepsze generalization│
└─────────────────────────────────────┘
```

**Matematyka:**

- Loss function: `L = (y - ŷ)² + α·Σw²`
- Regularyzacja "karze" duże wagi
- α=0 → no regularization
- α=∞ → wszystkie wagi → 0

**Zalety:**

- ✅ Bardzo odporny na overfitting
- ✅ Działa gdy features > samples
- ✅ Stabilny numerycznie
- ✅ Szybki

**Wady:**

- ❌ Nie robi feature selection (wszystkie features mają wagi)
- ❌ Mniej elastyczny niż modele nieliniowe

**Kiedy użyć:**

- Dużo cech (high-dimensional)
- Problem z overfittingiem
- Stabilne predykcje ważniejsze niż max accuracy

**W naszym projekcie:**

- Alpha=1.0 (standardowa regularizacja)
- Dobra alternatywa dla Logistic Regression

---

### 5. 👥 K-Nearest Neighbors (KNN)

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Zapisz wszystkie dane     │
│  treningowe (no training!)          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Nowy tekst → znajdź K     │
│  najbliższych sąsiadów              │
│                                      │
│      ?                               │
│     / \                             │
│    •1 •2  (K=3)                    │
│      •3                             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Głosowanie                 │
│  Sąsiad 1: Romance                  │
│  Sąsiad 2: Romance                  │
│  Sąsiad 3: Mystery                  │
│  → Predykcja: Romance (2/3)        │
└─────────────────────────────────────┘
```

**Metryki odległości:**

- **Cosine similarity** (używamy): kąt między wektorami
  - Dobra dla tekstu (niezależna od długości)
  - `similarity = (A·B)/(||A||×||B||)`
- **Euclidean**: zwykła odległość
- **Manhattan**: suma różnic bezwzględnych

**Zalety:**

- ✅ Prosty koncepcyjnie
- ✅ Brak fazy treningu
- ✅ Może uchwycić złożone granice decyzyjne
- ✅ Naturalnie multi-class

**Wady:**

- ❌ WOLNE predykcje (musi porównać ze wszystkimi samples)
- ❌ Wrażliwy na irrelevant features
- ❌ Potrzebuje dużo pamięci
- ❌ Curse of dimensionality

**Kiedy użyć:**

- Mały dataset
- Nieregularne granice klas
- Nie ma czasu na tuning
- Chcesz "explainable" predictions (pokaż sąsiadów)

**W naszym projekcie:**

- K=20 neighbors
- Metric: cosine
- Weights: distance (bliżsi ważniejsi)

---

### 6. 🎯 Nearest Centroid

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Oblicz centroid (średnią)  │
│  dla każdej klasy                   │
│                                      │
│  Centroid_Romance = średnia wszystkich│
│                     tekstów Romance  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Reprezentuj centroids      │
│  jako wektory TF-IDF                │
│                                      │
│    C₁ (Romance)                     │
│       ★                             │
│           ★ C₂ (Mystery)           │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Nowy tekst → który        │
│  centroid jest najbliżej?           │
│                                      │
│    C₁        ?        C₂           │
│     ★        •         ★           │
│        d₁ < d₂                     │
│    → Predykcja: C₁ (Romance)       │
└─────────────────────────────────────┘
```

**Intuicja:**

- Każdy gatunek ma "prototypowy" dokument (centroid)
- Nowy tekst → który prototyp jest najbardziej podobny?

**Zalety:**

- ✅ BARDZO szybkie predykcje (tylko N porównań, nie N×samples)
- ✅ Mało pamięci (tylko centroids)
- ✅ Prosty i interpretowalny
- ✅ Można zobaczyć "typowe słowa" dla każdego gatunku

**Wady:**

- ❌ Zakłada, że klasy są "kuliste" (convex)
- ❌ Wrażliwy na outliery
- ❌ Nie uczy się złożonych granic

**Kiedy użyć:**

- Potrzebujesz SZYBKOŚCI (production)
- Mało pamięci
- Klasy są dobrze separowane
- Chcesz zrozumieć "typowy" dokument każdej klasy

**W naszym projekcie:**

- Metric: euclidean (cosine nie działa w sklearn)
- Shrink_threshold: None (no shrinkage)

---

## Modele Oparte na Cechach

### 7. 🌳 Random Forest

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Zbuduj wiele drzew        │
│  decyzyjnych (forest)               │
│                                      │
│  Drzewo 1      Drzewo 2   Drzewo N │
│     🌲           🌲          🌲     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Każde drzewo głosuje      │
│                                      │
│  Tree 1: Romance                    │
│  Tree 2: Romance                    │
│  Tree 3: Mystery                    │
│  ...                                │
│  Tree 200: Romance                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Majority vote             │
│  → Final prediction: Romance        │
└─────────────────────────────────────┘
```

**Jak budować drzewa:**

```
                Root
                 │
         [avg_word_len > 5.2?]
               /    \
             Yes     No
             │       │
      [dialogue_ratio>0.3?]  [Chapter count>10?]
         /    \               /    \
     Romance Mystery      SciFi  History
```

**Randomizacja:**

1. **Bagging**: Każde drzewo trenuje na losowym subsecie danych
2. **Feature sampling**: Każdy split patrzy na losowy subset cech

**Zalety:**

- ✅ Bardzo mocny (często top performance)
- ✅ Obsługuje nieliniowe zależności
- ✅ Feature importance (które cechy ważne)
- ✅ Nie potrzebuje skalowania
- ✅ Odporny na outliery

**Wady:**

- ❌ Może overfittować (jak w naszym projekcie!)
- ❌ Wolniejszy niż linear models
- ❌ Trudniej interpretować
- ❌ Duży rozmiar modelu

**Jak zapobiec overfittingowi:**

```python
# PRZED (overfitting)
max_depth=15          # Zbyt głębokie drzewa
min_samples_split=10  # Zbyt małe

# PO (lepsze)
max_depth=8           # Płytsze drzewa
min_samples_split=20  # Więcej sampli na split
min_samples_leaf=10   # Min sampli w liściu
max_features='sqrt'   # Mniej cech na split
n_estimators=200      # Więcej drzew
```

**Kiedy użyć:**

- Masz dużo różnych typów cech
- Nieliniowe zależności
- Feature importance jest ważna
- Możesz poświęcić trochę czasu na trening

**W naszym projekcie:**

- **OLD: 100% train → 55.8% test** (OVERFITTING!)
- **NEW (po fix): spodziewamy się ~65-70%**
- 200 drzew, max_depth=8

---

### 8. 🚀 XGBoost (eXtreme Gradient Boosting)

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Zbuduj drzewo #1          │
│  (próbuje predyktować labels)       │
│                                      │
│     Predictions₁ = [0.3, 0.7, ...]  │
│     Errors₁ = y - Predictions₁      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Zbuduj drzewo #2          │
│  (próbuje predyktować ERRORS₁!)     │
│                                      │
│     Predictions₂ = [0.1, -0.2, ...] │
│     Errors₂ = Errors₁ - Predictions₂│
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Powtarzaj dla N drzew     │
│                                      │
│  Final = Σ (learning_rate × Tree_i) │
└─────────────────────────────────────┘
```

**Gradient Boosting:**

- Każde nowe drzewo "naprawia" błędy poprzednich
- Gradient descent w przestrzeni funkcji!
- Learning rate kontroluje jak "mocno" poprawiamy

**Zalety:**

- ✅ Stan-of-the-art performance
- ✅ Obsługuje missing values
- ✅ Wbudowana regularyzacja
- ✅ Szybki (parallelizacja)
- ✅ Feature importance

**Wady:**

- ❌ Dużo hiperparametrów do tuningu
- ❌ Może overfittować bez early stopping
- ❌ Trudny do interpretacji
- ❌ Potrzebuje dobrego tuningu

**Hiperparametry:**

- `n_estimators`: liczba drzew (więcej = lepiej, ale wolniej)
- `max_depth`: głębokość drzew (mniej = mniej overfitting)
- `learning_rate`: jak "mocno" uczymy (mniej = bezpieczniej)
- `subsample`: % sampli na drzewo (80% = więcej diversity)

**Kiedy użyć:**

- Konkursy ML (Kaggle)
- Tabularne dane (cechy numeryczne)
- Masz czas na tuning
- Chcesz najlepszej accuracy

**W naszym projekcie:**

- 100 drzew, max_depth=6
- learning_rate=0.1
- subsample=0.8

---

### 9. ⚡ LightGBM (Light Gradient Boosting Machine)

**Jak działa:**

```
Similar do XGBoost, ale z optymalizacjami:

┌─────────────────────────────────────┐
│  1. Leaf-wise growth (not level)    │
│                                      │
│     XGBoost:        LightGBM:       │
│        ▲               ▲            │
│       / \             / \           │
│      /   \           /   \          │
│     /     \         /     \         │
│    (równo) (głębiej tam gdzie gain) │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. Histogram-based splits          │
│  (grupuje cechy → szybsze)          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Efekt: 10-20x szybszy od XGBoost! │
└─────────────────────────────────────┘
```

**Zalety:**

- ✅ BARDZO szybki
- ✅ Mało pamięci
- ✅ Podobna accuracy do XGBoost
- ✅ Dobry dla dużych datasets

**Wady:**

- ❌ Może overfittować na małych danych
- ❌ Wrażliwy na parametry

**Kiedy użyć:**

- Duży dataset (>10K sampli)
- Potrzebujesz szybkości
- Tabularne dane

**W naszym projekcie:**

- Podobne parametry do XGBoost
- num_leaves=31 (max liści)

---

## Modele Hybrydowe

### 10. ✍️ Style-based Model

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Extract style features     │
│  from text                          │
│                                      │
│  • avg_sentence_length              │
│  • avg_word_length                  │
│  • vocabulary_richness              │
│  • dialogue_ratio                   │
│  • punctuation_patterns             │
│  • capitalization_rate              │
│  • ...26 features total             │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Train Random Forest        │
│  on style features                  │
└─────────────────────────────────────┘
```

**Intuicja:**

- Różne gatunki mają różny styl pisania
- Romance: krótsze zdania, więcej wykrzykników, więcej dialogów
- SciFi: dłuższe słowa, bardziej technical vocab
- Biography: konkretne daty, imiona, fakty

**Zalety:**

- ✅ Nie potrzebuje dużego słownika
- ✅ Szybki (tylko 26 cech)
- ✅ Interpretowalne cechy

**Wady:**

- ❌ Gubi semantykę (treść)
- ❌ Słabszy niż TF-IDF models

**Kiedy użyć:**

- Jako dodatkowy model w ensemble
- Analiza stylu autora
- Gdy semantyka nie wystarcza

---

### 11. 🔑 Baseline Keyword Model

**Jak działa:**

```
┌─────────────────────────────────────┐
│  KROK 1: Define keywords per genre  │
│                                      │
│  Romance: ["love", "kiss", "heart"] │
│  Mystery: ["murder", "detective"]   │
│  SciFi: ["space", "alien", "robot"] │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 2: Count keyword occurrences  │
│                                      │
│  Text: "She loved the detective..."│
│  Romance_score = 1 (love)           │
│  Mystery_score = 1 (detective)      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  KROK 3: Logistic Regression on    │
│  keyword counts                     │
└─────────────────────────────────────┘
```

**Zalety:**

- ✅ Bardzo prosty
- ✅ Interpretowalny
- ✅ Szybki
- ✅ Dobry baseline

**Wady:**

- ❌ Wymaga ręcznego wyboru keywords
- ❌ Nie uczy się automatycznie
- ❌ Słabszy od TF-IDF

---

## Ensemble Models

### 12. 🤝 Ensemble Voting

**Soft Voting:**

```
┌─────────────────────────────────────┐
│  Model 1 (SVM):                     │
│    Romance: 0.7, Mystery: 0.2, ...  │
│                                      │
│  Model 2 (LogReg):                  │
│    Romance: 0.6, Mystery: 0.3, ...  │
│                                      │
│  Model 3 (RF):                      │
│    Romance: 0.8, Mystery: 0.1, ...  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Average probabilities:             │
│    Romance: (0.7+0.6+0.8)/3 = 0.70 │
│    Mystery: (0.2+0.3+0.1)/3 = 0.20 │
│  → Prediction: Romance              │
└─────────────────────────────────────┘
```

**Hard Voting:**

```
┌─────────────────────────────────────┐
│  Model 1: Romance                   │
│  Model 2: Romance                   │
│  Model 3: Mystery                   │
│  Model 4: Romance                   │
│  Model 5: Romance                   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Majority vote: Romance (4/5)       │
└─────────────────────────────────────┘
```

**Weighted Voting:**

```
Model 1 (best): weight = 0.5
Model 2: weight = 0.3
Model 3: weight = 0.2

Final = 0.5×Pred₁ + 0.3×Pred₂ + 0.2×Pred₃
```

**Zalety:**

- ✅ Prawie zawsze lepszy niż single model
- ✅ Bardziej stabilny (robust)
- ✅ Łączy różne "spojrzenia" na problem

**Wady:**

- ❌ Wolniejszy (N modeli)
- ❌ Więcej pamięci

**Kiedy użyć:**

- Production (najważniejsza accuracy)
- Różnorodne modele (SVM + RF + NB lepsze niż 3× SVM)
- Masz zasoby obliczeniowe

---

## Porównanie Algorytmów

### Performance vs Complexity

```
High Performance ↑
                 │
            ⭐SVM│                  🌳XGBoost
                 │         ⚡LightGBM
                 │    📊LogReg
                 │               🌲RF
                 │   🎲NB
                 │      👥KNN
                 │
                 └─────────────────────────→
                   Simple        Complex
```

### Training Time

```
Fast  ◀────────────────────────▶  Slow
      │                          │
   🎲NB  🎯Centroid  📊LogReg  📏Ridge  ⭐SVM  👥KNN  🌲RF  🌳XGBoost
```

### Interpretability

```
Easy to Understand  ◀──────────▶  Black Box
                    │            │
    🎲NB  🔑Keywords  📊LogReg  📏Ridge  ⭐SVM  🌲RF  🌳XGBoost  ⚡LightGBM
```

### Memory Usage

```
Low Memory  ◀───────────────▶  High Memory
            │                 │
  🎯Centroid  ⭐SVM  📊LogReg  🎲NB  📏Ridge  👥KNN  🌲RF  🌳XGBoost
```

---

## Kiedy Którego Użyć

### 🎯 Chcę najlepszej accuracy (competition)

1. **XGBoost** / **LightGBM** - extensive tuning
2. **Ensemble** z top 3-5 modeli
3. **Linear SVM** - jeśli text-based

### ⚡ Potrzebuję szybkości (production)

1. **Nearest Centroid** - najszybszy
2. **Naive Bayes** - bardzo szybki
3. **Linear SVM** - szybkie predykcje

### 🔍 Chcę interpretowalności

1. **Logistic Regression** - wagi dla każdego słowa
2. **Naive Bayes** - prawdopodobieństwa słów
3. **Decision Trees** - ścieżka decyzyjna

### 📦 Mały dataset (<1000 sampli)

1. **Naive Bayes** - działa na małych danych
2. **Logistic Regression** z regularyzacją
3. **KNN** - no training needed

### 🎨 Różnorodne typy cech

1. **Random Forest** - nie potrzebuje skalowania
2. **XGBoost** - obsługuje różne typy
3. **LightGBM** - szybki na mixed features

### 🏁 Pierwszy model (baseline)

1. **Logistic Regression** - zawsze zacznij tutaj
2. **Naive Bayes** - szybki baseline
3. **Linear SVM** - jeśli LogReg nie wystarcza

---

## Praktyczne Wskazówki

### 🎓 Workflow dla Nowego Projektu

```
KROK 1: Quick baselines (1 dzień)
  ├─ Naive Bayes
  ├─ Logistic Regression
  └─ See what accuracy is possible

KROK 2: Try stronger models (2-3 dni)
  ├─ Linear SVM
  ├─ Random Forest
  └─ XGBoost

KROK 3: Hyperparameter tuning (3-5 dni)
  ├─ Grid search na top 2-3 modelach
  └─ Cross-validation

KROK 4: Ensemble (1 dzień)
  └─ Combine best models

KROK 5: Production optimization
  ├─ Speed vs accuracy tradeoff
  └─ Deploy simplest model that meets requirements
```

### ⚠️ Częste Błędy

1. **Zaczynanie od XGBoost**

   - ❌ Złe: "XGBoost jest najlepszy, zacznę od niego"
   - ✅ Dobre: Zacznij od prostych modeli → zrozum dane → potem XGBoost

2. **Nie sprawdzanie overfittingu**

   - ❌ Złe: "100% train accuracy! Super!"
   - ✅ Dobre: Zawsze porównaj train vs test accuracy

3. **Ignorowanie baseline**

   - ❌ Złe: Pomijanie Logistic Regression
   - ✅ Dobre: LogReg mówi Ci czy problem jest łatwy czy trudny

4. **Złe metryki**

   - ❌ Złe: Patrzenie tylko na accuracy (niezbalansowane klasy)
   - ✅ Dobre: F1 score, confusion matrix, per-class metrics

5. **Nie testowanie na nowych danych**
   - ❌ Złe: Test na tym samym zbiorze
   - ✅ Dobre: Hold-out test set OR cross-validation

---

## Podsumowanie

### Top 3 dla Text Classification:

1. **🥇 Linear SVM** - best accuracy, fast predictions
2. **🥈 Logistic Regression** - probabilites, interpretable
3. **🥉 XGBoost** - with feature engineering

### Top 3 dla Quick Prototyping:

1. **🥇 Naive Bayes** - fastest
2. **🥈 Logistic Regression** - good baseline
3. **🥉 Nearest Centroid** - simple and fast

### Top 3 dla Production:

1. **🥇 Ensemble** (SVM + LogReg + XGBoost)
2. **🥈 Linear SVM** - single model
3. **🥉 Logistic Regression** - interpretable

---

## Dodatkowe Źródła

### Książki:

- "Hands-On Machine Learning" - Aurélien Géron
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman

### Kursy Online:

- Andrew Ng - Machine Learning (Coursera)
- Fast.ai - Practical Deep Learning
- Scikit-learn documentation

### Papers:

- SVM: "Support Vector Networks" (Cortes & Vapnik, 1995)
- XGBoost: "XGBoost: A Scalable Tree Boosting System" (Chen & Guestrin, 2016)
- Random Forest: "Random Forests" (Breiman, 2001)

---

**Pytania? Sugestie? Issues?**
https://github.com/your-username/book-genre-classifier

**Autor:** Kasia
**Ostatnia aktualizacja:** 2026-02-09
