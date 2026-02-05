import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_FILE = DATA_DIR / "metadata.csv"

MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

GENRES = [
    "Adventure/Action",
    "Biography",
    "Mystery/Crime",
    "Science Fiction",
    "Historical Fiction",
    "Thriller/Horror",
    "Fantasy",
    "Romance",
]

BOOKS_PER_GENRE = 500
MIN_BOOK_LENGTH = 10000
MAX_BOOK_LENGTH = 500000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

REMOVE_GUTENBERG_HEADERS = True
CHAPTER_MARKERS = [
    "CHAPTER",
    "Chapter",
    "PART",
    "Part",
    "BOOK",
    "Book",
    "VOLUME",
    "Volume",
]

STANZA_LANGUAGE = "en"
STANZA_PROCESSORS = "tokenize,mwt,pos,lemma,ner"
STANZA_USE_GPU = False

MAX_SENTENCE_LENGTH = 100
MIN_SENTENCE_LENGTH = 1

MANUAL_KEYWORDS = {
    "Romance": [
        "love",
        "heart",
        "kiss",
        "passion",
        "romance",
        "wedding",
        "beloved",
        "embrace",
    ],
    "Fantasy": [
        "magic",
        "wizard",
        "dragon",
        "spell",
        "enchant",
        "quest",
        "realm",
        "sword",
    ],
    "Thriller/Horror": [
        "scream",
        "fear",
        "terror",
        "blood",
        "murder",
        "dark",
        "shadow",
        "death",
    ],
    "Historical Fiction": [
        "century",
        "war",
        "king",
        "queen",
        "empire",
        "ancient",
        "historical",
    ],
    "Science Fiction": [
        "space",
        "alien",
        "robot",
        "future",
        "technology",
        "galaxy",
        "ship",
        "planet",
    ],
    "Mystery/Crime": [
        "detective",
        "murder",
        "investigation",
        "clue",
        "suspect",
        "evidence",
        "crime",
    ],
    "Biography": [
        "born",
        "life",
        "early",
        "career",
        "death",
        "childhood",
        "education",
        "family",
    ],
    "Adventure/Action": [
        "adventure",
        "journey",
        "battle",
        "fight",
        "escape",
        "chase",
        "danger",
        "hero",
    ],
}

TOP_KEYWORDS_PER_GENRE = 100

TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "oob_score": True,
}

SVM_PARAMS = {"C": 1.0, "kernel": "linear", "random_state": RANDOM_STATE}

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

XGBOOST_EARLY_STOPPING_ROUNDS = 10

KNN_PARAMS = {
    "n_neighbors": 20,
    "metric": "cosine",
    "weights": "distance",
    "n_jobs": -1,
}

NAIVE_BAYES_PARAMS = {
    "alpha": 1.0,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

LIGHTGBM_EARLY_STOPPING_ROUNDS = 10

RIDGE_PARAMS = {
    "alpha": 1.0,
    "random_state": RANDOM_STATE,
}

NEAREST_CENTROID_PARAMS = {
    "metric": "cosine",
    "shrink_threshold": None,
}

CV_FOLDS = 5

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
