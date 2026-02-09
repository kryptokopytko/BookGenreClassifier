from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_FILE = DATA_DIR / "metadata.csv"

MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_PER_SITE = 25

GENRES = [
    "Adventure",
    "Biographies",
    "Poetry",
    "Romance",
    "Science-Fiction & Fantasy",
    "Crime, Thrillers & Mystery",
    "Children & Young Adult Reading",
    "Engineering & Technology",
    "History - Other",
    "Politics",
    "Cooking & Drinking",
]

BOOKS_PER_GENRE = 400
MIN_BOOK_LENGTH = 3000
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
    "Poetry": [
        "love",
        "heart",
        "nature",
        "dream",
        "soul",
        "life",
        "emotion",
        "beauty",
    ],
    "Science-Fiction & Fantasy": [
        "magic",
        "wizard",
        "dragon",
        "spell",
        "enchant",
        "quest",
        "realm",
        "sword",
    ],
    "Crime, Thrillers & Mystery": [
        "fear",
        "blood",
        "murder",
        "dark",
        "death",
        "detective",
        "suspect",
        "evidence",
    ],
    "Children & Young Adult Reading": [
        "friend",
        "school",
        "adventure",
        "magic",
        "family",
        "play",
        "journey",
        "fun",
    ],
    "Engineering & Technology": [
        "machine",
        "engineer",
        "design",
        "technology",
        "experiment",
        "system",
        "robot",
        "innovation",
    ],
    "History - Other": [
        "war",
        "king",
        "queen",
        "empire",
        "battle",
        "revolution",
        "century",
        "ancient",
    ],
    "Politics": [
        "government",
        "policy",
        "law",
        "election",
        "party",
        "power",
        "leader",
        "democracy",
    ],
    "Cooking & Drinking": [
        "recipe",
        "cook",
        "kitchen",
        "flavor",
        "ingredient",
        "dish",
        "meal",
        "drink",
    ],
    "Biographies": [
        "born",
        "life",
        "early",
        "career",
        "death",
        "childhood",
        "education",
        "family",
    ],
    "Adventure": [
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

LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(message)s"
# LOG_FORMAT = "%(levelname)s: %(message)s"
