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

MAX_SENTENCE_LENGTH = 100
MIN_SENTENCE_LENGTH = 1

TOP_KEYWORDS_PER_GENRE = 100

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.85
TFIDF_NGRAM_RANGE = (1, 2)

LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(message)s"
# LOG_FORMAT = "%(levelname)s: %(message)s"
