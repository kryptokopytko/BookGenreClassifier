import re
import gc
from collections import Counter
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.logger import get_logger
from ..utils.config import TOP_KEYWORDS_PER_GENRE, GENRES

logger = get_logger(__name__)

class VocabularyAnalyzer:

    def __init__(self, use_stopwords: bool = True):
        try:
            self.stopwords = set(stopwords.words('english')) if use_stopwords else set()
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english')) if use_stopwords else set()

        self.stemmer = PorterStemmer()

    def tokenize(self, text: str, lowercase: bool = True) -> List[str]:
        text = text.lower() if lowercase else text
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t.lower() not in self.stopwords]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(t) for t in tokens]

    def get_word_frequencies(self, tokens: List[str]) -> Counter:
        return Counter(tokens)

    def get_top_words(self, tokens: List[str], n: int = 100) -> List[tuple]:
        freq = self.get_word_frequencies(tokens)
        return freq.most_common(n)

    def calculate_tfidf_scores(
        self,
        genre_documents: Dict[str, List[str]],
        top_n: int = TOP_KEYWORDS_PER_GENRE
    ) -> Dict[str, List[tuple]]:

        logger.info("Calculating TF-IDF scores for characteristic words")

        characteristic_words = {}

        for genre, documents in genre_documents.items():
            if not documents:
                logger.warning(f"No documents for genre {genre}")
                continue

            logger.info(f"Processing {genre} with {len(documents)} documents")

            doc_texts = [' '.join(doc) for doc in documents]

            vectorizer = TfidfVectorizer(
                max_features=top_n * 2,
                min_df=2,
                max_df=0.8,
                stop_words=list(self.stopwords)
            )

            try:
                tfidf_matrix = vectorizer.fit_transform(doc_texts)
                feature_names = vectorizer.get_feature_names_out()

                avg_scores = tfidf_matrix.mean(axis=0).A1
                word_scores = list(zip(feature_names, avg_scores))

                word_scores.sort(key=lambda x: x[1], reverse=True)

                characteristic_words[genre] = word_scores[:top_n]

                logger.info(f"{genre}: Top 10 words: {[w for w, s in word_scores[:10]]}")

                del doc_texts, tfidf_matrix, feature_names, avg_scores, word_scores
                gc.collect()

            except Exception as e:
                logger.error(f"Error calculating TF-IDF for {genre}: {e}")
                characteristic_words[genre] = []

        return characteristic_words

    def extract_genre_keywords(
        self,
        metadata_file: Path,
        stem_tokens: False
    ) -> Dict[str, List[str]]:
        df = pd.read_csv(metadata_file)

        logger.info("Processing books to extract keywords...")
        logger.info("Processing each genre separately to save memory")

        genre_keywords = {}

        for genre in GENRES:
            logger.info(f"\nProcessing genre: {genre}")

            genre_df = df[df['genre'] == genre]
            logger.info(f"  Found {len(genre_df)} books")

            if len(genre_df) == 0:
                logger.warning(f"  No books found for {genre}")
                genre_keywords[genre] = []
                continue

            documents = []

            for idx, row in tqdm(genre_df.iterrows(), total=len(genre_df), desc=f"  Processing {genre}"):
                try:
                    text_path = Path(row['processed_path'])
                    if not text_path.is_absolute():
                        from ..utils.config import PROCESSED_DATA_DIR
                        text_path = PROCESSED_DATA_DIR.parent / text_path

                    text = text_path.read_text(encoding='utf-8')

                    tokens = self.tokenize(text)
                    tokens = self.remove_stopwords(tokens)

                    if stem_tokens:
                        tokens = self.stem_tokens(tokens)

                    documents.append(tokens)

                    del text, tokens

                except Exception as e:
                    logger.error(f"  Error processing book {row['book_id']}: {e}")
                    continue

            if documents:
                genre_documents = {genre: documents}
                characteristic_words = self.calculate_tfidf_scores(genre_documents)

                if genre in characteristic_words:
                    genre_keywords[genre] = [word for word, score in characteristic_words[genre]]
                else:
                    genre_keywords[genre] = []

                del documents, genre_documents, characteristic_words
                gc.collect()
            else:
                genre_keywords[genre] = []

        return genre_keywords

    def get_all_keywords(
        self,
        metadata_file: Path = None,
        genres = GENRES
    ) -> Dict[str, List[str]]:
        from collections import defaultdict
        all_keywords = defaultdict(list)

        if metadata_file is None:
            raise ValueError("metadata_file required for automatic keyword extraction")

        automatic_keywords = self.extract_genre_keywords(metadata_file, genres=genres)

        for genre in GENRES:
            if genre in automatic_keywords:
                all_keywords[genre].extend(automatic_keywords[genre])

        for genre in GENRES:
            all_keywords[genre] = list(set(all_keywords[genre]))

        for genre in GENRES:
            logger.info(f"{genre}: {len(all_keywords[genre])} total keywords")

        return all_keywords

    def create_keyword_features(
        self,
        text: str,
        keywords_dict: Dict[str, List[str]],
        normalize: bool = True
    ) -> Dict[str, float]:
        tokens = self.tokenize(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        features = {}

        for genre, keywords in keywords_dict.items():
            count = sum(token_counts[keyword] for keyword in keywords)

            if normalize and total_tokens > 0:
                features[f'keyword_{genre.replace("/", "_")}'] = count / total_tokens
            else:
                features[f'keyword_{genre.replace("/", "_")}'] = count

        return features

def extract_and_save_keywords(
    metadata_file: Path,
    output_file: Path,
    genres = GENRES
) -> Dict[str, List[str]]:

    analyzer = VocabularyAnalyzer()

    keywords = analyzer.get_all_keywords(
        metadata_file=metadata_file,
        genres=genres
    )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix == '.json':
        with open(output_file, 'w') as f:
            json.dump(keywords, f, indent=2)
    else:

        rows = []
        for genre, words in keywords.items():
            for word in words:
                rows.append({'genre': genre, 'keyword': word})
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)

    logger.info(f"Keywords saved to {output_file}")

    return keywords
