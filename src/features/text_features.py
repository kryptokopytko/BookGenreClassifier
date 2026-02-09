import re
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import Counter
import pandas as pd
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config import PROCESSED_DATA_DIR

logger = get_logger(__name__)

class TextFeatureExtractor:

    def __init__(self):
        pass

    def extract_sentence_features(self, sentence_lengths: List[int]) -> Dict[str, float]:
        if not sentence_lengths:
            return {
                'avg_sentence_length': 0,
                'std_sentence_length': 0,
                'median_sentence_length': 0,
                'pct_short_sentences': 0,
                'pct_long_sentences': 0,
                'max_sentence_length': 0,
                'min_sentence_length': 0
            }

        lengths = np.array(sentence_lengths)

        features = {
            'avg_sentence_length': float(np.mean(lengths)),
            'std_sentence_length': float(np.std(lengths)),
            'median_sentence_length': float(np.median(lengths)),
            'pct_short_sentences': 100 * np.sum(lengths < 5) / len(lengths),
            'pct_long_sentences': 100 * np.sum(lengths > 30) / len(lengths),
            'max_sentence_length': float(np.max(lengths)),
            'min_sentence_length': float(np.min(lengths))
        }

        return features

    def extract_word_features(self, text: str, lemmas: List[str] = None) -> Dict[str, float]:
        words = text.split()

        if not words:
            return {
                'avg_word_length': 0,
                'std_word_length': 0,
                'type_token_ratio': 0,
                'pct_hapax_legomena': 0,
                'pct_unique_words': 0,
                'lexical_density': 0,
                'total_words': 0,
                'unique_words': 0
            }

        word_lengths = [len(w) for w in words]

        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0

        word_counts = Counter(words)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        pct_hapax = 100 * hapax_count / unique_words if unique_words > 0 else 0

        if lemmas:
            unique_lemmas = len(set(lemmas))
            pct_unique_lemmas = 100 * unique_lemmas / len(lemmas) if lemmas else 0
        else:
            unique_lemmas = unique_words
            pct_unique_lemmas = 100 * ttr

        content_words = sum(1 for w in words if len(w) > 3)
        lexical_density = content_words / total_words if total_words > 0 else 0

        features = {
            'avg_word_length': float(np.mean(word_lengths)),
            'std_word_length': float(np.std(word_lengths)),
            'type_token_ratio': ttr,
            'pct_hapax_legomena': pct_hapax,
            'pct_unique_words': 100 * ttr,
            'lexical_density': lexical_density,
            'total_words': total_words,
            'unique_words': unique_words,
            'pct_unique_lemmas': pct_unique_lemmas
        }

        return features

    def extract_chapter_features(self, text: str) -> Dict[str, float]:
        chapter_pattern = r'^(CHAPTER|Chapter|PART|Part|BOOK|Book)\s+([IVXLCDM\d]+)'

        lines = text.split('\n')
        chapter_positions = []

        for i, line in enumerate(lines):
            if re.match(chapter_pattern, line.strip()):
                chapter_positions.append(i)

        num_chapters = len(chapter_positions)

        if num_chapters == 0:
            return {
                'num_chapters': 0,
                'avg_chapter_length': 0,
                'std_chapter_length': 0
            }

        chapter_lengths = []
        for i in range(len(chapter_positions)):
            if i < len(chapter_positions) - 1:
                length = chapter_positions[i + 1] - chapter_positions[i]
            else:
                length = len(lines) - chapter_positions[i]
            chapter_lengths.append(length)

        features = {
            'num_chapters': num_chapters,
            'avg_chapter_length': float(np.mean(chapter_lengths)) if chapter_lengths else 0,
            'std_chapter_length': float(np.std(chapter_lengths)) if chapter_lengths else 0
        }

        return features

    def extract_punctuation_features(self, text: str) -> Dict[str, float]:
        total_chars = len(text)

        if total_chars == 0:
            return {
                'pct_exclamation': 0,
                'pct_question': 0,
                'pct_comma': 0,
                'pct_semicolon': 0,
                'pct_colon': 0,
                'pct_dash': 0,
                'pct_quote': 0
            }

        features = {
            'pct_exclamation': 100 * text.count('!') / total_chars,
            'pct_question': 100 * text.count('?') / total_chars,
            'pct_comma': 100 * text.count(',') / total_chars,
            'pct_semicolon': 100 * text.count(';') / total_chars,
            'pct_colon': 100 * text.count(':') / total_chars,
            'pct_dash': 100 * (text.count('-') + text.count('—')) / total_chars,
            'pct_quote': 100 * (text.count('"') + text.count("'")) / total_chars
        }

        return features

    def extract_dialogue_features(self, text: str) -> Dict[str, float]:
        lines = text.split('\n')
        dialogue_lines = sum(1 for line in lines if '"' in line or "'" in line)

        total_lines = len(lines)

        features = {
            'pct_dialogue_lines': 100 * dialogue_lines / total_lines if total_lines > 0 else 0
        }

        return features

    def extract_all_features(
        self,
        text: str,
        sentence_lengths: List[int] = None,
        lemmas: List[str] = None,
        pos_ratios: Dict[str, float] = None,
        num_entities: int = 0
    ) -> Dict[str, float]:
        features = {}

        if sentence_lengths:
            features.update(self.extract_sentence_features(sentence_lengths))
        else:

            sentences = re.split(r'[.!?]+', text)
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            features.update(self.extract_sentence_features(sentence_lengths))

        features.update(self.extract_word_features(text, lemmas))

        features.update(self.extract_chapter_features(text))

        features.update(self.extract_punctuation_features(text))

        features.update(self.extract_dialogue_features(text))

        if pos_ratios:
            features.update(pos_ratios)

        if num_entities is not None:
            total_words = features.get('total_words', 1)
            features['pct_named_entities'] = 100 * num_entities / total_words if total_words > 0 else 0

        return features

def extract_features_from_dataset(
    metadata_file: Path,
    output_file: Path = PROCESSED_DATA_DIR / "features.csv",
    batch_size: int = 50
) -> pd.DataFrame:
    df = pd.read_csv(metadata_file)
    logger.info(f"Extracting features from {len(df)} books")
    logger.info(f"Processing in batches of {batch_size} books to save memory")

    extractor = TextFeatureExtractor()

    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:

            text_path = Path(row['processed_path'])
            if not text_path.is_absolute():
                text_path = PROCESSED_DATA_DIR.parent / text_path

            text = text_path.read_text(encoding='utf-8')

            pos_ratios = None
            sentence_lengths = None
            num_entities = 0

            features = extractor.extract_all_features(
                text,
                sentence_lengths=sentence_lengths,
                pos_ratios=pos_ratios,
                num_entities=num_entities
            )

            features['book_id'] = row['book_id']
            features['title'] = row['title']
            features['author'] = row['author']
            features['genre'] = row['genre']

            all_features.append(features)

            del text

            if (idx + 1) % batch_size == 0:
                gc.collect()
                logger.info(f"Processed {idx + 1}/{len(df)} books, freed memory")

        except Exception as e:
            logger.error(f"Error extracting features from book {row['book_id']}: {e}")
            continue

    gc.collect()

    features_df = pd.DataFrame(all_features)

    metadata_cols = ['book_id', 'title', 'author', 'genre']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    features_df = features_df[metadata_cols + feature_cols]

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_file, index=False)
    logger.info(f"Features saved to {output_file}")

    logger.info(f"\nExtracted {len(feature_cols)} features:")
    logger.info(f"Sample feature names: {feature_cols[:10]}")

    return features_df
