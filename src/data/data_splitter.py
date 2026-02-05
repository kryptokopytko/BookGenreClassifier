import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger
from ..utils.config import (
    PROCESSED_DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
)

logger = get_logger(__name__)

class DataSplitter:
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def split_by_author(
        self,
        df: pd.DataFrame,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"

        authors = df['author'].unique()
        logger.info(f"Total unique authors: {len(authors)}")

        np.random.seed(self.random_state)
        np.random.shuffle(authors)

        n_train = int(len(authors) * train_ratio)
        n_val = int(len(authors) * val_ratio)

        train_authors = authors[:n_train]
        val_authors = authors[n_train:n_train + n_val]
        test_authors = authors[n_train + n_val:]

        logger.info(f"Train authors: {len(train_authors)}")
        logger.info(f"Val authors: {len(val_authors)}")
        logger.info(f"Test authors: {len(test_authors)}")

        train_df = df[df['author'].isin(train_authors)].copy()
        val_df = df[df['author'].isin(val_authors)].copy()
        test_df = df[df['author'].isin(test_authors)].copy()

        logger.info(f"Train books: {len(train_df)}")
        logger.info(f"Val books: {len(val_df)}")
        logger.info(f"Test books: {len(test_df)}")

        self._verify_no_author_overlap(train_df, val_df, test_df)

        self._log_genre_distribution(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def stratified_split_by_author(
        self,
        df: pd.DataFrame,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        stratify_by: str = 'genre'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        author_genres = df.groupby('author')[stratify_by].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()
        author_genres.columns = ['author', 'primary_genre']

        logger.info(f"Splitting {len(author_genres)} authors stratified by {stratify_by}")

        try:

            train_authors, temp_authors, train_genres, temp_genres = train_test_split(
                author_genres['author'],
                author_genres['primary_genre'],
                train_size=train_ratio,
                stratify=author_genres['primary_genre'],
                random_state=self.random_state
            )

            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

            val_authors, test_authors = train_test_split(
                temp_authors,
                train_size=val_ratio_adjusted,
                stratify=temp_genres,
                random_state=self.random_state
            )

            logger.info("Using stratified split by genre")

        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}")
            logger.warning("Falling back to non-stratified split by author")

            return self.split_by_author(df, train_ratio, val_ratio, test_ratio)

        train_df = df[df['author'].isin(train_authors)].copy()
        val_df = df[df['author'].isin(val_authors)].copy()
        test_df = df[df['author'].isin(test_authors)].copy()

        logger.info(f"Stratified split complete:")
        logger.info(f"  Train: {len(train_authors)} authors, {len(train_df)} books")
        logger.info(f"  Val: {len(val_authors)} authors, {len(val_df)} books")
        logger.info(f"  Test: {len(test_authors)} authors, {len(test_df)} books")

        self._verify_no_author_overlap(train_df, val_df, test_df)

        self._log_genre_distribution(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _verify_no_author_overlap(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        train_authors = set(train_df['author'].unique())
        val_authors = set(val_df['author'].unique())
        test_authors = set(test_df['author'].unique())

        overlap_train_val = train_authors & val_authors
        overlap_train_test = train_authors & test_authors
        overlap_val_test = val_authors & test_authors

        if overlap_train_val:
            logger.warning(f"Author overlap between train and val: {overlap_train_val}")
        if overlap_train_test:
            logger.warning(f"Author overlap between train and test: {overlap_train_test}")
        if overlap_val_test:
            logger.warning(f"Author overlap between val and test: {overlap_val_test}")

        if not (overlap_train_val or overlap_train_test or overlap_val_test):
            logger.info("✓ No author overlap detected - splits are clean!")

    def _log_genre_distribution(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        logger.info("\nGenre distribution:")

        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            logger.info(f"\n{split_name}:")
            genre_counts = split_df['genre'].value_counts()
            for genre, count in genre_counts.items():
                percentage = 100 * count / len(split_df)
                logger.info(f"  {genre}: {count} ({percentage:.1f}%)")

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path = PROCESSED_DATA_DIR
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        logger.info(f"Splits saved to {output_dir}")

    def load_splits(self, input_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        train_df = pd.read_csv(input_dir / "train.csv")
        val_df = pd.read_csv(input_dir / "val.csv")
        test_df = pd.read_csv(input_dir / "test.csv")

        logger.info(f"Loaded splits from {input_dir}")
        logger.info(f"  Train: {len(train_df)} books")
        logger.info(f"  Val: {len(val_df)} books")
        logger.info(f"  Test: {len(test_df)} books")

        return train_df, val_df, test_df

    def get_author_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        author_stats = df.groupby('author').agg({
            'book_id': 'count',
            'genre': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
            'word_count': 'mean'
        }).reset_index()

        author_stats.columns = ['author', 'num_books', 'primary_genre', 'avg_word_count']
        author_stats = author_stats.sort_values('num_books', ascending=False)

        return author_stats

def create_splits(
    metadata_file: Path,
    output_dir: Path = PROCESSED_DATA_DIR,
    stratified: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(metadata_file)
    logger.info(f"Loaded {len(df)} books from {metadata_file}")

    splitter = DataSplitter()

    if stratified:
        train_df, val_df, test_df = splitter.stratified_split_by_author(df)
    else:
        train_df, val_df, test_df = splitter.split_by_author(df)

    splitter.save_splits(train_df, val_df, test_df, output_dir)

    author_stats = splitter.get_author_stats(df)
    logger.info(f"\nTop 10 authors by book count:")
    logger.info(author_stats.head(10).to_string())

    return train_df, val_df, test_df
