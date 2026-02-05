#!/usr/bin/env python3
"""
Script to download and preprocess books from Project Gutenberg.

Usage:
    python scripts/download_books.py --books_per_genre 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import GENRES, BOOKS_PER_GENRE, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.gutenberg_scraper import GutenbergScraper
from src.data.preprocessor import GutenbergPreprocessor
from src.data.data_splitter import create_splits

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess books from Project Gutenberg"
    )
    parser.add_argument(
        '--books_per_genre',
        type=int,
        default=BOOKS_PER_GENRE,
        help=f'Number of books to download per genre (default: {BOOKS_PER_GENRE})'
    )
    parser.add_argument(
        '--genres',
        nargs='+',
        default=GENRES,
        help=f'List of genres to download (default: {GENRES})'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip downloading, only preprocess existing books'
    )
    parser.add_argument(
        '--skip_preprocessing',
        action='store_true',
        help='Skip preprocessing, only download books'
    )
    parser.add_argument(
        '--skip_splitting',
        action='store_true',
        help='Skip data splitting'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("BOOK DOWNLOAD AND PREPROCESSING")
    logger.info("="*60)

    if not args.skip_download:
        logger.info("\nStep 1: Downloading books from Project Gutenberg")
        logger.info("-"*60)

        priority_genres = [
            "Adventure/Action",
            "Thriller/Horror",
            "Fantasy",
        ]

        genres_to_download = args.genres.copy()
        if genres_to_download == GENRES:
            remaining_genres = [g for g in genres_to_download if g not in priority_genres]
            genres_to_download = priority_genres + remaining_genres

            logger.info("\n⚠️  Prioritizing underrepresented genres:")
            for genre in priority_genres:
                logger.info(f"   📚 {genre}")
            logger.info("")

        scraper = GutenbergScraper(output_dir=RAW_DATA_DIR)
        scraper.scrape_all_genres(
            genres=genres_to_download,
            books_per_genre=args.books_per_genre
        )

        logger.info(f"\nDownloaded books to: {RAW_DATA_DIR}")
    else:
        logger.info("\nStep 1: Skipping download (--skip_download)")

    if not args.skip_preprocessing:
        logger.info("\nStep 2: Preprocessing books")
        logger.info("-"*60)

        from src.utils.config import METADATA_FILE

        if not METADATA_FILE.exists():
            logger.error(f"Metadata file not found: {METADATA_FILE}")
            logger.error("Please run download first (without --skip_download)")
            sys.exit(1)

        preprocessor = GutenbergPreprocessor()
        processed_df = preprocessor.preprocess_all(metadata_file=METADATA_FILE)

        logger.info(f"\nPreprocessed books saved to: {PROCESSED_DATA_DIR}")
        logger.info(f"Processed {len(processed_df)} books successfully")
    else:
        logger.info("\nStep 2: Skipping preprocessing (--skip_preprocessing)")

    if not args.skip_splitting:
        logger.info("\nStep 3: Splitting data by author")
        logger.info("-"*60)

        metadata_file = PROCESSED_DATA_DIR / "metadata_processed.csv"

        if not metadata_file.exists():
            logger.error(f"Processed metadata file not found: {metadata_file}")
            logger.error("Please run preprocessing first")
            sys.exit(1)

        train_df, val_df, test_df = create_splits(
            metadata_file=metadata_file,
            output_dir=PROCESSED_DATA_DIR,
            stratified=True
        )

        logger.info(f"\nData splits saved to: {PROCESSED_DATA_DIR}")
        logger.info(f"  Train: {len(train_df)} books")
        logger.info(f"  Val: {len(val_df)} books")
        logger.info(f"  Test: {len(test_df)} books")
    else:
        logger.info("\nStep 3: Skipping data splitting (--skip_splitting)")

    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD AND PREPROCESSING COMPLETE")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Extract features: python scripts/extract_features.py")
    logger.info("2. Train models: python scripts/train_all_models.py")


if __name__ == "__main__":
    main()
