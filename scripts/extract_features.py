import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import PROCESSED_DATA_DIR, GENRES
from src.features.text_features import extract_features_from_dataset
from src.features.vocabulary_analyzer import extract_and_save_keywords

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from preprocessed books"
    )

    parser.add_argument(
        '--skip_keywords',
        action='store_true',
        help='Skip keyword extraction'
    )

    parser.add_argument(
        '--skip_statistical',
        action='store_true',
        help='Skip statistical extraction'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("FEATURE EXTRACTION")
    logger.info("="*60)

    metadata_file = PROCESSED_DATA_DIR / "metadata_processed.csv"

    if not metadata_file.exists():
        logger.error(f"Processed metadata file not found: {metadata_file}")
        logger.error("Please run download_books.py first")
        sys.exit(1)

    if not args.skip_keywords:
        logger.info("\nStep 2: Extracting characteristic keywords")
        logger.info("-"*60)

        keywords = extract_and_save_keywords(
            metadata_file=metadata_file,
            output_file=PROCESSED_DATA_DIR / "genre_keywords.json",
        )

        logger.info(f"\nKeyword extraction complete")
        for genre, words in keywords.items():
            logger.info(f"  {genre}: {len(words)} keywords")
    else:
        logger.info("\nStep 2: Skipping keyword extraction (--skip_keywords)")

    if not args.skip_statistical:
        logger.info("\nStep 3: Extracting statistical text features")
        logger.info("-"*60)

        features_df = extract_features_from_dataset(
            metadata_file=metadata_file,
            output_file=PROCESSED_DATA_DIR / "features.csv"
        )

        logger.info(f"\nFeature extraction complete: {len(features_df)} books, {len(features_df.columns)} features") 
    else:
        logger.info("\nStep 3: Skipping statistical extraction (--skip_statistical)")

    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info("\nNext step:")
    logger.info("  Train models: python scripts/train_all_models.py")


if __name__ == "__main__":
    main()
