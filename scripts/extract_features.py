#!/usr/bin/env python3
"""
Script to extract features from preprocessed books.

Usage:
    python scripts/extract_features.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import PROCESSED_DATA_DIR
from src.features.pos_analyzer import analyze_dataset
from src.features.text_features import extract_features_from_dataset
from src.features.vocabulary_analyzer import extract_and_save_keywords

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from preprocessed books"
    )
    parser.add_argument(
        '--skip_pos',
        action='store_true',
        help='Skip POS analysis (slow)'
    )
    parser.add_argument(
        '--skip_keywords',
        action='store_true',
        help='Skip keyword extraction'
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

    if not args.skip_pos:
        logger.info("\nStep 1: POS Analysis with Stanza (this may take several hours)")
        logger.info("-"*60)
        logger.info("Processing linguistic features: POS tags, lemmas, NER")

        pos_df = analyze_dataset(
            metadata_file=metadata_file,
            output_dir=PROCESSED_DATA_DIR / "pos_analysis",
            remove_propn=False
        )

        logger.info(f"\nPOS analysis complete: {len(pos_df)} books analyzed")
    else:
        logger.info("\nStep 1: Skipping POS analysis (--skip_pos)")
        logger.info("Note: POS ratios will not be available for feature model")

    if not args.skip_keywords:
        logger.info("\nStep 2: Extracting characteristic keywords")
        logger.info("-"*60)

        keywords = extract_and_save_keywords(
            metadata_file=metadata_file,
            output_file=PROCESSED_DATA_DIR / "genre_keywords.json",
            use_stemming=False
        )

        logger.info(f"\nKeyword extraction complete")
        for genre, words in keywords.items():
            logger.info(f"  {genre}: {len(words)} keywords")
    else:
        logger.info("\nStep 2: Skipping keyword extraction (--skip_keywords)")

    logger.info("\nStep 3: Extracting statistical text features")
    logger.info("-"*60)

    pos_stats_file = PROCESSED_DATA_DIR / "pos_analysis" / "pos_stats.csv"
    if not pos_stats_file.exists():
        logger.warning("POS statistics not found, feature extraction will not include POS ratios")
        logger.warning("Run without --skip_pos to include POS features")
        pos_stats_file = None

    features_df = extract_features_from_dataset(
        metadata_file=metadata_file,
        pos_stats_file=pos_stats_file,
        output_file=PROCESSED_DATA_DIR / "features.csv"
    )

    logger.info(f"\nFeature extraction complete: {len(features_df)} books, {len(features_df.columns)} features")

    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info("\nNext step:")
    logger.info("  Train models: python scripts/train_all_models.py")


if __name__ == "__main__":
    main()
