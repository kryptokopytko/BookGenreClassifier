#!/usr/bin/env python3
"""
Create a beautiful summary of downloaded books.

Usage:
    python scripts/summarize_dataset.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROCESSED_DATA_DIR


def print_section(title, char="="):
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def print_subsection(title):
    print(f"\n{title}")
    print("-" * len(title))


def main():
    metadata_file = PROCESSED_DATA_DIR / "metadata_processed.csv"
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print("Please run download_books.py first")
        return

    df = pd.read_csv(metadata_file)

    train_file = PROCESSED_DATA_DIR / "train.csv"
    val_file = PROCESSED_DATA_DIR / "val.csv"
    test_file = PROCESSED_DATA_DIR / "test.csv"

    has_splits = all([f.exists() for f in [train_file, val_file, test_file]])

    if has_splits:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

    print_section("📚 BOOK GENRE CLASSIFIER - DATASET SUMMARY 📚")

    print_section("📊 OVERALL STATISTICS", "=")

    print(f"{'Total Books:':<30} {len(df):>10,}")
    print(f"{'Unique Authors:':<30} {df['author'].nunique():>10,}")
    print(f"{'Total Genres:':<30} {df['genre'].nunique():>10}")
    print(f"{'Total Words (all books):':<30} {df['word_count'].sum():>10,}")
    print(f"{'Average Book Length:':<30} {df['word_count'].mean():>10,.0f} words")
    print(f"{'Median Book Length:':<30} {df['word_count'].median():>10,.0f} words")
    print(f"{'Shortest Book:':<30} {df['word_count'].min():>10,} words")
    print(f"{'Longest Book:':<30} {df['word_count'].max():>10,} words")

    print_section("📖 BOOKS PER GENRE", "=")

    genre_counts = df['genre'].value_counts().sort_index()

    max_count = genre_counts.max()
    max_genre_len = max(len(str(g)) for g in genre_counts.index)

    print(f"{'Genre':<{max_genre_len+2}} {'Count':>7}  {'Percentage':>10}  Distribution")
    print("-" * 80)

    for genre, count in genre_counts.items():
        percentage = 100 * count / len(df)
        bar_length = int(40 * count / max_count)
        bar = "█" * bar_length
        print(f"{genre:<{max_genre_len+2}} {count:>7}  {percentage:>9.1f}%  {bar}")

    print_section("👤 AUTHORS STATISTICS", "=")

    author_counts = df.groupby('author').size().sort_values(ascending=False)

    print(f"{'Total Authors:':<30} {len(author_counts)}")
    print(f"{'Authors with 1 book:':<30} {(author_counts == 1).sum()}")
    print(f"{'Authors with 2+ books:':<30} {(author_counts >= 2).sum()}")
    print(f"{'Authors with 5+ books:':<30} {(author_counts >= 5).sum()}")
    print(f"{'Max books by one author:':<30} {author_counts.max()}")

    print_subsection("\nTop 10 Most Prolific Authors:")
    print(f"\n{'Author':<40} {'Books':>7}  {'Primary Genre':<20}")
    print("-" * 75)

    for author in author_counts.head(10).index:
        count = author_counts[author]
        primary_genre = df[df['author'] == author]['genre'].mode().iloc[0]
        author_short = author[:37] + "..." if len(author) > 40 else author
        print(f"{author_short:<40} {count:>7}  {primary_genre:<20}")

    print_section("📏 WORD COUNT STATISTICS", "=")

    print(f"{'Statistic':<30} {'Words':>15}")
    print("-" * 50)
    print(f"{'Mean:':<30} {df['word_count'].mean():>15,.0f}")
    print(f"{'Median:':<30} {df['word_count'].median():>15,.0f}")
    print(f"{'Std Dev:':<30} {df['word_count'].std():>15,.0f}")
    print(f"{'25th percentile:':<30} {df['word_count'].quantile(0.25):>15,.0f}")
    print(f"{'75th percentile:':<30} {df['word_count'].quantile(0.75):>15,.0f}")

    print_subsection("\nWord Count by Genre:")
    print(f"\n{'Genre':<25} {'Mean':>12}  {'Median':>12}  {'Min':>10}  {'Max':>10}")
    print("-" * 80)

    for genre in sorted(df['genre'].unique()):
        genre_df = df[df['genre'] == genre]
        print(f"{genre:<25} {genre_df['word_count'].mean():>12,.0f}  "
              f"{genre_df['word_count'].median():>12,.0f}  "
              f"{genre_df['word_count'].min():>10,}  "
              f"{genre_df['word_count'].max():>10,}")

    if has_splits:
        print_section("🔀 TRAIN/VAL/TEST SPLIT", "=")

        print(f"{'Split':<15} {'Books':>10}  {'Authors':>10}  {'Percentage':>12}")
        print("-" * 55)

        for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
            n_books = len(split_df)
            n_authors = split_df['author'].nunique()
            percentage = 100 * n_books / len(df)
            print(f"{name:<15} {n_books:>10}  {n_authors:>10}  {percentage:>11.1f}%")

        print_subsection("\nGenre Distribution Across Splits:")
        print(f"\n{'Genre':<25} {'Train':>8}  {'Val':>6}  {'Test':>6}  {'Total':>7}")
        print("-" * 65)

        for genre in sorted(df['genre'].unique()):
            train_count = len(train_df[train_df['genre'] == genre])
            val_count = len(val_df[val_df['genre'] == genre])
            test_count = len(test_df[test_df['genre'] == genre])
            total_count = train_count + val_count + test_count

            print(f"{genre:<25} {train_count:>8}  {val_count:>6}  {test_count:>6}  {total_count:>7}")

        print_subsection("\n⚠️  Data Quality Warnings:")
        warnings = []

        for genre in sorted(df['genre'].unique()):
            test_count = len(test_df[test_df['genre'] == genre])
            if test_count == 0:
                warnings.append(f"  ❌ {genre}: NO books in test set!")
            elif test_count < 3:
                warnings.append(f"  ⚠️  {genre}: Only {test_count} book(s) in test set (very low!)")

        if warnings:
            print()
            for warning in warnings:
                print(warning)
        else:
            print("\n  ✅ All genres have adequate representation in test set!")

        train_authors = set(train_df['author'].unique())
        val_authors = set(val_df['author'].unique())
        test_authors = set(test_df['author'].unique())

        overlap = (train_authors & val_authors) | (train_authors & test_authors) | (val_authors & test_authors)

        if overlap:
            print(f"\n  ⚠️  WARNING: {len(overlap)} authors appear in multiple splits!")
        else:
            print("\n  ✅ No author overlap between splits (clean split by author!)")

    print_section("📚 SAMPLE BOOKS (Random 5)", "=")

    sample_books = df.sample(min(5, len(df)))

    for _, row in sample_books.iterrows():
        print(f"\n{row['title']}")
        print(f"  Author: {row['author']}")
        print(f"  Genre: {row['genre']}")
        print(f"  Length: {row['word_count']:,} words")
        print(f"  ID: {row['book_id']}")

    print_section("💡 RECOMMENDATIONS", "=")

    total_books = len(df)

    if total_books < 200:
        print("❌ CRITICAL: Dataset too small!")
        print(f"   Current: {total_books} books")
        print(f"   Recommended: 400-800 books (50-100 per genre)")
        print()
        print("   Action: Run download_books.py with --books_per_genre 50 or 100")
    elif total_books < 400:
        print("⚠️  WARNING: Dataset is small")
        print(f"   Current: {total_books} books")
        print(f"   Recommended: 400-800 books")
        print()
        print("   Consider downloading more books for better model performance")
    else:
        print("✅ Dataset size is adequate!")
        print(f"   Current: {total_books} books")

    if has_splits:
        test_size = len(test_df)
        n_genres = df['genre'].nunique()
        books_per_genre_test = test_size / n_genres

        print()
        if books_per_genre_test < 10:
            print(f"❌ Test set too small: {test_size} books ({books_per_genre_test:.1f} per genre)")
            print(f"   Recommended: at least 80-120 books in test set (~10-15 per genre)")
        elif books_per_genre_test < 15:
            print(f"⚠️  Test set is small: {test_size} books ({books_per_genre_test:.1f} per genre)")
        else:
            print(f"✅ Test set size is good: {test_size} books ({books_per_genre_test:.1f} per genre)")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
