import sys
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_DIR, DATA_DIR

logger = get_logger(__name__)


def extract_book_id_from_filename(filepath: Path) -> str:
    filename = filepath.stem
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    else:
        return filename


def extract_title_from_text(text: str, max_lines: int = 100) -> str:
    lines = text.split('\n')[:max_lines]

    patterns = [
        r'Title:\s*(.+)',
        r'title:\s*(.+)',
        r'TITLE:\s*(.+)',
        r'The Project Gutenberg [eE]Book of (.+?),',
        r'The Project Gutenberg [eE]Book of (.+)',
    ]

    for line in lines:
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                title = title.replace('***', '').strip()
                if len(title) > 5 and len(title) < 200:
                    return title

    return "Unknown"


def extract_author_from_text(text: str, max_lines: int = 100) -> str:
    lines = text.split('\n')[:max_lines]

    patterns = [
        r'Author:\s*(.+)',
        r'author:\s*(.+)',
        r'AUTHOR:\s*(.+)',
        r'by\s+([A-Z][a-zA-Z\s\.]+)',
        r'By\s+([A-Z][a-zA-Z\s\.]+)',
    ]

    for line in lines:
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                author = match.group(1).strip()
                author = re.sub(r'\s+', ' ', author)
                author = author.replace('***', '').strip()
                author = re.sub(r'[,\.\:]$', '', author)
                if len(author) > 3 and len(author) < 100:
                    if not any(word in author.lower() for word in ['project', 'gutenberg', 'ebook', 'release']):
                        return author

    return "Unknown"


def count_words(text: str) -> int:
    words = text.split()
    return len(words)


def extract_metadata_from_file(filepath: Path, genre: str) -> dict:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        book_id = extract_book_id_from_filename(filepath)
        title = extract_title_from_text(text)
        author = extract_author_from_text(text)
        word_count = count_words(text)

        relative_path = filepath.relative_to(Path(RAW_DATA_DIR).parent)

        metadata = {
            'book_id': book_id,
            'title': title,
            'author': author,
            'genre': genre,
            'word_count': word_count,
            'file_path': str(relative_path),
            'url': f'https://www.gutenberg.org/ebooks/{book_id}',
            'processed_path': str(relative_path).replace('raw', 'processed')
        }

        return metadata

    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None


def scan_and_generate_metadata(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    raw_dir = Path(raw_dir)

    logger.info(f"Scanning {raw_dir} for books...")

    all_metadata = []

    for genre_dir in raw_dir.iterdir():
        if not genre_dir.is_dir() or genre_dir.name.startswith('.'):
            continue

        genre = genre_dir.name

        book_files = list(genre_dir.glob('*.txt'))
        logger.info(f"Found {len(book_files)} books in {genre}")

        for filepath in tqdm(book_files, desc=f"Processing {genre}", unit="book"):
            metadata = extract_metadata_from_file(filepath, genre)
            if metadata:
                all_metadata.append(metadata)

    df = pd.DataFrame(all_metadata)

    logger.info(f"\nTotal books processed: {len(df)}")
    logger.info(f"Books per genre:")
    genre_counts = df['genre'].value_counts()
    for genre, count in genre_counts.items():
        logger.info(f"  {genre}: {count}")

    return df


def main():
    logger.info("="*60)
    logger.info("UPDATING METADATA FROM EXISTING BOOKS")
    logger.info("="*60)

    raw_dir = Path(RAW_DATA_DIR)
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        sys.exit(1)

    output_file = DATA_DIR / "metadata.csv"
    existing_metadata = pd.DataFrame()

    if output_file.exists():
        logger.info(f"Found existing metadata: {output_file}")
        try:
            existing_metadata = pd.read_csv(output_file, engine='python', on_bad_lines='warn')
            logger.info(f"Existing books: {len(existing_metadata)}")
            existing_book_ids = set(existing_metadata['book_id'].astype(str))
        except Exception as e:
            logger.warning(f"Could not read existing metadata: {e}")
            logger.info("Backing up old metadata and creating fresh")
            backup_file = output_file.with_suffix('.csv.bak')
            output_file.rename(backup_file)
            logger.info(f"Old metadata backed up to: {backup_file}")
            existing_metadata = pd.DataFrame()
            existing_book_ids = set()
    else:
        logger.info("No existing metadata found, creating from scratch")
        existing_book_ids = set()

    new_metadata_df = scan_and_generate_metadata(raw_dir)

    if new_metadata_df.empty:
        logger.error("No books found! Please download books first.")
        sys.exit(1)

    new_metadata_df['book_id'] = new_metadata_df['book_id'].astype(str)
    truly_new = new_metadata_df[~new_metadata_df['book_id'].isin(existing_book_ids)]

    logger.info(f"\nSummary:")
    logger.info(f"  Books found in files: {len(new_metadata_df)}")
    logger.info(f"  Already in metadata: {len(new_metadata_df) - len(truly_new)}")
    logger.info(f"  New books to add: {len(truly_new)}")

    if not existing_metadata.empty and not truly_new.empty:
        existing_metadata['book_id'] = existing_metadata['book_id'].astype(str)
        metadata_df = pd.concat([existing_metadata, truly_new], ignore_index=True)
        logger.info(f"\n✓ Added {len(truly_new)} new books to existing metadata")
    elif not existing_metadata.empty:
        metadata_df = existing_metadata
        logger.info(f"\n✓ No new books found, metadata unchanged")
    else:
        metadata_df = new_metadata_df
        logger.info(f"\n✓ Created new metadata with {len(metadata_df)} books")

    metadata_df['word_count'] = pd.to_numeric(metadata_df['word_count'], errors='coerce')

    metadata_df.to_csv(output_file, index=False)
    logger.info(f"✓ Metadata saved to: {output_file}")

    logger.info("\n" + "="*60)
    logger.info("METADATA STATISTICS")
    logger.info("="*60)

    logger.info(f"\nTotal books: {len(metadata_df)}")
    logger.info(f"Books with known titles: {sum(metadata_df['title'] != 'Unknown')}")
    logger.info(f"Books with known authors: {sum(metadata_df['author'] != 'Unknown')}")

    logger.info(f"\nWord count statistics:")
    logger.info(f"  Mean: {metadata_df['word_count'].mean():.0f}")
    logger.info(f"  Median: {metadata_df['word_count'].median():.0f}")
    logger.info(f"  Min: {metadata_df['word_count'].min()}")
    logger.info(f"  Max: {metadata_df['word_count'].max()}")

    logger.info(f"\nBooks per genre:")
    genre_counts = metadata_df['genre'].value_counts().sort_index()
    for genre, count in genre_counts.items():
        percentage = 100 * count / len(metadata_df)
        logger.info(f"  {genre}: {count} ({percentage:.1f}%)")

    logger.info("\n" + "="*60)
    logger.info("QUALITY CHECKS")
    logger.info("="*60)

    unknown_titles = sum(metadata_df['title'] == 'Unknown')
    unknown_authors = sum(metadata_df['author'] == 'Unknown')
    if unknown_titles > 0:
        logger.warning(f"{unknown_titles} books have unknown titles")
    if unknown_authors > 0:
        logger.warning(f"{unknown_authors} books have unknown authors")

    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("\n1. Preprocess books:")
    logger.info("   python scripts/preprocess_books.py")
    logger.info("\n2. Split data (by author):")
    logger.info("   python scripts/split_data.py")
    logger.info("\n3. Extract features:")
    logger.info("   python scripts/extract_features.py")
    logger.info("\n4. Train models:")
    logger.info("   python scripts/train_models_improved.py")
    logger.info("   python scripts/train_new_models.py --ensemble")

    logger.info("\n✓ Metadata generation complete!")


if __name__ == "__main__":
    main()
