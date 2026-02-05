import re
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_FILE,
    CHAPTER_MARKERS
)

logger = get_logger(__name__)

class GutenbergPreprocessor:
    HEADER_MARKERS = [
        r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*",
        r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK",
        r"\*END\*THE SMALL PRINT",
    ]

    FOOTER_MARKERS = [
        r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*",
        r"END OF (THIS|THE) PROJECT GUTENBERG EBOOK",
        r"End of (the )?Project Gutenberg",
    ]

    def __init__(self):
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def remove_gutenberg_header(self, text: str) -> str:
        for pattern in self.HEADER_MARKERS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = text[match.end():]
                break

        return text

    def remove_gutenberg_footer(self, text: str) -> str:
        for pattern in self.FOOTER_MARKERS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = text[:match.start()]
                break

        return text

    def clean_whitespace(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        if keep_punctuation:

            text = re.sub(r'[^\w\s.,:;!?\'"()\-\[\]{}]', ' ', text)
        else:

            text = re.sub(r'[^\w\s]', ' ', text)

        return text

    def detect_chapters(self, text: str) -> List[Tuple[int, str]]:

        pattern = r'^(' + '|'.join(CHAPTER_MARKERS) + r')\s+([IVXLCDM\d]+|[A-Z][a-z]+).*$'

        for i, line in enumerate(text.split('\n')):
            match = re.match(pattern, line.strip(), re.IGNORECASE)
            if match:
                chapters.append((i, line.strip()))

        return chapters

    def preprocess_book(self, text: str, remove_chapters: bool = False) -> str:
        text = self.remove_gutenberg_header(text)
        text = self.remove_gutenberg_footer(text)

        text = self.clean_whitespace(text)

        if remove_chapters:
            chapter_pattern = r'^(' + '|'.join(CHAPTER_MARKERS) + r').*$'
            lines = text.split('\n')
            lines = [line for line in lines if not re.match(chapter_pattern, line.strip(), re.IGNORECASE)]
            text = '\n'.join(lines)

        return text

    def preprocess_file(self, file_path: Path, output_path: Path) -> bool:
        try:
            text = file_path.read_text(encoding='utf-8')

            cleaned_text = self.preprocess_book(text)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text(cleaned_text, encoding='utf-8')

            return True

        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {e}")
            return False

    def preprocess_all(self, metadata_file: Path = METADATA_FILE) -> pd.DataFrame:
        df = pd.read_csv(metadata_file, engine='python', on_bad_lines='warn')

        original_count = len(df)
        df = df.dropna(subset=['file_path'])
        if len(df) < original_count:
            logger.warning(f"Removed {original_count - len(df)} rows with missing file_path")

        logger.info(f"Preprocessing {len(df)} books")

        processed_paths = []
        successful_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing books"):

            input_path = Path(row['file_path'])
            if not input_path.is_absolute():
                input_path = Path(RAW_DATA_DIR).parent / input_path

            relative_path = input_path.relative_to(RAW_DATA_DIR)
            output_path = self.processed_dir / relative_path

            success = self.preprocess_file(input_path, output_path)

            if success:
                processed_paths.append(str(output_path.relative_to(self.processed_dir.parent)))
                successful_count += 1
            else:
                processed_paths.append(None)

        df['processed_path'] = processed_paths

        df = df[df['processed_path'].notna()].copy()

        logger.info(f"Successfully preprocessed {successful_count}/{len(processed_paths)} books")

        output_metadata = self.processed_dir / "metadata_processed.csv"
        df.to_csv(output_metadata, index=False)
        logger.info(f"Updated metadata saved to {output_metadata}")

        return df

    def get_book_stats(self, text: str) -> dict:
        sentences = re.split(r'[.!?]+', text)

        chapters = self.detect_chapters(text)

        stats = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'chapter_count': len(chapters),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }

        return stats

def preprocess_single_text(text: str) -> str:
    return preprocessor.preprocess_book(text)
