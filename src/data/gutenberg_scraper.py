import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from ..utils.logger import get_logger
from ..utils.config import (
    RAW_DATA_DIR,
    METADATA_FILE,
    BOOKS_PER_GENRE,
    MIN_BOOK_LENGTH,
    MAX_BOOK_LENGTH,
    RESULTS_PER_SITE,
    GENRES
)

logger = get_logger(__name__)

class UrlNotFoundError(Exception):
    pass

class GutenbergScraper:
    BASE_URL = "https://www.gutenberg.org"
    BOOKSHELF_URL = f"{BASE_URL}/ebooks/bookshelf/"

    def __init__(self, output_dir: Path = RAW_DATA_DIR, books_per_genre = BOOKS_PER_GENRE):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; BookGenreClassifier/1.0; Educational Research)"
            }
        )
        self.new_metadata = []
        self.existing_metadata = []
        self.books_per_genre = books_per_genre

    def is_english_book(self, book_id: str) -> bool:
        book_url = f"{self.BASE_URL}/ebooks/{book_id}"
        try:
            response = self.session.get(book_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for tr in soup.select("table.bibrec tr"):
                th = tr.find("th")
                td = tr.find("td")
                if th and td and th.text.strip() == "Language":
                    languages = [lang.strip() for lang in td.text.split(",")]
                    return "English" in languages

        except requests.RequestException:
            logger.warning(f"Could not fetch language info for book {book_id}")
            return False

        return False
    
    def is_multi_genre(self, book_id: str) -> bool:
        book_url = f"{self.BASE_URL}/ebooks/{book_id}"
        try:
            response = self.session.get(book_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            category_links = soup.find_all("a", href=re.compile(r"^/ebooks/bookshelf/\d+$"))
            categories = set(link.text.strip() for link in category_links)
            clean_categories = {
                cat.replace("In Category: ", "")
                .replace(" and ", " & ")
                .strip()
                for cat in categories
            }
            
            matching_genres = [cat for cat in clean_categories if cat in GENRES]
            return len(matching_genres) > 1

        except requests.RequestException:
            logger.warning(f"Could not fetch categories for book {book_id}")
            return False

        return True
    
    def get_category_map(self) -> bool:
        categories_url = f"{self.BASE_URL}/ebooks/categories"
        try:
            response = self.session.get(categories_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            category_map = {}

            for a in soup.find_all("a", href=re.compile(r"^/ebooks/bookshelf/\d+$")):
                shelf_id = int(a['href'].split('/')[-1])
                category_name = a.text.strip()
                category_map[category_name] = shelf_id

            return category_map

        except requests.RequestException as e:
            logger.error(f"Error fetching categories: {e}")
            return {}

    def search_bookshelf(
        self, bookshelf_name: str, bookshelf_id: int, max_results: int = 300, existing = {}, start_idx = 1
    ) -> List[Dict]:

        try:
            url = f"{self.BOOKSHELF_URL}{bookshelf_id}?start_index={start_idx}"

            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                if response.status_code == 404:
                    raise UrlNotFoundError(
                        f"Url '{url}' not found (HTTP 404)"
                    )
                else:
                    logger.warning(
                        f"Bookshelf '{bookshelf_name}' not found (status {response.status_code})"
                    )
                    return []

            soup = BeautifulSoup(response.text, "html.parser")

            books = []
            book_list = soup.find_all("li", class_="booklink")


            idx = 0
            while len(books) < max_results and idx < len(book_list):
                book_item = book_list[idx]
                link = book_item.find("a", class_="link")

                book_url = link.get("href")
                book_id_match = re.search(r"/ebooks/(\d+)", book_url)

                book_id = book_id_match.group(1)

                logger.debug(f"Consider {bookshelf_name} #{idx + start_idx}, id: {book_id}")

                if book_id in existing:
                    logger.debug(f"   |---> skipping, already in dataset")
                    idx += 1
                    continue

                
                title_span = book_item.find("span", class_="title")
                title = title_span.text.strip() if title_span else f"Book {book_id}"

                author_span = book_item.find("span", class_="subtitle")
                author = author_span.text.strip() if author_span else "Unknown"

                if self.is_english_book(book_id):
                    if not self.is_multi_genre(book_id):
                        text_result = self.download_book(book_id, title)
                        if text_result:
                            text, word_count = text_result
                            books.append(
                                {
                                    "book_id": book_id,
                                    "title": title,
                                    "author": author,
                                    "url": f"{self.BASE_URL}/ebooks/{book_id}",
                                    "text": text,
                                    "word_count": word_count,
                                }
                            )
                            logger.debug("   |---> adding")
                    else:
                        logger.debug("   |---> skipping, book belongs to multiple genres")
                else:
                    logger.debug("   |---> skipping, book language is not English")
                idx += 1

            return books

        except UrlNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error searching bookshelf '{bookshelf_name}': {e}")
            return []

    def get_book_text_url(self, book_id: str) -> Optional[str]:
        text_urls = [
            f"{self.BASE_URL}/files/{book_id}/{book_id}-0.txt",
            f"{self.BASE_URL}/files/{book_id}/{book_id}.txt",
            f"{self.BASE_URL}/cache/epub/{book_id}/pg{book_id}.txt",
        ]

        for url in text_urls:
            try:
                response = self.session.head(url, timeout=5)
                if response.status_code == 200:
                    return url
            except requests.RequestException:
                continue

        return None

    def download_book(
        self, book_id: str, title: str, min_length: int = MIN_BOOK_LENGTH
    ) -> Optional[Tuple[str, int]]:

        text_url = self.get_book_text_url(book_id)
        if not text_url:
            logger.warning(f"No text URL found for book {book_id}: {title}")
            return None

        try:
            response = self.session.get(text_url, timeout=30)
            response.raise_for_status()

            try:
                text = response.content.decode("utf-8")
            except UnicodeDecodeError:
                text = response.content.decode("latin-1")

            word_count = len(text.split())

            if word_count < min_length:
                logger.debug(
                    f"   |---> skipping, book too short ({word_count} < {min_length} words)"
                )
                return None

            if word_count > MAX_BOOK_LENGTH:
                logger.debug(f"skipping, book too long ({word_count} words)")
                return None

            return text, word_count

        except requests.RequestException as e:
            logger.error(f"Error downloading book {book_id}: {e}")
            return None

    def save_book(self, book_id: str, text: str, genre: str) -> Path:
        genre_dir = self.output_dir / genre.replace("/", "_")
        genre_dir.mkdir(parents=True, exist_ok=True)

        file_path = genre_dir / f"{book_id}.txt"
        file_path.write_text(text, encoding="utf-8")

        return file_path

    def scrape_genre(
        self, genre: str, shelf_id: int, target_count: int = BOOKS_PER_GENRE
    ) -> List[Dict]:
        existing_books = {}
        if not self.existing_metadata.empty:

            genre_metadata = self.existing_metadata[self.existing_metadata["genre"] == genre]

            existing_books = {
                str(row["book_id"]): row.to_dict()
                for _, row in genre_metadata.iterrows()
            }
            logger.info(
                f"Found {len(existing_books)} existing books for {genre} in metadata"
            )

        books_needed = target_count - len(existing_books)
        if books_needed <= 0:
            logger.info(
                f"Needed {target_count} books, target met!\n{'-' * 70}\n"
            )

            return list(existing_books.values())
        
        logger.info(
            f"Need {books_needed} new books (target: {target_count}, existing: {len(existing_books)})"
        )

        start_idx = 426 
        search_results = []
        books_needed_copy = books_needed

        while books_needed > 0:
            try:
                results = self.search_bookshelf(genre, shelf_id, max_results=books_needed, existing=existing_books, start_idx=start_idx)
                search_results.extend(results)
                books_needed -= len(results)

                logger.debug(f"Found {len(results)} new books in bookshelf '{genre}', from {start_idx}. position")
                if books_needed > 0:
                    logger.debug(f"Still need {books_needed} more")

                start_idx += RESULTS_PER_SITE
            except UrlNotFoundError:
                break

        logger.info(f"Found {len(search_results)} new books in bookshelf '{genre}'")

        downloaded_books = list(existing_books.values())
        new_downloads = 0
        authors_seen = set(book.get("author", "Unknown") for book in downloaded_books)

        progress_bar = tqdm(
            search_results,
            desc=f"Downloading {genre}",
            total=min(len(search_results), books_needed_copy),
        )

        for book_info in progress_bar:
            if len(downloaded_books) >= target_count:
                break

            book_id = book_info["book_id"]
            title = book_info["title"]
            author = book_info["author"]

            genre_dir = self.output_dir / genre.replace("/", "_")
            file_path = genre_dir / f"{book_id}.txt"

            if book_id in existing_books and file_path.exists():

                metadata_entry = existing_books[book_id]

                metadata_entry["file_path"] = str(
                    file_path.relative_to(self.output_dir.parent)
                )
                downloaded_books.append(metadata_entry)
                authors_seen.add(metadata_entry["author"])
                logger.debug(
                    f"Book {book_id} already exists (verified), skipping download"
                )
                continue

            if file_path.exists():

                try:
                    text = file_path.read_text(encoding="utf-8")
                    word_count = len(text.split())
                    logger.debug(
                        f"Book {book_id} file exists but not in metadata, adding"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error reading existing book {book_id}: {e}, re-downloading"
                    )

                    result = self.download_book(
                        book_id, title, min_length=MIN_BOOK_LENGTH
                    )
                    if result is None:
                        continue
                    text, word_count = result
                    file_path = self.save_book(book_id, text, genre)
            else:

                result = self.download_book(book_id, title, min_length=MIN_BOOK_LENGTH)

                if result is None:
                    continue

                text, word_count = result

                file_path = self.save_book(book_id, text, genre)

            metadata_entry = {
                "book_id": book_id,
                "title": title,
                "author": author,
                "genre": genre,
                "word_count": word_count,
                "file_path": str(file_path.relative_to(self.output_dir.parent)),
                "url": book_info["url"],
            }

            downloaded_books.append(metadata_entry)

            if book_id not in existing_books:
                self.new_metadata.append(metadata_entry)
                new_downloads += 1

            authors_seen.add(author)

            progress_bar.set_postfix(
                {
                    "total": len(downloaded_books),
                    "new": new_downloads,
                    "authors": len(authors_seen),
                }
            )

        if len(downloaded_books) < target_count:
            logger.warning(
                f"⚠️  Only got {len(downloaded_books)}/{target_count} books for {genre}. "
                f"Gutenberg may not have enough books in this category."
            )

        logger.info(f"{'-' * 70}\n")
        
        return downloaded_books

    def scrape_all_genres(
        self, genres: List[str]
    ):
        self.existing_metadata = self.load_metadata()

        category_map = self.get_category_map()

        for genre in genres:
            try:
                shelf_id = category_map[genre]
                self.scrape_genre(genre, shelf_id, self.books_per_genre)
            except Exception as e:
                logger.error(f"Error scraping genre {genre}: {e}")
                continue

        self.save_metadata()

    def save_metadata(self):

        existing_df = self.load_metadata()

        new_df = pd.DataFrame(self.new_metadata)

        if not existing_df.empty and not new_df.empty:

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            combined_df = combined_df.drop_duplicates(subset=["book_id"], keep="last")
            df = combined_df
        elif not existing_df.empty:

            df = existing_df
            logger.debug("No new metadata to add, keeping existing data")
        elif not new_df.empty:

            df = new_df
        else:

            logger.warning("No metadata to save")
            return

        df.to_csv(METADATA_FILE, index=False)
        logger.info(f"Metadata saved to {METADATA_FILE}")
        logger.info(f"Books downloaded to {RAW_DATA_DIR}")

        logger.info("\nDataset Summary:")
        logger.info(f"Total books: {len(df)}")
        logger.info(f"Unique authors: {df['author'].nunique()}")

        logger.info(f"\nBooks per genre:")
        for genre, count in df["genre"].value_counts().items():
            logger.info(f"  {genre}: {count}")

    def load_metadata(self) -> pd.DataFrame:
        if METADATA_FILE.exists():
            return pd.read_csv(METADATA_FILE)
        return pd.DataFrame()
