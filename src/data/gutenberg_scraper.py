import requests
from bs4 import BeautifulSoup
import time
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
)

logger = get_logger(__name__)

class GutenbergScraper:
    BASE_URL = "https://www.gutenberg.org"
    SEARCH_URL = f"{BASE_URL}/ebooks/search/"
    BOOKSHELF_URL = f"{BASE_URL}/ebooks/bookshelf/"

    GENRE_BOOKSHELVES = {
        "Adventure/Action": [
            "Adventure",
            "Pirates Buccaneers Corsairs",
            "Sea and Ships",
            "Western",
        ],
        "Thriller/Horror": [
            "Horror",
            "Ghost Stories",
            "Gothic Fiction",
            "Detective Fiction",
        ],
        "Fantasy": ["Fantasy", "Mythology", "Fairy Tales"],
        "Historical Fiction": [
            "Historical Fiction",
            "World War I",
            "World War II",
            "US Civil War",
        ],
        "Science Fiction": ["Science Fiction", "Precursors of Science Fiction"],
        "Mystery/Crime": ["Detective Fiction", "Crime Fiction", "Mystery Fiction"],
        "Biography": ["Biography", "Memoirs", "US History"],
        "Romance": ["Love", "Romance"],
    }

    GENRE_QUERIES = {
        "Romance": [
            "romance love",
            "romantic fiction",
            "love story",
            "courtship marriage",
            "passion romantic",
        ],
        "Fantasy": [
            "fantasy",
            "fantasy magic",
            "fantasy fiction",
            "magical",
            "wizards dragons",
            "fairy tale",
            "mythical creatures",
            "enchanted kingdom",
        ],
        "Thriller/Horror": [
            "horror",
            "thriller",
            "scary ghost",
            "suspense terror",
            "supernatural horror",
            "haunted mysterious",
            "macabre dark",
            "creepy frightening",
        ],
        "Historical Fiction": [
            "historical fiction",
            "historical novel",
            "history war",
            "victorian era",
            "ancient rome",
            "medieval times",
        ],
        "Science Fiction": [
            "science fiction",
            "sci-fi",
            "space alien",
            "future technology",
            "robot android",
            "dystopia utopia",
            "time travel",
        ],
        "Mystery/Crime": [
            "detective mystery crime",
            "murder investigation",
            "detective story",
            "whodunit police",
            "criminal case",
            "sleuth investigation",
        ],
        "Biography": [
            "biography memoir",
            "autobiography",
            "life story",
            "personal narrative",
            "historical figure",
        ],
        "Adventure/Action": [
            "action",
            "adventure action",
            "adventure story",
            "exploration quest",
            "journey expedition",
            "treasure hunt",
            "survival wilderness",
            "heroic tale",
            "daring escape",
            "pirate sea",
            "wild west frontier",
        ],
    }

    def __init__(self, output_dir: Path = RAW_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; BookGenreClassifier/1.0; Educational Research)"
            }
        )
        self.metadata = []

    def search_books(self, query: str, max_results: int = 500) -> List[Dict]:
        books = []
        page = 1

        while len(books) < max_results:
            params = {"query": query, "submit_search": "Go!", "sort_order": "downloads"}

            if page > 1:
                params["start_index"] = (page - 1) * 25

            try:
                response = self.session.get(self.SEARCH_URL, params=params, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                book_list = soup.find_all("li", class_="booklink")

                if not book_list:
                    logger.info(f"No more results found at page {page}")
                    break

                for book_item in book_list:
                    if len(books) >= max_results:
                        break

                    link = book_item.find("a", class_="link")
                    if not link:
                        continue

                    book_url = link.get("href")
                    book_id_match = re.search(r"/ebooks/(\d+)", book_url)
                    if not book_id_match:
                        continue

                    book_id = book_id_match.group(1)

                    title_span = book_item.find("span", class_="title")
                    title = title_span.text.strip() if title_span else f"Book {book_id}"

                    author_span = book_item.find("span", class_="subtitle")
                    author = author_span.text.strip() if author_span else "Unknown"

                    books.append(
                        {
                            "book_id": book_id,
                            "title": title,
                            "author": author,
                            "url": f"{self.BASE_URL}/ebooks/{book_id}",
                        }
                    )

                page += 1
                time.sleep(1)

            except requests.RequestException as e:
                logger.error(f"Error searching books: {e}")
                break

        logger.info(f"Found {len(books)} books for query '{query}'")
        return books

    def search_bookshelf(
        self, bookshelf_name: str, max_results: int = 300
    ) -> List[Dict]:

        try:

            bookshelf_id = bookshelf_name.lower().replace(" ", "-")
            url = f"{self.BOOKSHELF_URL}{bookshelf_id}"

            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                logger.warning(
                    f"Bookshelf '{bookshelf_name}' not found (status {response.status_code})"
                )
                return []

            soup = BeautifulSoup(response.text, "html.parser")

            books = []
            book_list = soup.find_all("li", class_="booklink")

            for book_item in book_list[:max_results]:

                link = book_item.find("a", class_="link")
                if not link:
                    continue

                book_url = link.get("href")
                book_id_match = re.search(r"/ebooks/(\d+)", book_url)
                if not book_id_match:
                    continue

                book_id = book_id_match.group(1)

                title_span = book_item.find("span", class_="title")
                title = title_span.text.strip() if title_span else f"Book {book_id}"

                author_span = book_item.find("span", class_="subtitle")
                author = author_span.text.strip() if author_span else "Unknown"

                books.append(
                    {
                        "book_id": book_id,
                        "title": title,
                        "author": author,
                        "url": f"{self.BASE_URL}/ebooks/{book_id}",
                    }
                )

            logger.info(f"Found {len(books)} books in bookshelf '{bookshelf_name}'")
            return books

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
                    f"Book {book_id} too short ({word_count} < {min_length} words), skipping"
                )
                return None

            if word_count > MAX_BOOK_LENGTH:
                logger.info(f"Book {book_id} too long ({word_count} words), skipping")
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
        self, genre: str, target_count: int = BOOKS_PER_GENRE
    ) -> List[Dict]:

        queries = self.GENRE_QUERIES.get(genre, [genre])
        if isinstance(queries, str):
            queries = [queries]

        existing_metadata = self.load_metadata()
        existing_books = {}
        if not existing_metadata.empty:

            genre_metadata = existing_metadata[existing_metadata["genre"] == genre]

            existing_books = {
                str(row["book_id"]): row.to_dict()
                for _, row in genre_metadata.iterrows()
            }
            logger.info(
                f"Found {len(existing_books)} existing books for {genre} in metadata"
            )

        all_search_results = []
        seen_book_ids = set()

        for query in queries:
            logger.info(f"Searching with query: '{query}'")
            results = self.search_books(query, max_results=target_count * 2)

            for book in results:
                book_id = book["book_id"]
                if book_id not in seen_book_ids:
                    all_search_results.append(book)
                    seen_book_ids.add(book_id)

            logger.info(f"Total unique books found so far: {len(all_search_results)}")

        bookshelves = self.GENRE_BOOKSHELVES.get(genre, [])
        for bookshelf in bookshelves:
            logger.info(f"Searching bookshelf: '{bookshelf}'")
            results = self.search_bookshelf(bookshelf, max_results=target_count)

            for book in results:
                book_id = book["book_id"]
                if book_id not in seen_book_ids:
                    all_search_results.append(book)
                    seen_book_ids.add(book_id)

            logger.info(f"Total unique books found so far: {len(all_search_results)}")

            if len(all_search_results) >= target_count * 3:
                break

        search_results = all_search_results
        logger.info(f"Total search results for {genre}: {len(search_results)} books")

        books_needed = target_count - len(existing_books)
        logger.info(
            f"Need {books_needed} new books (target: {target_count}, existing: {len(existing_books)})"
        )

        if books_needed <= 0:
            logger.info(
                f"Already have {len(existing_books)} books for {genre}, target met!"
            )
            return list(existing_books.values())

        rare_genres = ["Adventure/Action", "Thriller/Horror", "Fantasy"]
        if genre in rare_genres:
            min_book_length = 7000
            logger.info(
                f"⚠️  Using lower minimum ({min_book_length} words) for rare genre: {genre}"
            )
        else:
            min_book_length = MIN_BOOK_LENGTH

        if len(search_results) < books_needed:
            logger.warning(
                f"⚠️  Only found {len(search_results)} candidates for {genre}, "
                f"but need {books_needed} books. Will download what we can."
            )

        downloaded_books = list(existing_books.values())
        new_downloads = 0
        authors_seen = set(book.get("author", "Unknown") for book in downloaded_books)

        progress_bar = tqdm(
            search_results,
            desc=f"Downloading {genre}",
            total=min(len(search_results), books_needed),
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
                logger.info(
                    f"Book {book_id} already exists (verified), skipping download"
                )
                continue

            if file_path.exists():

                try:
                    text = file_path.read_text(encoding="utf-8")
                    word_count = len(text.split())
                    logger.info(
                        f"Book {book_id} file exists but not in metadata, adding"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error reading existing book {book_id}: {e}, re-downloading"
                    )

                    result = self.download_book(
                        book_id, title, min_length=min_book_length
                    )
                    if result is None:
                        continue
                    text, word_count = result
                    file_path = self.save_book(book_id, text, genre)
            else:

                result = self.download_book(book_id, title, min_length=min_book_length)

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
                self.metadata.append(metadata_entry)
                new_downloads += 1

            authors_seen.add(author)

            progress_bar.set_postfix(
                {
                    "total": len(downloaded_books),
                    "new": new_downloads,
                    "authors": len(authors_seen),
                }
            )

            time.sleep(2)

        logger.info(f"Genre {genre} complete:")
        logger.info(f"  Total books: {len(downloaded_books)}")
        logger.info(f"  New downloads: {new_downloads}")
        logger.info(f"  Existing books: {len(existing_books)}")
        logger.info(f"  Unique authors: {len(authors_seen)}")

        if len(downloaded_books) < target_count:
            logger.warning(
                f"⚠️  Only got {len(downloaded_books)}/{target_count} books for {genre}. "
                f"Gutenberg may not have enough books in this category."
            )

        return downloaded_books

    def scrape_all_genres(
        self, genres: List[str], books_per_genre: int = BOOKS_PER_GENRE
    ):
        logger.info(f"Target: {books_per_genre} books per genre")

        for genre in genres:
            try:
                self.scrape_genre(genre, books_per_genre)
            except Exception as e:
                logger.error(f"Error scraping genre {genre}: {e}")
                continue

        self.save_metadata()

        logger.info(f"Scraping complete. Total books: {len(self.metadata)}")

    def save_metadata(self):

        existing_df = self.load_metadata()

        new_df = pd.DataFrame(self.metadata)

        if not existing_df.empty and not new_df.empty:

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            combined_df = combined_df.drop_duplicates(subset=["book_id"], keep="last")
            df = combined_df
        elif not existing_df.empty:

            df = existing_df
            logger.info("No new metadata to add, keeping existing data")
        elif not new_df.empty:

            df = new_df
        else:

            logger.warning("No metadata to save")
            return

        df.to_csv(METADATA_FILE, index=False)
        logger.info(f"Metadata saved to {METADATA_FILE}")

        logger.info("\nDataset Summary:")
        logger.info(f"Total books: {len(df)}")
        logger.info(f"\nBooks per genre:")
        for genre, count in df["genre"].value_counts().items():
            logger.info(f"  {genre}: {count}")
        logger.info(f"\nUnique authors: {df['author'].nunique()}")

    def load_metadata(self) -> pd.DataFrame:
        if METADATA_FILE.exists():
            return pd.read_csv(METADATA_FILE)
        return pd.DataFrame()
