import stanza
from pathlib import Path
from typing import List, Dict, Optional
import pickle
from collections import Counter
import pandas as pd
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config import (
    STANZA_LANGUAGE, STANZA_PROCESSORS, STANZA_USE_GPU,
    PROCESSED_DATA_DIR
)

logger = get_logger(__name__)

class POSAnalyzer:

    def __init__(self, use_gpu: bool = STANZA_USE_GPU):
        logger.info("Downloading models if needed (this may take a few minutes)...")

        try:

            stanza.download(STANZA_LANGUAGE, verbose=False)

            self.nlp = stanza.Pipeline(
                STANZA_LANGUAGE,
                processors=STANZA_PROCESSORS,
                use_gpu=use_gpu,
                verbose=False
            )
            logger.info("Stanza pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Stanza: {e}")
            raise

    def analyze_text(self, text: str) -> stanza.Document:
        if len(text) > 1000000:
            logger.warning(f"Text is very long ({len(text)} chars), processing may be slow")

        doc = self.nlp(text)
        return doc

    def get_pos_counts(self, doc: stanza.Document) -> Dict[str, int]:

        for sentence in doc.sentences:
            for word in sentence.words:
                pos_counts[word.upos] += 1

        return dict(pos_counts)

    def get_lemmas(self, doc: stanza.Document) -> List[str]:

        for sentence in doc.sentences:
            for word in sentence.words:
                lemmas.append(word.lemma.lower())

        return lemmas

    def get_named_entities(self, doc: stanza.Document) -> List[Dict[str, str]]:

        for sentence in doc.sentences:
            for entity in sentence.ents:
                entities.append({
                    'text': entity.text,
                    'type': entity.type
                })

        return entities

    def remove_named_entities(self, doc: stanza.Document) -> str:

        for sentence in doc.sentences:
            for word in sentence.words:

                if word.upos != 'PROPN':
                    tokens.append(word.text)

        return ' '.join(tokens)

    def get_sentence_lengths(self, doc: stanza.Document) -> List[int]:
        return [len(sentence.words) for sentence in doc.sentences]

    def calculate_pos_ratios(self, pos_counts: Dict[str, int]) -> Dict[str, float]:
        total = sum(pos_counts.values())
        if total == 0:
            return {}

        adj_count = pos_counts.get('ADJ', 0)
        noun_count = pos_counts.get('NOUN', 0)
        verb_count = pos_counts.get('VERB', 0)
        propn_count = pos_counts.get('PROPN', 0)
        adv_count = pos_counts.get('ADV', 0)
        pron_count = pos_counts.get('PRON', 0)

        ratios = {
            'adj_noun_ratio': adj_count / noun_count if noun_count > 0 else 0,
            'adj_verb_ratio': adj_count / verb_count if verb_count > 0 else 0,
            'noun_verb_ratio': noun_count / verb_count if verb_count > 0 else 0,
            'propn_percentage': 100 * propn_count / total,
            'adv_percentage': 100 * adv_count / total,
            'pron_percentage': 100 * pron_count / total,
            'adj_percentage': 100 * adj_count / total,
            'noun_percentage': 100 * noun_count / total,
            'verb_percentage': 100 * verb_count / total
        }

        return ratios

    def analyze_book(self, text: str, remove_propn: bool = False) -> Dict:

        doc = self.analyze_text(text)

        pos_counts = self.get_pos_counts(doc)

        lemmas = self.get_lemmas(doc)

        entities = self.get_named_entities(doc)

        sentence_lengths = self.get_sentence_lengths(doc)

        pos_ratios = self.calculate_pos_ratios(pos_counts)

        text_without_propn = None
        if remove_propn:
            text_without_propn = self.remove_named_entities(doc)

        results = {
            'pos_counts': pos_counts,
            'pos_ratios': pos_ratios,
            'lemmas': lemmas,
            'entities': entities,
            'sentence_lengths': sentence_lengths,
            'num_sentences': len(doc.sentences),
            'text_without_propn': text_without_propn
        }

        return results

    def save_analysis(self, results: Dict, output_path: Path):
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Analysis saved to {output_path}")

    def load_analysis(self, input_path: Path) -> Dict:
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
        return results

    def batch_analyze(
        self,
        texts: List[str],
        output_dir: Optional[Path] = None,
        remove_propn: bool = False
    ) -> List[Dict]:

        for i, text in enumerate(tqdm(texts, desc="Analyzing texts")):
            try:
                result = self.analyze_book(text, remove_propn=remove_propn)
                results.append(result)

                if output_dir:
                    output_path = Path(output_dir) / f"analysis_{i}.pkl"
                    self.save_analysis(result, output_path)

            except Exception as e:
                logger.error(f"Error analyzing text {i}: {e}")
                results.append(None)

        return results

def analyze_dataset(
    metadata_file: Path,
    output_dir: Path = PROCESSED_DATA_DIR / "pos_analysis",
    remove_propn: bool = False
) -> pd.DataFrame:
    df = pd.read_csv(metadata_file)
    logger.info(f"Analyzing {len(df)} books")

    analyzer = POSAnalyzer()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_stats = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="POS analysis"):
        try:

            text_path = Path(row['processed_path'])
            if not text_path.is_absolute():
                text_path = PROCESSED_DATA_DIR.parent / text_path

            text = text_path.read_text(encoding='utf-8')

            result = analyzer.analyze_book(text, remove_propn=remove_propn)

            analysis_path = output_dir / f"{row['book_id']}.pkl"
            analyzer.save_analysis(result, analysis_path)

            stats = {
                'book_id': row['book_id'],
                **result['pos_ratios'],
                'num_sentences': result['num_sentences'],
                'num_entities': len(result['entities']),
                'analysis_path': str(analysis_path.relative_to(output_dir.parent))
            }
            pos_stats.append(stats)

        except Exception as e:
            logger.error(f"Error processing book {row['book_id']}: {e}")
            continue

    pos_df = pd.DataFrame(pos_stats)

    stats_file = output_dir / "pos_stats.csv"
    pos_df.to_csv(stats_file, index=False)
    logger.info(f"POS statistics saved to {stats_file}")

    return pos_df
