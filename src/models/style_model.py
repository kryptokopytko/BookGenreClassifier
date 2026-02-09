import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import re

from ..utils.logger import get_logger
from ..utils.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_FOREST_PARAMS

logger = get_logger(__name__)

def extract_style_features(text: str) -> np.ndarray:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = text.split()

    total_chars = len(text)
    alpha_chars = sum(c.isalpha() for c in text)
    digit_chars = sum(c.isdigit() for c in text)
    space_chars = sum(c.isspace() for c in text)

    exclamation_marks = text.count('!')
    question_marks = text.count('?')
    commas = text.count(',')
    semicolons = text.count(';')
    colons = text.count(':')
    dashes = text.count('--') + text.count('—')
    quotes = text.count('"') + text.count("'")

    num_sentences = len(sentences)
    if num_sentences > 0:
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        std_sentence_length = np.std([len(s.split()) for s in sentences])
        max_sentence_length = max([len(s.split()) for s in sentences])
        min_sentence_length = min([len(s.split()) for s in sentences])
    else:
        avg_sentence_length = 0
        std_sentence_length = 0
        max_sentence_length = 0
        min_sentence_length = 0

    num_words = len(words)
    if num_words > 0:
        avg_word_length = np.mean([len(w) for w in words])
        std_word_length = np.std([len(w) for w in words])
        unique_words = len(set(word.lower() for word in words))
        vocabulary_richness = unique_words / num_words
    else:
        avg_word_length = 0
        std_word_length = 0
        unique_words = 0
        vocabulary_richness = 0

    long_words = sum(1 for w in words if len(w) > 6)
    short_words = sum(1 for w in words if len(w) <= 3)
    capitalized_words = sum(1 for w in words if w[0].isupper() if w)

    dialogue_lines = text.count('"') // 2 + text.count("'") // 2

    paragraphs = len([p for p in text.split('\n\n') if p.strip()])

    features = [

        num_words,
        num_sentences,
        total_chars,
        paragraphs,

        alpha_chars / (total_chars + 1),
        digit_chars / (total_chars + 1),
        space_chars / (total_chars + 1),

        avg_sentence_length,
        std_sentence_length,
        max_sentence_length,
        min_sentence_length,
        num_words / (num_sentences + 1),

        avg_word_length,
        std_word_length,
        vocabulary_richness,
        long_words / (num_words + 1),
        short_words / (num_words + 1),
        capitalized_words / (num_words + 1),

        exclamation_marks / (num_sentences + 1),
        question_marks / (num_sentences + 1),
        commas / (num_words + 1),
        semicolons / (num_words + 1),
        colons / (num_words + 1),
        dashes / (num_words + 1),
        quotes / (num_words + 1),

        dialogue_lines / (num_sentences + 1),
    ]

    return np.array(features)

def train_style_model():

    train_file = PROCESSED_DATA_DIR / "train.csv"
    val_file = PROCESSED_DATA_DIR / "val.csv"
    test_file = PROCESSED_DATA_DIR / "test.csv"

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    logger.info("Extracting style features from train set...")
    X_train = []
    for _, row in train_df.iterrows():
        file_path = Path(row['processed_path'])
        if not file_path.is_absolute():
            file_path = PROCESSED_DATA_DIR.parent / file_path
        try:
            text = file_path.read_text(encoding='utf-8')
            features = extract_style_features(text)
            X_train.append(features)
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            X_train.append(np.zeros(26))

    X_train = np.array(X_train)
    y_train = train_df['genre'].values

    logger.info("Extracting style features from validation set...")
    X_val = []
    for _, row in val_df.iterrows():
        file_path = Path(row['processed_path'])
        if not file_path.is_absolute():
            file_path = PROCESSED_DATA_DIR.parent / file_path
        try:
            text = file_path.read_text(encoding='utf-8')
            features = extract_style_features(text)
            X_val.append(features)
        except Exception as e:
            X_val.append(np.zeros(26))

    X_val = np.array(X_val)
    y_val = val_df['genre'].values

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logger.info("Training Random Forest on style features...")
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    val_acc = model.score(X_val_scaled, y_val)

    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    logger.info("Extracting style features from test set...")
    X_test = []
    for _, row in test_df.iterrows():
        file_path = Path(row['processed_path'])
        if not file_path.is_absolute():
            file_path = PROCESSED_DATA_DIR.parent / file_path
        try:
            text = file_path.read_text(encoding='utf-8')
            features = extract_style_features(text)
            X_test.append(features)
        except Exception as e:
            X_test.append(np.zeros(26))

    X_test = np.array(X_test)
    y_test = test_df['genre'].values
    X_test_scaled = scaler.transform(X_test)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    logger.info(f"\nTest Performance:")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1: {f1:.4f}")

    feature_names = [
        'num_words', 'num_sentences', 'total_chars', 'paragraphs',
        'alpha_ratio', 'digit_ratio', 'space_ratio',
        'avg_sentence_len', 'std_sentence_len', 'max_sentence_len', 'min_sentence_len',
        'words_per_sentence', 'avg_word_len', 'std_word_len', 'vocab_richness',
        'long_words_ratio', 'short_words_ratio', 'capitalized_ratio',
        'exclamation_ratio', 'question_ratio', 'comma_ratio', 'semicolon_ratio',
        'colon_ratio', 'dash_ratio', 'quote_ratio', 'dialogue_ratio'
    ]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]

    logger.info("\nTop 10 most important style features:")
    for i, idx in enumerate(indices, 1):
        logger.info(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")

    model_path = MODELS_DIR / "style_model.pkl"
    scaler_path = MODELS_DIR / "style_scaler.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"\nModel saved to {model_path}")

    metrics = {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return model, scaler, metrics
