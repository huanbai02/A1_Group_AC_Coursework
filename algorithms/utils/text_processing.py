"""Lightweight text processing utilities for the initial coursework pipeline."""
import re
from typing import Iterable
BASIC_STOPWORDS = {"a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its","of","on","that","the","to","was","were","will","with","this","these","those","or","but","not","we","you","they","i","our","their"}

def basic_clean_text(text: object, lowercase: bool = True) -> str:
    """Apply simple reproducible text cleaning and whitespace normalization."""
    if text is None:
        return ""
    cleaned = str(text)
    if lowercase:
        cleaned = cleaned.lower()
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def simple_tokenize(text: object) -> list[str]:
    """Tokenize text with a lightweight regular expression."""
    return re.findall(r"[A-Za-z0-9_']+", basic_clean_text(text))

def remove_stopwords(tokens: Iterable[str], stopwords: set[str] | None = None) -> list[str]:
    """Remove simple English stopwords from tokens."""
    stopword_set = BASIC_STOPWORDS if stopwords is None else stopwords
    return [token for token in tokens if token.lower() not in stopword_set]
