"""
preprocessor.py
───────────────
Cleans raw social-media text before it is fed into the model.
Steps applied in order:
  1. Lowercase
  2. Remove HTML entities (&amp; → &, etc.)
  3. Remove URLs
  4. Remove @mentions
  5. Remove #hashtag symbols (keeps the word itself)
  6. Expand common English contractions
  7. Remove characters that are not letters, digits, or spaces
  8. Collapse repeated characters (loooove → loove)
  9. Strip extra whitespace
"""

import re
import html
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── NLTK resource bootstrap (downloads only if not already present) ─────────
def _ensure_nltk_resources():
    resources = [
        ("corpora/stopwords",      "stopwords"),
        ("corpora/wordnet",        "wordnet"),
        ("tokenizers/punkt",       "punkt"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_resources()

# ── Contraction map ─────────────────────────────────────────────────────────
_CONTRACTIONS = {
    r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not", r"\bisn't\b": "is not",
    r"\baren't\b": "are not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
    r"\bhadn't\b": "had not", r"\bhasn't\b": "has not", r"\bhaven't\b": "have not",
    r"\bcouldn't\b": "could not", r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not", r"\bi'm\b": "i am", r"\bi've\b": "i have",
    r"\bi'll\b": "i will", r"\bi'd\b": "i would", r"\byou're\b": "you are",
    r"\byou've\b": "you have", r"\byou'll\b": "you will", r"\byou'd\b": "you would",
    r"\bhe's\b": "he is", r"\bshe's\b": "she is", r"\bit's\b": "it is",
    r"\bthey're\b": "they are", r"\bthey've\b": "they have",
    r"\bthey'll\b": "they will", r"\bthey'd\b": "they would",
    r"\bwe're\b": "we are", r"\bwe've\b": "we have",
    r"\bwe'll\b": "we will", r"\bwe'd\b": "we would",
    r"\bthat's\b": "that is", r"\bthere's\b": "there is",
    r"\bwhat's\b": "what is", r"\bwho's\b": "who is",
    r"\blet's\b": "let us",
}


class TextPreprocessor:
    """
    Stateless text cleaner. Call `clean(text)` on a single string, or
    `transform(series)` on a pandas Series.

    Parameters
    ----------
    remove_stopwords : bool
        Whether to remove common English stopwords.
        For short social-media messages, keeping stopwords often helps
        context (e.g., "you are" vs just "are"), so this defaults to False.
    lemmatize : bool
        Whether to lemmatize tokens. Slightly slower but reduces vocabulary size.
    """

    def __init__(self, remove_stopwords: bool = False, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self._stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None

    # ── Public API ──────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = self._decode_html(text)
        text = text.lower()
        text = self._remove_urls(text)
        text = self._remove_mentions(text)
        text = self._remove_hashtag_symbols(text)
        text = self._expand_contractions(text)
        text = self._remove_non_alpha(text)
        text = self._collapse_repeated_chars(text)
        text = self._normalize_unicode(text)
        if self.lemmatize or self.remove_stopwords:
            text = self._tokenize_and_filter(text)
        return text.strip()

    def transform(self, series):
        """Apply clean() to every element of a pandas Series."""
        return series.apply(self.clean)

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _decode_html(text: str) -> str:
        return html.unescape(text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", " ", text)

    @staticmethod
    def _remove_mentions(text: str) -> str:
        return re.sub(r"@\w+", " ", text)

    @staticmethod
    def _remove_hashtag_symbols(text: str) -> str:
        # Keep the word after #, remove only the symbol
        return re.sub(r"#(\w+)", r"\1", text)

    @staticmethod
    def _expand_contractions(text: str) -> str:
        for pattern, replacement in _CONTRACTIONS.items():
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _remove_non_alpha(text: str) -> str:
        # Keep letters, digits, and spaces
        return re.sub(r"[^a-z0-9\s]", " ", text)

    @staticmethod
    def _collapse_repeated_chars(text: str) -> str:
        # "loooove" → "loove" (keeps 2 max so "looool" still feels emphatic)
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    def _tokenize_and_filter(self, text: str) -> str:
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]
        if self.lemmatize and self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
