"""POS tagging for the POS-guided denoising strategy (paper 3.2.1).

Tagging happens at whitespace-word level *before* WordPiece so that tags align
one-to-one with the units the corruption function deletes. spaCy's UPOS tagset
is used because Eq. 1 includes ``AUX``, which NLTK's universal tagset folds into
``VERB``.

Tags are cached on disk keyed by a hash of the sentence, so repeated epochs and
repeated runs over the same corpus tag only once.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback used when spaCy or its model is unavailable. Tagging every token as
# ``X`` makes S_POS(x) empty, so Eq. 3 falls back to Uniform(V_all) -- i.e. the
# plain random-infilling ablation ("BERT w/ IN" in Table 8).
_UNKNOWN_TAG = "X"


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


class POSTagger:
    """Lazy spaCy wrapper with a disk cache.

    ``enabled=False`` (or a missing spaCy install) degrades to the unknown tag,
    which is exactly the random span-start ablation rather than an error.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        cache_dir: str | Path | None = ".cache/pos",
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._nlp = None
        self._loaded = False
        self._mem: dict[str, list[str]] = {}
        self._cache_path: Path | None = None
        self._dirty = False
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = self.cache_dir / f"{model_name}.jsonl"
            self._load_cache()

    # -- cache ------------------------------------------------------------- #

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        with self._cache_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._mem[record["k"]] = record["t"]
        logger.info("loaded %d cached POS sequences", len(self._mem))

    def flush(self) -> None:
        """Rewrite the cache file. Cheap; the cache is small relative to a corpus."""
        if self._cache_path is None or not self._dirty:
            return
        tmp = self._cache_path.with_suffix(".tmp")
        with tmp.open("w") as fh:
            for key, tags in self._mem.items():
                fh.write(json.dumps({"k": key, "t": tags}) + "\n")
        tmp.replace(self._cache_path)
        self._dirty = False

    # -- spaCy ------------------------------------------------------------- #

    def _ensure_model(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.enabled:
            return
        try:
            import spacy

            # Only the tagger is needed; disabling the rest is a large speedup.
            self._nlp = spacy.load(
                self.model_name, disable=["parser", "ner", "lemmatizer", "textcat"]
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "spaCy model %r unavailable (%s); POS-guided span selection will "
                "fall back to uniform span starts (the 'w/ IN' ablation).",
                self.model_name,
                exc,
            )
            self._nlp = None

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._nlp is not None

    # -- tagging ----------------------------------------------------------- #

    def tag(self, text: str) -> list[str]:
        """Return one UPOS tag per whitespace token of ``text``."""
        return self.tag_batch([text])[0]

    def tag_batch(self, texts: list[str]) -> list[list[str]]:
        """Tag a batch, consulting the cache first and spaCy only for misses."""
        results: list[list[str] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self._mem.get(self._key(text))
            if cached is not None:
                results[i] = cached
            else:
                misses.append(i)

        if misses:
            self._ensure_model()
            if self._nlp is None:
                for i in misses:
                    tags = [_UNKNOWN_TAG] * len(whitespace_tokenize(texts[i]))
                    results[i] = tags
            else:
                batch = [texts[i] for i in misses]
                for i, doc in zip(misses, self._nlp.pipe(batch, batch_size=64)):
                    tags = self._align_to_whitespace(texts[i], doc)
                    results[i] = tags
                    self._mem[self._key(texts[i])] = tags
                    self._dirty = True

        return [r if r is not None else [] for r in results]

    @staticmethod
    def _align_to_whitespace(text: str, doc) -> list[str]:
        """Project spaCy's sub-word-ish tokens onto whitespace tokens.

        spaCy splits punctuation and clitics off, so a whitespace token can map
        to several spaCy tokens. The whitespace token takes the tag of its first
        content-bearing spaCy token, which is the one the span-start rule cares
        about.
        """
        words = whitespace_tokenize(text)
        tags = [_UNKNOWN_TAG] * len(words)
        if not words:
            return tags

        # Character offset of each whitespace token.
        bounds: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            bounds.append((start, start + len(word)))
            cursor = start + len(word)

        # Gather every spaCy token overlapping each whitespace token.
        overlapping: list[list[str]] = [[] for _ in words]
        widx = 0
        for token in doc:
            if token.is_space:
                continue
            start = token.idx
            while widx < len(words) and start >= bounds[widx][1]:
                widx += 1
            if widx >= len(words):
                break
            if start >= bounds[widx][0]:
                overlapping[widx].append(token.pos_)

        skip = {"PUNCT", "SPACE", "SYM"}
        for i, candidates in enumerate(overlapping):
            if not candidates:
                continue
            content = next((p for p in candidates if p not in skip), None)
            tags[i] = content if content is not None else candidates[0]
        return tags
