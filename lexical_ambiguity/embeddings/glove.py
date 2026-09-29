"""GloVe target and lexical-context representations."""

from __future__ import annotations

import hashlib
import mmap
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from lexical_ambiguity.config import ContextWindow, GloveConfig
from lexical_ambiguity.types import Side, WicExample

FloatVector = NDArray[np.float32]
WORD_PATTERN = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", flags=re.UNICODE)
GLOVE_ROW_PATTERN = re.compile(rb"(?m)^(\S+) ")


class GloveError(ValueError):
    """Raised for malformed vectors or unsafe GloVe artifacts."""


@dataclass(frozen=True, slots=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class StaticEncoding:
    vector: FloatVector | None
    used_tokens: tuple[str, ...]
    oov_count: int
    used_target_fallback: bool


class EmbeddingStore:
    """A small in-memory subset of a larger text embedding file."""

    def __init__(self, vectors: dict[str, NDArray], dimensions: int) -> None:
        if dimensions <= 0:
            raise GloveError("dimensions must be positive")
        normalized: dict[str, FloatVector] = {}
        for token, raw_vector in vectors.items():
            vector = np.asarray(raw_vector, dtype=np.float32)
            if vector.shape != (dimensions,):
                raise GloveError(
                    f"vector {token!r} has shape {vector.shape}; expected {(dimensions,)}"
                )
            if not np.isfinite(vector).all():
                raise GloveError(f"vector {token!r} contains non-finite values")
            vector.setflags(write=False)
            normalized[token.casefold()] = vector
        self._vectors = normalized
        self.dimensions = dimensions

    def __len__(self) -> int:
        return len(self._vectors)

    def lookup(self, token: str) -> FloatVector | None:
        return self._vectors.get(token.casefold())

    def to_arrays(self) -> tuple[NDArray[np.str_], NDArray[np.float32]]:
        """Return a stable, pickle-free representation suitable for the cache."""

        tokens = sorted(self._vectors)
        vectors = (
            np.stack([self._vectors[token] for token in tokens])
            if tokens
            else np.empty((0, self.dimensions), dtype=np.float32)
        )
        return np.asarray(tokens, dtype=np.str_), vectors.astype(np.float32, copy=False)

    @classmethod
    def from_text(
        cls,
        path: Path,
        *,
        dimensions: int,
        vocabulary: set[str] | None = None,
    ) -> EmbeddingStore:
        requested = {token.casefold() for token in vocabulary} if vocabulary is not None else None
        vectors: dict[str, FloatVector] = {}
        try:
            with Path(path).open("rb") as handle, mmap.mmap(
                handle.fileno(), length=0, access=mmap.ACCESS_READ
            ) as mapped:
                for line_number, match in enumerate(
                    GLOVE_ROW_PATTERN.finditer(mapped), start=1
                ):
                    try:
                        token = match.group(1).decode("utf-8").casefold()
                    except UnicodeDecodeError as error:
                        raise GloveError(
                            f"{path}:{line_number} contains an invalid UTF-8 token"
                        ) from error
                    if requested is not None and token not in requested:
                        continue
                    line_end = mapped.find(b"\n", match.end())
                    if line_end < 0:
                        line_end = len(mapped)
                    raw_values = mapped[match.end() : line_end]
                    vector = np.fromstring(raw_values, dtype=np.float32, sep=" ")
                    if len(vector) != dimensions:
                        raise GloveError(
                            f"{path}:{line_number} has {len(vector)} values; "
                            f"expected {dimensions}"
                        )
                    vectors[token] = vector
                    if requested is not None and len(vectors) == len(requested):
                        break
        except OSError as error:
            raise GloveError(f"could not read GloVe file {path}: {error}") from error
        return cls(vectors, dimensions)


def word_tokens(sentence: str) -> tuple[TokenSpan, ...]:
    return tuple(
        TokenSpan(match.group(), match.start(), match.end())
        for match in WORD_PATTERN.finditer(sentence)
    )


def context_tokens(
    example: WicExample,
    *,
    side: Side,
    window: ContextWindow,
) -> tuple[TokenSpan, ...]:
    """Collect nearby words, leaving out the word whose meaning we are testing."""
    sentence = example.sentence(side)
    target_start, target_end = example.target_span(side)
    tokens = word_tokens(sentence)
    target_indices = [
        index
        for index, token in enumerate(tokens)
        if token.end > target_start and token.start < target_end
    ]
    if not target_indices:
        raise GloveError(
            f"example {example.split}:{example.idx} side {side}: no word token overlaps "
            f"target span {target_start}:{target_end}"
        )

    if window == "sentence":
        eligible = range(len(tokens))
    else:
        left = max(0, min(target_indices) - window)
        right = min(len(tokens), max(target_indices) + window + 1)
        eligible = range(left, right)
    target_set = set(target_indices)
    return tuple(tokens[index] for index in eligible if index not in target_set)


def collect_vocabulary(examples: Iterable[WicExample]) -> set[str]:
    vocabulary: set[str] = set()
    for example in examples:
        vocabulary.add(example.word.casefold())
        for sentence in (example.sentence1, example.sentence2):
            vocabulary.update(token.text.casefold() for token in word_tokens(sentence))
    return vocabulary


class StaticTargetEncoder:
    """Give the target its dictionary vector, regardless of the sentence."""

    def __init__(self, store: EmbeddingStore) -> None:
        self.store = store

    def encode(self, example: WicExample, side: Side) -> FloatVector | None:
        del side  # The identical lookup is the point of this diagnostic.
        return self.store.lookup(example.word)


class StaticContextEncoder:
    """Represent a target by the words around it, using an ordinary average."""

    def __init__(self, store: EmbeddingStore, *, window: ContextWindow) -> None:
        self.store = store
        self.window = window

    def encode(self, example: WicExample, side: Side) -> StaticEncoding:
        tokens = context_tokens(example, side=side, window=self.window)
        used_tokens: list[str] = []
        vectors: list[FloatVector] = []
        for token in tokens:
            normalized = token.text.casefold().replace("\u2019", "'")
            vector = self.store.lookup(normalized)
            if vector is not None:
                used_tokens.append(normalized)
                vectors.append(vector)

        if vectors:
            averaged = np.mean(np.stack(vectors), axis=0, dtype=np.float32)
            return StaticEncoding(
                vector=averaged,
                used_tokens=tuple(used_tokens),
                oov_count=len(tokens) - len(vectors),
                used_target_fallback=False,
            )

        # No nearby GloVe words here. Use the target instead so this pair still
        # counts in the results; a zero vector would not give a valid cosine.
        return StaticEncoding(
            vector=self.store.lookup(example.word),
            used_tokens=(),
            oov_count=len(tokens),
            used_target_fallback=True,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_glove_file(config: GloveConfig, raw_dir: Path) -> Path:
    """Download the GloVe archive if needed and extract only the selected member."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "glove.6B.zip"
    destination = raw_dir / config.member
    if not archive.exists():
        with tempfile.NamedTemporaryFile(
            prefix="glove.", suffix=".part", dir=raw_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with httpx.stream(
                    "GET", config.url, follow_redirects=True, timeout=600.0
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        temporary.write(chunk)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        temporary_path.replace(archive)
    if config.sha256 is not None:
        actual = _sha256(archive)
        if actual != config.sha256:
            raise GloveError(f"GloVe checksum mismatch: expected {config.sha256}, got {actual}")
    if not destination.exists():
        try:
            with zipfile.ZipFile(archive) as bundle:
                if config.member not in bundle.namelist():
                    raise GloveError(f"GloVe archive does not contain {config.member}")
                with (
                    bundle.open(config.member) as source,
                    tempfile.NamedTemporaryFile(
                        prefix=f"{config.member}.", suffix=".part", dir=raw_dir, delete=False
                    ) as temporary,
                ):
                    temporary_path = Path(temporary.name)
                    shutil.copyfileobj(source, temporary)
                temporary_path.replace(destination)
        except zipfile.BadZipFile as error:
            raise GloveError(f"invalid GloVe ZIP archive: {archive}") from error
    return destination
