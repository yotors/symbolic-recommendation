"""Build and load model-independent article text-embedding sidecars.

The expensive sentence-transformer dependency belongs only to the offline
builder.  Importing this module, and especially loading an existing sidecar,
requires NumPy but never imports ``sentence_transformers`` or Torch.

Supported article sources are:

* a replay ``.json`` or ``.json.gz`` cache produced by :mod:`mind`;
* a RecZoo archive containing ``news_corpus.tsv``; or
* an extracted RecZoo ``news_corpus.tsv`` file.

The saved NPZ contains normalized float32 vectors, runtime article IDs,
source IDs, and canonical JSON provenance.  It is written to a temporary file
in the destination directory, fsynced, and atomically replaced.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np


SIDECAR_SCHEMA = "mindplex-text-embeddings-v1"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_RECIPE = "NFKC whitespace-folded title + optional double-newline + abstract"


class TextEmbeddingError(ValueError):
    """Raised when source articles or a sidecar violate the format contract."""


class SentenceEncoder(Protocol):
    """Minimal interface implemented by ``SentenceTransformer`` and test fakes."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        ...


@dataclass(frozen=True)
class TextArticle:
    """One canonical article input to the offline encoder."""

    article_id: str
    source_id: str
    text: str


@dataclass(frozen=True)
class ArticleCorpus:
    """Canonical articles plus provenance for the source that supplied them."""

    articles: tuple[TextArticle, ...]
    source: Mapping[str, Any]
    content_sha256: str


@dataclass(frozen=True)
class TextEmbeddingSidecar:
    """Validated, Torch-free in-memory view of a sidecar."""

    article_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    vectors: np.ndarray
    metadata: Mapping[str, Any]

    def as_mapping(self, namespace: str = "article_id") -> dict[str, np.ndarray]:
        """Return vectors keyed by ``article_id`` or original ``source_id``."""

        if namespace == "article_id":
            keys = self.article_ids
        elif namespace == "source_id":
            keys = self.source_ids
        else:
            raise TextEmbeddingError(
                "namespace must be either 'article_id' or 'source_id'"
            )
        return dict(zip(keys, self.vectors))


def _normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split())


def article_text(title: object, abstract: object = "") -> str:
    """Render title and abstract using the versioned sidecar text recipe."""

    normalized_title = _normalized_text(title)
    normalized_abstract = _normalized_text(abstract)
    if normalized_title and normalized_abstract:
        return f"{normalized_title}\n\n{normalized_abstract}"
    return normalized_title or normalized_abstract


def _hash_records(records: Iterable[Sequence[object]]) -> str:
    """Hash records with canonical JSON framing (no delimiter ambiguity)."""

    digest = hashlib.sha256()
    for record in records:
        payload = json.dumps(
            list(record), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _canonical_corpus(
    rows: Iterable[tuple[object, object, object, object]],
    *,
    source: Mapping[str, Any],
) -> ArticleCorpus:
    articles: list[TextArticle] = []
    seen_article_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    for raw_article_id, raw_source_id, title, abstract in rows:
        article_id = str(raw_article_id or "").strip()
        source_id = str(raw_source_id or "").strip()
        if not article_id or not source_id:
            raise TextEmbeddingError("article and source IDs must be non-empty")
        if article_id in seen_article_ids:
            raise TextEmbeddingError(f"duplicate article ID: {article_id!r}")
        if source_id in seen_source_ids:
            raise TextEmbeddingError(f"duplicate source ID: {source_id!r}")
        text = article_text(title, abstract)
        if not text:
            raise TextEmbeddingError(
                f"article {article_id!r} has neither a title nor an abstract"
            )
        seen_article_ids.add(article_id)
        seen_source_ids.add(source_id)
        articles.append(TextArticle(article_id, source_id, text))

    if not articles:
        raise TextEmbeddingError("article source contains no records")
    articles.sort(key=lambda item: (item.article_id, item.source_id))
    content_sha256 = _hash_records(
        (item.article_id, item.source_id, item.text) for item in articles
    )
    return ArticleCorpus(tuple(articles), dict(source), content_sha256)


def _read_replay_cache(path: Path) -> ArticleCorpus:
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TextEmbeddingError(f"cannot read replay cache {path}: {exc}") from exc
    raw_articles = payload.get("articles") if isinstance(payload, dict) else None
    if not isinstance(raw_articles, list):
        raise TextEmbeddingError("replay cache must contain an 'articles' list")
    rows = []
    for index, item in enumerate(raw_articles, start=1):
        if not isinstance(item, dict):
            raise TextEmbeddingError(f"replay article {index} is not an object")
        article_id = item.get("id")
        source_id = item.get("source_id", article_id)
        rows.append((article_id, source_id, item.get("title"), item.get("abstract")))
    replay_metadata = payload.get("metadata", {})
    source_metadata = {
        "kind": "mind-replay-cache",
        "path": str(path.resolve()),
        "dataset": (
            replay_metadata.get("dataset")
            if isinstance(replay_metadata, dict)
            else None
        ),
        "projection": (
            replay_metadata.get("projection")
            if isinstance(replay_metadata, dict)
            else None
        ),
    }
    return _canonical_corpus(rows, source=source_metadata)


def _tsv_rows(handle: io.TextIOBase, label: str):
    reader = csv.DictReader(handle, delimiter="\t")
    required = {"news_id", "title", "abstract"}
    missing = required.difference(reader.fieldnames or ())
    if missing:
        raise TextEmbeddingError(
            f"{label} is missing columns: {', '.join(sorted(missing))}"
        )
    for line_number, row in enumerate(reader, start=2):
        source_id = (row.get("news_id") or "").strip()
        if not source_id:
            raise TextEmbeddingError(f"{label}:{line_number}: empty news_id")
        # A raw RecZoo corpus has no adapter-specific projected ID.  Preserve
        # the source namespace in both fields; replay caches retain both.
        yield source_id, source_id, row.get("title"), row.get("abstract")


def _read_reczoo_zip(path: Path) -> ArticleCorpus:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise TextEmbeddingError(f"cannot read RecZoo archive {path}: {exc}") from exc
    with archive:
        members = [
            name for name in archive.namelist()
            if Path(name).name == "news_corpus.tsv"
        ]
        if len(members) != 1:
            raise TextEmbeddingError(
                f"RecZoo archive must contain exactly one news_corpus.tsv; found {len(members)}"
            )
        member = members[0]
        try:
            with archive.open(member) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                    rows = list(_tsv_rows(text, f"{path}!{member}"))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise TextEmbeddingError(
                f"cannot parse {path}!{member}: {exc}"
            ) from exc
    return _canonical_corpus(
        rows,
        source={
            "kind": "reczoo-zip",
            "path": str(path.resolve()),
            "member": member,
        },
    )


def _read_reczoo_tsv(path: Path) -> ArticleCorpus:
    try:
        with path.open("rt", encoding="utf-8-sig", newline="") as handle:
            rows = list(_tsv_rows(handle, str(path)))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TextEmbeddingError(f"cannot parse {path}: {exc}") from exc
    return _canonical_corpus(
        rows, source={"kind": "reczoo-tsv", "path": str(path.resolve())}
    )


def read_article_corpus(source: str | os.PathLike[str]) -> ArticleCorpus:
    """Read and canonically order articles from a supported source."""

    path = Path(source)
    if not path.is_file():
        raise TextEmbeddingError(f"article source does not exist: {path}")
    suffixes = [suffix.casefold() for suffix in path.suffixes]
    if path.suffix.casefold() == ".zip":
        return _read_reczoo_zip(path)
    if path.suffix.casefold() == ".tsv":
        return _read_reczoo_tsv(path)
    if suffixes[-2:] == [".json", ".gz"] or path.suffix.casefold() == ".json":
        return _read_replay_cache(path)
    raise TextEmbeddingError(
        "unsupported article source; expected replay .json[.gz], RecZoo .zip, or news_corpus.tsv"
    )


def _encoder_class_name(encoder: object) -> str:
    cls = type(encoder)
    return f"{cls.__module__}.{cls.__qualname__}"


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolved_model_revision(encoder: object) -> str | None:
    """Best-effort Hugging Face commit recorded on a loaded transformer."""

    named_children = getattr(encoder, "named_children", None)
    if not callable(named_children):
        return None
    for _name, module in named_children():
        auto_model = getattr(module, "auto_model", None)
        config = getattr(auto_model, "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def _load_sentence_transformer(
    model_name: str, *, revision: str | None, device: str | None
) -> SentenceEncoder:
    # Deliberately local: sidecar consumers do not need this dependency stack.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise TextEmbeddingError(
            "building requires sentence-transformers; install it in the offline encoder environment"
        ) from exc
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def _normalize_vectors(raw: object, expected_rows: int) -> np.ndarray:
    try:
        vectors = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TextEmbeddingError(f"encoder returned non-numeric vectors: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise TextEmbeddingError(
            f"encoder returned shape {vectors.shape}; expected ({expected_rows}, dimensions)"
        )
    if vectors.shape[1] <= 0:
        raise TextEmbeddingError("encoder returned zero-dimensional vectors")
    if not np.isfinite(vectors).all():
        raise TextEmbeddingError("encoder returned NaN or infinite values")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise TextEmbeddingError("encoder returned a zero or invalid vector")
    vectors = np.ascontiguousarray(vectors / norms[:, None], dtype=np.float32)
    return vectors


def _vector_sha256(
    article_ids: Sequence[str], source_ids: Sequence[str], vectors: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(
        bytes.fromhex(_hash_records(zip(article_ids, source_ids)))
    )
    canonical = np.ascontiguousarray(vectors, dtype="<f4")
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # Best-effort directory durability after the atomic rename.
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_text_embedding_sidecar(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    encoder: SentenceEncoder | None = None,
    model_name: str = DEFAULT_MODEL,
    model_revision: str | None = None,
    device: str | None = None,
    batch_size: int = 128,
    show_progress: bool = False,
) -> Mapping[str, Any]:
    """Encode a corpus and atomically write a compressed, provenanced NPZ.

    ``encoder`` is injectable so the builder can be tested without downloading
    a model.  When omitted, SentenceTransformer is imported lazily.
    """

    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<=0:
        raise TextEmbeddingError("batch_size must be a positive integer")
    corpus = read_article_corpus(source)
    active_encoder = encoder or _load_sentence_transformer(
        model_name, revision=model_revision, device=device
    )
    texts = [item.text for item in corpus.articles]
    encoded = active_encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        # Normalize here rather than trusting backend-specific behavior.
        normalize_embeddings=False,
    )
    vectors = _normalize_vectors(encoded, len(texts))
    article_ids = tuple(item.article_id for item in corpus.articles)
    source_ids = tuple(item.source_id for item in corpus.articles)
    model: dict[str, Any] = {
        "name": model_name,
        "revision": model_revision,
        "resolved_revision": _resolved_model_revision(active_encoder),
        "encoder_class": _encoder_class_name(active_encoder),
        "sentence_transformers_version": _distribution_version(
            "sentence-transformers"
        ),
        "transformers_version": _distribution_version("transformers"),
        "torch_version": _distribution_version("torch"),
    }
    metadata: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "article_count": len(article_ids),
        "dimensions": int(vectors.shape[1]),
        "dtype": "float32",
        "normalization": "l2-unit",
        "text_recipe": TEXT_RECIPE,
        "content_sha256": corpus.content_sha256,
        "vector_sha256": _vector_sha256(article_ids, source_ids, vectors),
        "source": dict(corpus.source),
        "model": model,
    }
    arrays = {
        "article_ids": np.asarray(article_ids, dtype=np.str_),
        "source_ids": np.asarray(source_ids, dtype=np.str_),
        "vectors": vectors,
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
    }
    _atomic_savez(Path(output), arrays)
    return metadata


def load_text_embedding_sidecar(
    path: str | os.PathLike[str],
    *,
    expected_model: str | None = None,
    expected_revision: str | None = None,
    expected_content_sha256: str | None = None,
    verify: bool = True,
) -> TextEmbeddingSidecar:
    """Load a sidecar without importing its encoder or any ML framework."""

    sidecar_path = Path(path)
    try:
        with np.load(sidecar_path, allow_pickle=False) as payload:
            required = {"article_ids", "source_ids", "vectors", "metadata_json"}
            missing = required.difference(payload.files)
            if missing:
                raise TextEmbeddingError(
                    f"sidecar is missing arrays: {', '.join(sorted(missing))}"
                )
            article_ids_array = np.asarray(payload["article_ids"])
            source_ids_array = np.asarray(payload["source_ids"])
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            metadata_value = np.asarray(payload["metadata_json"])
    except TextEmbeddingError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise TextEmbeddingError(f"cannot load text sidecar {sidecar_path}: {exc}") from exc

    if metadata_value.ndim != 0:
        raise TextEmbeddingError("metadata_json must be a scalar string")
    try:
        metadata = json.loads(str(metadata_value.item()))
    except (TypeError, json.JSONDecodeError) as exc:
        raise TextEmbeddingError(f"invalid sidecar metadata: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != SIDECAR_SCHEMA:
        raise TextEmbeddingError("unsupported text sidecar schema")
    if article_ids_array.ndim != 1 or source_ids_array.ndim != 1:
        raise TextEmbeddingError("sidecar IDs must be one-dimensional arrays")
    article_ids = tuple(str(value) for value in article_ids_array.tolist())
    source_ids = tuple(str(value) for value in source_ids_array.tolist())
    count = len(article_ids)
    if count == 0 or len(source_ids) != count:
        raise TextEmbeddingError("sidecar ID arrays are empty or misaligned")
    if any(not value for value in (*article_ids, *source_ids)):
        raise TextEmbeddingError("sidecar IDs must be non-empty")
    if len(set(article_ids)) != count or len(set(source_ids)) != count:
        raise TextEmbeddingError("sidecar IDs must be unique in each namespace")
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] <= 0:
        raise TextEmbeddingError("sidecar vectors and IDs are misaligned")
    if not np.isfinite(vectors).all():
        raise TextEmbeddingError("sidecar contains NaN or infinite vectors")

    model = metadata.get("model")
    if not isinstance(model, dict):
        raise TextEmbeddingError("sidecar has no model provenance")
    if expected_model is not None and model.get("name") != expected_model:
        raise TextEmbeddingError(
            f"model mismatch: expected {expected_model!r}, found {model.get('name')!r}"
        )
    if expected_revision is not None and model.get("revision") != expected_revision:
        raise TextEmbeddingError(
            "model revision mismatch: "
            f"expected {expected_revision!r}, found {model.get('revision')!r}"
        )
    if (
        expected_content_sha256 is not None
        and metadata.get("content_sha256") != expected_content_sha256
    ):
        raise TextEmbeddingError("article content provenance does not match")

    if verify:
        if metadata.get("article_count") != count:
            raise TextEmbeddingError("sidecar article_count does not match arrays")
        if metadata.get("dimensions") != vectors.shape[1]:
            raise TextEmbeddingError("sidecar dimensions do not match vectors")
        if metadata.get("dtype") != "float32":
            raise TextEmbeddingError("unsupported sidecar vector dtype")
        norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
            raise TextEmbeddingError("sidecar vectors are not L2-normalized")
        actual_vector_hash = _vector_sha256(article_ids, source_ids, vectors)
        if metadata.get("vector_sha256") != actual_vector_hash:
            raise TextEmbeddingError("sidecar vector checksum does not match")

    vectors.setflags(write=False)
    return TextEmbeddingSidecar(article_ids, source_ids, vectors, metadata)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a normalized article text-embedding NPZ sidecar."
    )
    parser.add_argument(
        "source", help="replay .json[.gz], RecZoo archive, or news_corpus.tsv"
    )
    parser.add_argument("output", help="destination .npz sidecar")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", help="SentenceTransformer device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quiet", action="store_true", help="hide encoder progress")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metadata = build_text_embedding_sidecar(
        args.source,
        args.output,
        model_name=args.model,
        model_revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        show_progress=not args.quiet,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
