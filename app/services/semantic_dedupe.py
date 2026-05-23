"""Semantic dedupe and diversity helpers backed by local embeddings."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from hashlib import sha1
from typing import TypeVar

from app.core.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

T = TypeVar("T")

QUERY_EMBEDDING_TASK = (
    "Given a web search query, encode it for semantic deduplication in a research workflow. "
    "Queries that differ in entity, model, year, place, source type, comparison target, "
    "or failure mode must remain distinct."
)
LANE_EMBEDDING_TASK = (
    "Given a research lane description, encode it for overlap detection. "
    "Lanes that would gather materially different evidence, sources, or tradeoffs must remain "
    "distinct."
)
SOURCE_CARD_EMBEDDING_TASK = (
    "Given a distilled research source card, encode it for novelty-aware evidence selection. "
    "Cards that provide the same claim or evidence should be close; cards that contribute new "
    "claims, caveats, or quantitative evidence should remain distinct."
)
PROTECTED_TOKEN_RE = re.compile(
    r"\b(?:"
    r"\d{2,4}[a-z0-9-]*|"
    r"[a-z]+[0-9][a-z0-9-]*|"
    r"vs|versus|compare|comparison|"
    r"review|reviews|reddit|forum|blog|youtube|podcast|interview|"
    r"problem|problems|issue|issues|complaint|complaints|failure|failures|"
    r"reliability|durability|battery|noise|quiet|loud|warranty|return|"
    r"new|old|latest|best|worst|near|local|cheap|premium|budget|"
    r"under|over|with|without|for|against|not"
    r")\b",
    re.IGNORECASE,
)
_EMBEDDING_CACHE: dict[str, tuple[float, ...]] = {}


class SemanticEmbeddingError(RuntimeError):
    """Raised when local embedding inference cannot run."""


def _resolve_embedding_device(device_setting: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise SemanticEmbeddingError("Torch is required for semantic dedupe") from exc

    if device_setting != "auto":
        return device_setting
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache
def _load_embedding_model(model_id: str, device: str):
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SemanticEmbeddingError("Transformers and torch are required for semantic dedupe") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            local_files_only=settings.semantic_embedding_local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_id,
            local_files_only=settings.semantic_embedding_local_files_only,
        )
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        raise SemanticEmbeddingError(f"Failed to load embedding model {model_id}: {exc}") from exc

    return tokenizer, model, torch, functional


def _last_token_pool(last_hidden_states, attention_mask, torch):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _material_guard_tokens(value: str) -> set[str]:
    normalized = _normalize_text(value)
    return {token.lower() for token in PROTECTED_TOKEN_RE.findall(normalized)}


def _has_material_delta(left: str, right: str) -> bool:
    left_tokens = _material_guard_tokens(left)
    right_tokens = _material_guard_tokens(right)
    return bool(left_tokens.symmetric_difference(right_tokens))


def _cache_key(task_description: str, value: str) -> str:
    payload = f"{settings.semantic_embedding_model_id}\n{task_description}\n{value}"
    return sha1(payload.encode("utf-8")).hexdigest()


def _build_instructed_text(task_description: str, value: str) -> str:
    return f"Instruct: {task_description}\nQuery: {value}"


def embed_texts(texts: list[str], *, task_description: str) -> list[tuple[float, ...]]:
    """Embed texts with the configured local model.

    Args:
        texts: Input texts to embed.
        task_description: Instruction text passed to the query encoder.

    Returns:
        L2-normalized embedding vectors.
    """

    if not texts:
        return []

    device = _resolve_embedding_device(settings.semantic_embedding_device)
    tokenizer, model, torch, functional = _load_embedding_model(
        settings.semantic_embedding_model_id,
        device,
    )
    normalized_texts = [_normalize_text(text) for text in texts]
    embeddings: list[tuple[float, ...] | None] = [None] * len(normalized_texts)
    missing_indices: list[int] = []
    missing_inputs: list[str] = []

    for idx, text in enumerate(normalized_texts):
        cache_key = _cache_key(task_description, text)
        cached = _EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            embeddings[idx] = cached
            continue
        missing_indices.append(idx)
        missing_inputs.append(_build_instructed_text(task_description, text))

    batch_size = max(1, settings.semantic_embedding_batch_size)
    for start in range(0, len(missing_inputs), batch_size):
        batch_inputs = missing_inputs[start : start + batch_size]
        batch_indices = missing_indices[start : start + batch_size]
        batch_dict = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=settings.semantic_embedding_max_length,
            return_tensors="pt",
        )
        batch_dict = {key: value.to(model.device) for key, value in batch_dict.items()}
        try:
            with torch.inference_mode():
                outputs = model(**batch_dict)
                pooled = _last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                    torch,
                )
                pooled = functional.normalize(pooled, p=2, dim=1)
        except Exception as exc:  # noqa: BLE001
            raise SemanticEmbeddingError(f"Embedding inference failed: {exc}") from exc
        for local_idx, vector in enumerate(pooled.cpu().tolist()):
            embedding = tuple(float(value) for value in vector)
            target_idx = batch_indices[local_idx]
            cache_key = _cache_key(task_description, normalized_texts[target_idx])
            _EMBEDDING_CACHE[cache_key] = embedding
            embeddings[target_idx] = embedding

    return [embedding for embedding in embeddings if embedding is not None]


def cluster_texts_by_similarity(
    texts: list[str],
    *,
    task_description: str,
    similarity_threshold: float,
) -> list[list[int]]:
    """Cluster similar texts with a greedy cosine-threshold pass.

    Args:
        texts: Input texts.
        task_description: Embedding task instruction.
        similarity_threshold: Cosine threshold for assigning to an existing cluster.

    Returns:
        Clusters as lists of input indices.
    """

    if not texts:
        return []

    normalized = [_normalize_text(text) for text in texts]
    exact_map: dict[str, list[int]] = {}
    for idx, text in enumerate(normalized):
        exact_map.setdefault(text, []).append(idx)

    unique_indices = [indices[0] for indices in exact_map.values()]
    unique_texts = [normalized[idx] for idx in unique_indices]

    if not settings.semantic_dedupe_enabled or len(unique_texts) <= 1:
        return [indices[:] for indices in exact_map.values()]

    try:
        embeddings = embed_texts(unique_texts, task_description=task_description)
    except SemanticEmbeddingError as exc:
        logger.warning("Semantic dedupe disabled for this run: %s", exc)
        return [indices[:] for indices in exact_map.values()]

    clusters: list[list[int]] = []
    representative_indices: list[int] = []
    representative_embeddings: list[tuple[float, ...]] = []

    for unique_position, _input_idx in enumerate(unique_indices):
        assigned_cluster: int | None = None
        current_embedding = embeddings[unique_position]
        current_text = unique_texts[unique_position]
        best_similarity = -1.0

        for cluster_idx, representative_embedding in enumerate(representative_embeddings):
            similarity = _cosine_similarity(current_embedding, representative_embedding)
            representative_text = unique_texts[representative_indices[cluster_idx]]
            if similarity < similarity_threshold:
                continue
            if _has_material_delta(current_text, representative_text):
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                assigned_cluster = cluster_idx

        target_indices = exact_map[current_text]
        if assigned_cluster is None:
            clusters.append(target_indices[:])
            representative_indices.append(unique_position)
            representative_embeddings.append(current_embedding)
            continue

        clusters[assigned_cluster].extend(target_indices)

    return [sorted(cluster) for cluster in clusters]


def dedupe_items_by_text(
    items: list[T],
    *,
    text_getter,
    task_description: str,
    similarity_threshold: float,
    utility_scorer=None,
    max_items: int | None = None,
) -> list[T]:
    """Dedupe items by semantic similarity while preserving stronger representatives."""

    if len(items) <= 1:
        return items

    clusters = cluster_texts_by_similarity(
        [text_getter(item) for item in items],
        task_description=task_description,
        similarity_threshold=similarity_threshold,
    )
    if not clusters:
        return items

    scored_items: list[tuple[int, T]] = []
    for cluster in clusters:
        best_idx = min(cluster)
        if utility_scorer is not None:
            best_idx = max(
                cluster,
                key=lambda idx: (utility_scorer(items[idx]), -idx),
            )
        scored_items.append((best_idx, items[best_idx]))

    scored_items.sort(key=lambda item: item[0])
    deduped = [item for _, item in scored_items]
    if max_items is not None:
        return deduped[:max_items]
    return deduped


def mmr_rank_texts(
    texts: list[str],
    relevance_scores: list[float],
    *,
    task_description: str,
    lambda_mult: float,
) -> list[int]:
    """Return indices ordered by maximal marginal relevance."""

    if len(texts) <= 1:
        return list(range(len(texts)))
    if len(texts) != len(relevance_scores):
        raise ValueError("texts and relevance_scores must have the same length")

    try:
        embeddings = embed_texts(texts, task_description=task_description)
    except SemanticEmbeddingError as exc:
        logger.warning("MMR fallback to relevance order: %s", exc)
        return sorted(
            range(len(texts)),
            key=lambda idx: (relevance_scores[idx], len(texts[idx])),
            reverse=True,
        )

    if not embeddings:
        return list(range(len(texts)))

    scale = max(relevance_scores) or 1.0
    normalized_relevance = [score / scale for score in relevance_scores]
    selected: list[int] = []
    remaining = set(range(len(texts)))

    while remaining:
        if not selected:
            best_idx = max(
                remaining,
                key=lambda idx: (normalized_relevance[idx], len(texts[idx])),
            )
            selected.append(best_idx)
            remaining.remove(best_idx)
            continue

        best_idx = max(
            remaining,
            key=lambda idx: (
                _mmr_score(
                    idx,
                    selected,
                    embeddings,
                    normalized_relevance,
                    lambda_mult=lambda_mult,
                ),
                normalized_relevance[idx],
            ),
        )
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def _mmr_score(
    idx: int,
    selected: list[int],
    embeddings: list[tuple[float, ...]],
    relevance_scores: list[float],
    *,
    lambda_mult: float,
) -> float:
    novelty_penalty = max(_cosine_similarity(embeddings[idx], embeddings[other]) for other in selected)
    return (lambda_mult * relevance_scores[idx]) - ((1 - lambda_mult) * novelty_penalty)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(left_val * right_val for left_val, right_val in zip(left, right, strict=True))
