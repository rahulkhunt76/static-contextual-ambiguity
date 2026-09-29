"""End-to-end orchestration for the frozen WiC experiment protocol."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split

from lexical_ambiguity.config import ContextWindow, ExperimentConfig, load_config
from lexical_ambiguity.data import (
    audit_examples,
    ensure_wic_data,
    load_wic_split,
    validate_expected_release,
    validate_no_cross_split_pairs,
)
from lexical_ambiguity.embeddings.bert import BertRepresentations, BertTargetEncoder
from lexical_ambiguity.embeddings.glove import (
    EmbeddingStore,
    StaticContextEncoder,
    StaticTargetEncoder,
    collect_vocabulary,
    ensure_glove_file,
)
from lexical_ambiguity.evaluation.bootstrap import bootstrap_intervals, paired_differences
from lexical_ambiguity.evaluation.metrics import MetricSet, evaluate_scores
from lexical_ambiguity.evaluation.threshold import make_tuning_split, select_threshold
from lexical_ambiguity.types import WicExample
from lexical_ambiguity.utils import (
    CACHE_SCHEMA_VERSION,
    atomic_write_json,
    canonical_hash,
    environment_info,
    fingerprint_examples,
    load_array_cache,
    save_array_cache,
    set_global_seed,
)

BertEncoderFactory = Callable[[ExperimentConfig], BertTargetEncoder]


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    name: str
    threshold: float
    tune_macro_f1: float
    tune_accuracy: float
    holdout_macro_f1: float
    holdout_accuracy: float
    holdout_roc_auc: float
    imputation_median: float


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    selection_ledger: Path
    metrics: Path
    bootstrap_ci: Path
    paired_differences: Path
    predictions: Path
    layer_metrics: Path
    raw_scores: Path
    config: Path
    environment: Path


def _labels(examples: Sequence[WicExample]) -> NDArray[np.bool_]:
    if any(example.label is None for example in examples):
        raise ValueError("experiment splits must be labeled")
    return np.asarray([bool(example.label) for example in examples], dtype=np.bool_)


def pairwise_cosine_scores(vectors: ArrayLike) -> NDArray[np.float64]:
    """Compute one cosine per ``(example, side, dimension)`` vector pair."""

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != 2 or array.shape[2] == 0:
        raise ValueError("vectors must have shape (examples, 2, dimensions)")
    if not np.isfinite(array).all():
        raise ValueError("vectors contain non-finite values")
    first = array[:, 0, :]
    second = array[:, 1, :]
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    if np.any(denominator == 0.0):
        raise ValueError("cosine similarity is undefined for zero vectors")
    return np.clip(np.einsum("ij,ij->i", first, second) / denominator, -1.0, 1.0)


def stratified_limit(
    examples: Sequence[WicExample], limit: int | None, *, seed: int
) -> list[WicExample]:
    """Take a deterministic label-stratified subset while preserving source order."""

    rows = list(examples)
    if limit is None or limit >= len(rows):
        return rows
    if limit < 4:
        raise ValueError("a stratified experiment limit must be at least four")
    labels = _labels(rows)
    indices = np.arange(len(rows), dtype=np.int64)
    chosen, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=labels,
    )
    return [rows[index] for index in np.sort(chosen)]


def fill_missing_scores(
    train_scores: ArrayLike, other_scores: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Impute missing scores with the median learned from the first array only."""

    train = np.asarray(train_scores, dtype=np.float64)
    other = np.asarray(other_scores, dtype=np.float64)
    if train.ndim != 1 or other.ndim != 1 or len(train) == 0:
        raise ValueError("score arrays must be one-dimensional and training must be non-empty")
    if np.isinf(train).any() or np.isinf(other).any():
        raise ValueError("scores may be missing but cannot be infinite")
    finite = train[np.isfinite(train)]
    if len(finite) == 0:
        raise ValueError("training scores contain no finite value for imputation")
    median = float(np.median(finite))
    return (
        np.where(np.isfinite(train), train, median),
        np.where(np.isfinite(other), other, median),
        median,
    )


def choose_candidate(
    score_sets: dict[str, ArrayLike],
    labels: ArrayLike,
    *,
    tune_indices: NDArray[np.int64],
    holdout_indices: NDArray[np.int64],
    simplicity_order: Sequence[str],
) -> tuple[str, list[CandidateRecord]]:
    """Fit on tune rows, rank on holdout rows, and break ties by declaration order."""

    label_array = np.asarray(labels, dtype=np.bool_)
    if set(score_sets) != set(simplicity_order):
        raise ValueError("simplicity_order must name every candidate exactly once")
    if set(tune_indices.tolist()) & set(holdout_indices.tolist()):
        raise ValueError("tune and holdout indices overlap")
    records: list[CandidateRecord] = []
    for name in simplicity_order:
        scores = np.asarray(score_sets[name], dtype=np.float64)
        if scores.shape != label_array.shape:
            raise ValueError(f"candidate {name} does not match label shape")
        tune_scores, holdout_scores, median = fill_missing_scores(
            scores[tune_indices], scores[holdout_indices]
        )
        selection = select_threshold(tune_scores, label_array[tune_indices])
        holdout = evaluate_scores(
            holdout_scores, label_array[holdout_indices], selection.threshold
        )
        records.append(
            CandidateRecord(
                name=name,
                threshold=selection.threshold,
                tune_macro_f1=selection.macro_f1,
                tune_accuracy=selection.accuracy,
                holdout_macro_f1=holdout.macro_f1,
                holdout_accuracy=holdout.accuracy,
                holdout_roc_auc=holdout.roc_auc,
                imputation_median=median,
            )
        )
    order = {name: index for index, name in enumerate(simplicity_order)}
    winner = max(
        records,
        key=lambda record: (
            record.holdout_macro_f1,
            record.holdout_accuracy,
            record.holdout_roc_auc,
            -order[record.name],
        ),
    )
    return winner.name, records


def _static_scores(
    examples: Sequence[WicExample],
    store: EmbeddingStore,
    windows: Iterable[ContextWindow],
) -> tuple[dict[str, NDArray[np.float64]], dict[str, dict[str, int]]]:
    score_sets: dict[str, NDArray[np.float64]] = {}
    diagnostics: dict[str, dict[str, int]] = {}

    target_encoder = StaticTargetEncoder(store)
    target_scores = np.full(len(examples), np.nan, dtype=np.float64)
    missing_targets = 0
    for row, example in enumerate(examples):
        first = target_encoder.encode(example, 1)
        second = target_encoder.encode(example, 2)
        if first is None or second is None:
            missing_targets += 1
        else:
            # Both sentences use the same lemma vector. Write 1.0 directly so
            # a tiny rounding difference does not look like a real difference.
            target_scores[row] = 1.0
    score_sets["glove_target"] = target_scores
    diagnostics["glove_target"] = {
        "missing_scores": missing_targets,
        "target_fallback_sides": 0,
        "oov_context_tokens": 0,
    }

    for window in windows:
        name = f"glove_context_{window}"
        encoder = StaticContextEncoder(store, window=window)
        scores = np.full(len(examples), np.nan, dtype=np.float64)
        fallback_sides = 0
        oov_tokens = 0
        for row, example in enumerate(examples):
            first = encoder.encode(example, 1)
            second = encoder.encode(example, 2)
            fallback_sides += int(first.used_target_fallback) + int(second.used_target_fallback)
            oov_tokens += first.oov_count + second.oov_count
            if first.vector is not None and second.vector is not None:
                scores[row] = pairwise_cosine_scores(
                    np.stack((first.vector, second.vector), axis=0)[None, :, :]
                )[0]
        score_sets[name] = scores
        diagnostics[name] = {
            "missing_scores": int(np.isnan(scores).sum()),
            "target_fallback_sides": fallback_sides,
            "oov_context_tokens": oov_tokens,
        }
    return score_sets, diagnostics


def _bert_cache_metadata(
    config: ExperimentConfig, split: str, examples: list[WicExample]
) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "kind": "bert_target_all_layers_mean_wordpieces",
        "split": split,
        "examples": len(examples),
        "fingerprint": fingerprint_examples(examples),
        "data_sha256": config.data.sha256,
        "model": config.bert.model,
        "revision": config.bert.revision,
        "max_length": config.bert.max_length,
        "seed": config.seed,
    }


def _glove_store(
    config: ExperimentConfig, path: Path, vocabulary: set[str]
) -> tuple[EmbeddingStore, bool]:
    cache_path = config.output.cache_dir / "glove_subset.npz"
    metadata = {
        "schema": CACHE_SCHEMA_VERSION,
        "kind": "glove_vocabulary_subset",
        "archive_sha256": config.glove.sha256,
        "member": config.glove.member,
        "dimensions": config.glove.dimensions,
        "vocabulary_hash": canonical_hash(sorted(vocabulary)),
        "requested_tokens": len(vocabulary),
    }
    cached = load_array_cache(cache_path, metadata)
    if cached is not None and set(cached) == {"tokens", "vectors"}:
        tokens = np.asarray(cached["tokens"])
        vectors = np.asarray(cached["vectors"], dtype=np.float32)
        if tokens.ndim == 1 and vectors.shape == (len(tokens), config.glove.dimensions):
            return (
                EmbeddingStore(
                    {str(token): vector for token, vector in zip(tokens, vectors, strict=True)},
                    config.glove.dimensions,
                ),
                True,
            )
    store = EmbeddingStore.from_text(
        path,
        dimensions=config.glove.dimensions,
        vocabulary=vocabulary,
    )
    tokens, vectors = store.to_arrays()
    save_array_cache(cache_path, {"tokens": tokens, "vectors": vectors}, metadata)
    return store, False


def _bert_representations(
    config: ExperimentConfig,
    split: str,
    examples: list[WicExample],
    *,
    encoder_factory: BertEncoderFactory | None,
) -> tuple[BertRepresentations, bool]:
    cache_path = config.output.cache_dir / f"bert_{split}.npz"
    metadata = _bert_cache_metadata(config, split, examples)
    cached = load_array_cache(cache_path, metadata)
    if cached is not None:
        counts = np.asarray(cached.pop("wordpiece_counts"), dtype=np.int16)
        return (
            BertRepresentations(
                vectors={
                    name: np.asarray(value, dtype=np.float32)
                    for name, value in cached.items()
                },
                wordpiece_counts=counts,
                device="cache",
                model=config.bert.model,
                revision=config.bert.revision,
            ),
            True,
        )
    encoder = (
        encoder_factory(config)
        if encoder_factory is not None
        else BertTargetEncoder(config.bert, device=config.runtime.device)
    )
    representations = encoder.encode(examples)
    arrays = dict(representations.vectors)
    arrays["wordpiece_counts"] = representations.wordpiece_counts
    save_array_cache(cache_path, arrays, metadata)
    return representations, False


def _config_record(config: ExperimentConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return {key: convert(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return convert(config)


def _metric_row(system: str, threshold: float, metrics: MetricSet) -> dict[str, Any]:
    return {
        "system": system,
        "threshold": threshold,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "roc_auc": metrics.roc_auc,
        "tn": metrics.confusion_matrix[0][0],
        "fp": metrics.confusion_matrix[0][1],
        "fn": metrics.confusion_matrix[1][0],
        "tp": metrics.confusion_matrix[1][1],
        "count": metrics.count,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def run_experiment(
    config: ExperimentConfig | str | Path,
    *,
    bert_encoder_factory: BertEncoderFactory | None = None,
) -> ExperimentArtifacts:
    """Run the prespecified experiment and write traceable raw/final artifacts."""

    if not isinstance(config, ExperimentConfig):
        config = load_config(config)
    set_global_seed(config.seed, deterministic=config.runtime.deterministic)
    run_environment = environment_info(config.project_root)
    for directory in (
        config.output.cache_dir,
        config.output.raw_results_dir,
        config.output.final_results_dir,
        config.output.examples_dir,
        config.output.figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    ensure_wic_data(
        url=config.data.url,
        expected_sha256=config.data.sha256,
        raw_dir=config.data.raw_dir,
        processed_dir=config.data.processed_dir,
    )
    full_train = load_wic_split(config.data.processed_dir, config.data.train_split)
    full_validation = load_wic_split(
        config.data.processed_dir, config.data.validation_split
    )
    test = load_wic_split(config.data.processed_dir, config.data.test_split)
    audits = [
        audit_examples(full_train, "train"),
        audit_examples(full_validation, "val"),
        audit_examples(test, "test"),
    ]
    validate_expected_release(audits)
    validate_no_cross_split_pairs(full_train, full_validation)

    train = stratified_limit(full_train, config.limits.train, seed=config.seed)
    validation = stratified_limit(
        full_validation, config.limits.validation, seed=config.seed + 1
    )
    train_labels = _labels(train)
    validation_labels = _labels(validation)
    tune_indices, holdout_indices = make_tuning_split(
        train_labels,
        holdout_fraction=config.evaluation.tune_holdout_fraction,
        seed=config.seed,
    )

    glove_path = ensure_glove_file(config.glove, config.data.raw_dir)
    vocabulary = collect_vocabulary([*train, *validation])
    store, glove_cache_hit = _glove_store(config, glove_path, vocabulary)
    train_static, train_static_diagnostics = _static_scores(
        train, store, config.glove.context_windows
    )
    context_names = [f"glove_context_{window}" for window in config.glove.context_windows]
    context_winner, context_records = choose_candidate(
        {name: train_static[name] for name in context_names},
        train_labels,
        tune_indices=tune_indices,
        holdout_indices=holdout_indices,
        simplicity_order=context_names,
    )

    shared_encoder: BertTargetEncoder | None = None

    def encoder_for_run(current_config: ExperimentConfig) -> BertTargetEncoder:
        nonlocal shared_encoder
        if shared_encoder is None:
            shared_encoder = (
                bert_encoder_factory(current_config)
                if bert_encoder_factory is not None
                else BertTargetEncoder(
                    current_config.bert, device=current_config.runtime.device
                )
            )
        return shared_encoder

    train_bert, train_cache_hit = _bert_representations(
        config,
        "train",
        train,
        encoder_factory=encoder_for_run,
    )
    train_bert_scores = {
        name: pairwise_cosine_scores(train_bert.vectors[name])
        for name in train_bert.vectors
    }
    bert_winner, bert_records = choose_candidate(
        {name: train_bert_scores[name] for name in config.bert.primary_candidates},
        train_labels,
        tune_indices=tune_indices,
        holdout_indices=holdout_indices,
        simplicity_order=config.bert.primary_candidates,
    )

    ledger_core = {
        "protocol": "train-only candidate selection; validation labels unopened for selection",
        "seed": config.seed,
        "train_examples": len(train),
        "tune_examples": len(tune_indices),
        "holdout_examples": len(holdout_indices),
        "tune_indices_hash": canonical_hash(tune_indices.tolist()),
        "holdout_indices_hash": canonical_hash(holdout_indices.tolist()),
        "static_context_candidates": [asdict(record) for record in context_records],
        "selected_static_context": context_winner,
        "bert_candidates": [asdict(record) for record in bert_records],
        "selected_bert_representation": bert_winner,
        "tie_break": "macro_f1, accuracy, roc_auc, declared simplicity order",
    }
    ledger = {**ledger_core, "selection_hash": canonical_hash(ledger_core)}
    selection_path = config.output.raw_results_dir / "selection_ledger.json"
    atomic_write_json(selection_path, ledger)

    # We read validation text earlier for the data checks and vocabulary. Its
    # labels are only used for scores after the choices above are saved.
    validation_static, validation_static_diagnostics = _static_scores(
        validation, store, config.glove.context_windows
    )
    validation_bert, validation_cache_hit = _bert_representations(
        config,
        "validation",
        validation,
        encoder_factory=encoder_for_run,
    )
    validation_bert_scores = {
        name: pairwise_cosine_scores(validation_bert.vectors[name])
        for name in validation_bert.vectors
    }

    primary_raw = {
        "glove_target": (
            train_static["glove_target"],
            validation_static["glove_target"],
        ),
        context_winner: (
            train_static[context_winner],
            validation_static[context_winner],
        ),
        f"bert_{bert_winner}": (
            train_bert_scores[bert_winner],
            validation_bert_scores[bert_winner],
        ),
    }
    primary_scores: dict[str, NDArray[np.float64]] = {}
    thresholds: dict[str, float] = {}
    imputation_medians: dict[str, float] = {}
    metrics_rows: list[dict[str, Any]] = []
    confusion: dict[str, Any] = {}
    for name, (raw_train, raw_validation) in primary_raw.items():
        filled_train, filled_validation, median = fill_missing_scores(
            raw_train, raw_validation
        )
        threshold = select_threshold(filled_train, train_labels).threshold
        metrics = evaluate_scores(filled_validation, validation_labels, threshold)
        primary_scores[name] = filled_validation
        thresholds[name] = threshold
        imputation_medians[name] = median
        metrics_rows.append(_metric_row(name, threshold, metrics))
        confusion[name] = metrics.confusion_matrix

    layer_rows: list[dict[str, Any]] = []
    layer_thresholds: dict[str, float] = {}
    for layer in range(1, 13):
        layer_name = f"layer_{layer}"
        threshold = select_threshold(train_bert_scores[layer_name], train_labels).threshold
        metrics = evaluate_scores(
            validation_bert_scores[layer_name], validation_labels, threshold
        )
        layer_thresholds[layer_name] = threshold
        row = _metric_row(layer_name, threshold, metrics)
        row["selected_primary"] = layer_name == bert_winner
        layer_rows.append(row)

    raw_rows: list[dict[str, Any]] = []
    for split, examples, labels, static, bert_scores in (
        ("train", train, train_labels, train_static, train_bert_scores),
        (
            "validation",
            validation,
            validation_labels,
            validation_static,
            validation_bert_scores,
        ),
    ):
        all_scores = {**static, **{f"bert_{key}": value for key, value in bert_scores.items()}}
        for row, example in enumerate(examples):
            record: dict[str, Any] = {
                "split": split,
                "row": row,
                "idx": example.idx,
                "word": example.word,
                "label": bool(labels[row]),
            }
            record.update(
                {
                    name: float(values[row]) if np.isfinite(values[row]) else np.nan
                    for name, values in all_scores.items()
                }
            )
            raw_rows.append(record)

    prediction_rows: list[dict[str, Any]] = []
    for row, example in enumerate(validation):
        record = {
            "row": row,
            "idx": example.idx,
            "word": example.word,
            "sentence1": example.sentence1,
            "sentence2": example.sentence2,
            "target1": example.target_surface(1),
            "target2": example.target_surface(2),
            "label": bool(validation_labels[row]),
        }
        for name, scores in primary_scores.items():
            record[f"{name}_score"] = float(scores[row])
            record[f"{name}_prediction"] = bool(scores[row] >= thresholds[name])
            record[f"{name}_correct"] = bool(
                (scores[row] >= thresholds[name]) == validation_labels[row]
            )
        prediction_rows.append(record)

    bootstrap_rows = [
        asdict(interval)
        for interval in bootstrap_intervals(
            primary_scores,
            validation_labels,
            thresholds,
            resamples=config.evaluation.bootstrap_resamples,
            confidence_level=config.evaluation.confidence_level,
            seed=config.seed,
        )
    ]
    paired_rows: list[dict[str, Any]] = []
    for first, second in combinations(primary_scores, 2):
        paired_rows.extend(
            asdict(difference)
            for difference in paired_differences(
                scores_a=primary_scores[first],
                threshold_a=thresholds[first],
                name_a=first,
                scores_b=primary_scores[second],
                threshold_b=thresholds[second],
                name_b=second,
                labels=validation_labels,
                resamples=config.evaluation.bootstrap_resamples,
                confidence_level=config.evaluation.confidence_level,
                seed=config.seed,
            )
        )

    raw_scores_path = config.output.raw_results_dir / "scores.csv"
    metrics_path = config.output.final_results_dir / "metrics.csv"
    bootstrap_path = config.output.final_results_dir / "bootstrap_ci.csv"
    paired_path = config.output.final_results_dir / "paired_differences.csv"
    predictions_path = config.output.final_results_dir / "predictions.csv"
    layer_path = config.output.final_results_dir / "layer_metrics.csv"
    config_path = config.output.final_results_dir / "config.json"
    environment_path = config.output.final_results_dir / "environment.json"
    _write_csv(raw_scores_path, raw_rows)
    _write_csv(metrics_path, metrics_rows)
    _write_csv(bootstrap_path, bootstrap_rows)
    _write_csv(paired_path, paired_rows)
    _write_csv(predictions_path, prediction_rows)
    _write_csv(layer_path, layer_rows)
    atomic_write_json(config_path, _config_record(config))
    atomic_write_json(
        config.output.final_results_dir / "confusion_matrices.json", confusion
    )
    atomic_write_json(
        config.output.final_results_dir / "run_diagnostics.json",
        {
            "audits": [asdict(audit) for audit in audits],
            "glove_loaded_vocabulary": len(store),
            "glove_requested_vocabulary": len(vocabulary),
            "glove_cache_hit": glove_cache_hit,
            "static_train": train_static_diagnostics,
            "static_validation": validation_static_diagnostics,
            "bert_train_cache_hit": train_cache_hit,
            "bert_validation_cache_hit": validation_cache_hit,
            "bert_train_device": train_bert.device,
            "bert_validation_device": validation_bert.device,
            "train_wordpiece_counts": {
                "min": int(train_bert.wordpiece_counts.min()),
                "max": int(train_bert.wordpiece_counts.max()),
                "mean": float(train_bert.wordpiece_counts.mean()),
            },
            "validation_wordpiece_counts": {
                "min": int(validation_bert.wordpiece_counts.min()),
                "max": int(validation_bert.wordpiece_counts.max()),
                "mean": float(validation_bert.wordpiece_counts.mean()),
            },
            "imputation_medians": imputation_medians,
            "layer_thresholds": layer_thresholds,
        },
    )
    atomic_write_json(environment_path, run_environment)

    return ExperimentArtifacts(
        selection_ledger=selection_path,
        metrics=metrics_path,
        bootstrap_ci=bootstrap_path,
        paired_differences=paired_path,
        predictions=predictions_path,
        layer_metrics=layer_path,
        raw_scores=raw_scores_path,
        config=config_path,
        environment=environment_path,
    )
