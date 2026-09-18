"""Dataset-agnostic quantile symbols for numeric recommendation evidence.

The miner needs a small, stable vocabulary rather than raw floating-point
values.  :class:`QuantileNumericEvidence` learns that vocabulary from one
training window and is immutable afterwards.  Evaluation/serving values are
only transformed, so they cannot move the learned boundaries.

Scalar values become ``q1`` ... ``qN``.  Pairwise evidence uses the quantile
of the *absolute* training delta while retaining direction, for example
``left_q2`` or ``right_q2``.  This makes swapping a candidate pair exactly
antisymmetric.  Missing values are represented explicitly as ``unknown``,
``left_known`` or ``right_known``; no dataset-specific imputation is used.

Repeated training values can make an equal-frequency split impossible.  We
therefore choose the nearest boundary between distinct observations and
collapse duplicate boundaries.  The resulting effective number of bins can
be smaller than ``requested_bins``, but no learned bin is empty merely because
a quantile cut landed inside a tie block.
"""

from __future__ import annotations

import bisect
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_SCHEMA = "quantile_numeric_evidence"
_VERSION = 1
_QUANTILE_METHOD = "nearest_distinct_empirical_split_v1"


def _finite_number(value: object) -> float | None:
    """Return a finite float, treating malformed values as missing evidence."""

    # Booleans are usually categorical flags; silently treating True as 1.0
    # would create an accidental numeric predicate.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    # Avoid serializing negative zero as a distinct-looking threshold.
    return 0.0 if number == 0.0 else number


def _absolute_delta(left: float, right: float) -> float:
    """Compute a finite absolute delta even for opposite extreme floats."""

    delta = abs(left - right)
    return delta if math.isfinite(delta) else sys.float_info.max


def _training_scale(values: Sequence[float]) -> float:
    """Return a finite feature scale used only for tolerant equality checks."""

    if len(values) < 2:
        return 0.0
    width = max(values) - min(values)
    return width if math.isfinite(width) else sys.float_info.max


def _scale_relative_tolerance(fraction: float, scale: float) -> float:
    """Multiply without emitting infinity into portable JSON metadata."""

    if fraction == 0.0 or scale == 0.0:
        return 0.0
    if fraction > sys.float_info.max / scale:
        return sys.float_info.max
    return fraction * scale


def _distinct_quantile_thresholds(
    values: Sequence[float], requested_bins: int, *,
    weights: Sequence[float] | None = None,
    relative_tolerance: float = 0.0,
    absolute_tolerance: float = 0.0,
) -> tuple[float, ...]:
    """Choose deterministic empirical cuts that never split equivalent values.

    Every eligible split is immediately before a new distinct value.  For
    each requested quantile we select the eligible split nearest its ideal
    cumulative mass, preferring the lower split on an exact tie.  Optional
    weights let every source group (for example an impression) contribute the
    same total mass.  Tolerant equivalence prevents representation noise such
    as ``0.04`` versus ``0.04000000000000001`` from inventing a boundary.
    Selecting the same split more than once reduces the effective bin count.
    """

    if weights is None:
        weights = (1.0,) * len(values)
    if len(weights) != len(values):
        raise ValueError("quantile weights length must equal values length")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    if len(ordered) < 2 or requested_bins <= 1:
        return ()
    eligible = [
        index
        for index in range(1, len(ordered))
        if not math.isclose(
            ordered[index - 1][0], ordered[index][0],
            rel_tol=relative_tolerance, abs_tol=absolute_tolerance,
        )
    ]
    if not eligible:
        return ()

    cumulative = []
    running = 0.0
    for _value, weight in ordered:
        running += weight
        cumulative.append(running)
    total_weight = cumulative[-1]
    if total_weight <= 0.0:
        return ()
    selected: set[int] = set()
    for quantile in range(1, requested_bins):
        target = quantile * total_weight / requested_bins
        selected.add(min(
            eligible,
            key=lambda index: (abs(cumulative[index - 1] - target), index),
        ))
    return tuple(ordered[index][0] for index in sorted(selected))


def _validated_thresholds(raw: object, field: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field} thresholds must be a list")
    values: list[float] = []
    for item in raw:
        value = _finite_number(item)
        if value is None:
            raise ValueError(f"{field} thresholds must all be finite numbers")
        values.append(value)
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{field} thresholds must be strictly increasing")
    return tuple(values)


@dataclass(frozen=True)
class QuantileNumericEvidence:
    """Frozen train-only encoder for one numeric predicate.

    Use one instance per semantic feature.  ``training_pairs`` should contain
    only pairs available in the fitting split.  Equal and incomplete pairs do
    not influence magnitude thresholds because they have dedicated labels.

    Quantiles are invariant under a common positive rescaling when the
    encoder is fitted and used in the same units.  ``scale_tolerance`` is a
    fraction of the observed training range, so near-tie handling scales with
    the feature rather than relying on a dataset-specific epsilon.
    """

    feature: str
    requested_bins: int
    scalar_thresholds: tuple[float, ...]
    delta_thresholds: tuple[float, ...]
    scalar_sample_count: int
    delta_sample_count: int
    training_scale: float
    relative_tolerance: float = 1e-12
    scale_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str) or not self.feature.strip():
            raise ValueError("feature must be a non-empty string")
        if isinstance(self.requested_bins, bool) or self.requested_bins < 1:
            raise ValueError("requested_bins must be at least 1")
        if self.scalar_sample_count < 0 or self.delta_sample_count < 0:
            raise ValueError("sample counts cannot be negative")
        for name, value in (
            ("training_scale", self.training_scale),
            ("relative_tolerance", self.relative_tolerance),
            ("scale_tolerance", self.scale_tolerance),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if any(
            not math.isfinite(value)
            for value in (*self.scalar_thresholds, *self.delta_thresholds)
        ):
            raise ValueError("quantile thresholds must be finite")
        if any(
            left >= right
            for thresholds in (self.scalar_thresholds, self.delta_thresholds)
            for left, right in zip(thresholds, thresholds[1:])
        ):
            raise ValueError("quantile thresholds must be strictly increasing")

    @classmethod
    def fit(
        cls,
        feature: str,
        training_values: Iterable[object],
        training_pairs: Iterable[Sequence[object]] = (),
        *,
        training_pair_weights: Iterable[object] | None = None,
        bins: int = 4,
        relative_tolerance: float = 1e-12,
        scale_tolerance: float = 1e-12,
    ) -> "QuantileNumericEvidence":
        """Fit deterministic thresholds from a training split.

        ``training_values`` determines scalar bins.  Non-equal finite
        ``training_pairs`` determine absolute-delta magnitude bins. Optional
        ``training_pair_weights`` are aligned one-for-one with those pairs and
        affect only the empirical delta quantiles. Pair endpoints also
        contribute to the scale used for equality tolerance, but never to
        scalar quantiles unless supplied in ``training_values``.
        """

        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be an integer of at least 1")
        for name, value in (
            ("relative_tolerance", relative_tolerance),
            ("scale_tolerance", scale_tolerance),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")

        scalar_values = [
            number
            for raw in training_values
            if (number := _finite_number(raw)) is not None
        ]
        raw_pairs = list(training_pairs)
        if training_pair_weights is None:
            raw_pair_weights = [1.0] * len(raw_pairs)
        else:
            raw_pair_weights = list(training_pair_weights)
            if len(raw_pair_weights) != len(raw_pairs):
                raise ValueError(
                    "training_pair_weights length must equal training_pairs length"
                )
        complete_pairs: list[tuple[float, float, float]] = []
        scale_values = list(scalar_values)
        for index, (pair, raw_weight) in enumerate(
            zip(raw_pairs, raw_pair_weights)
        ):
            if isinstance(pair, (str, bytes)):
                raise ValueError(f"training pair {index} must contain two values")
            try:
                left_raw, right_raw = pair
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"training pair {index} must contain exactly two values"
                ) from exc
            left = _finite_number(left_raw)
            right = _finite_number(right_raw)
            if left is None or right is None:
                continue
            weight = _finite_number(raw_weight)
            if weight is None or weight < 0.0:
                raise ValueError(
                    f"training pair weight {index} must be finite and non-negative"
                )
            if weight == 0.0:
                continue
            complete_pairs.append((left, right, weight))
            scale_values.extend((left, right))

        training_scale = _training_scale(scale_values)
        absolute_tolerance = _scale_relative_tolerance(
            float(scale_tolerance), training_scale
        )
        delta_rows = [
            (_absolute_delta(left, right), weight)
            for left, right, weight in complete_pairs
            if not math.isclose(
                left, right, rel_tol=float(relative_tolerance),
                abs_tol=absolute_tolerance,
            )
        ]
        deltas = [value for value, _weight in delta_rows]
        delta_weights = [weight for _value, weight in delta_rows]

        return cls(
            feature=feature.strip(),
            requested_bins=bins,
            scalar_thresholds=_distinct_quantile_thresholds(
                scalar_values, bins,
                relative_tolerance=float(relative_tolerance),
                absolute_tolerance=absolute_tolerance,
            ),
            delta_thresholds=_distinct_quantile_thresholds(
                deltas, bins, weights=delta_weights,
                relative_tolerance=float(relative_tolerance),
                absolute_tolerance=absolute_tolerance,
            ),
            scalar_sample_count=len(scalar_values),
            delta_sample_count=len(deltas),
            training_scale=training_scale,
            relative_tolerance=float(relative_tolerance),
            scale_tolerance=float(scale_tolerance),
        )

    @property
    def scalar_bin_count(self) -> int:
        return len(self.scalar_thresholds) + 1

    @property
    def delta_bin_count(self) -> int:
        return len(self.delta_thresholds) + 1

    @property
    def absolute_tolerance(self) -> float:
        return _scale_relative_tolerance(
            self.scale_tolerance, self.training_scale
        )

    def encode_scalar(self, value: object) -> str:
        """Encode a finite scalar as ``qN`` or return ``unknown``."""

        number = _finite_number(value)
        if number is None:
            return "unknown"
        bucket = bisect.bisect_right(self.scalar_thresholds, number) + 1
        return f"q{bucket}"

    def encode_pair(self, left: object, right: object) -> str:
        """Encode directional magnitude with exact swap antisymmetry."""

        left_number = _finite_number(left)
        right_number = _finite_number(right)
        if left_number is None and right_number is None:
            return "unknown"
        if left_number is None:
            return "right_known"
        if right_number is None:
            return "left_known"
        if math.isclose(
            left_number,
            right_number,
            rel_tol=self.relative_tolerance,
            abs_tol=self.absolute_tolerance,
        ):
            return "equal"

        magnitude = _absolute_delta(left_number, right_number)
        bucket = bisect.bisect_right(self.delta_thresholds, magnitude) + 1
        direction = "left" if left_number > right_number else "right"
        return f"{direction}_q{bucket}"

    def to_metadata(self) -> dict[str, Any]:
        """Return complete JSON-safe metadata for reproducible serving."""

        return {
            "schema": _SCHEMA,
            "version": _VERSION,
            "feature": self.feature,
            "requested_bins": self.requested_bins,
            "quantile_method": _QUANTILE_METHOD,
            "scalar": {
                "thresholds": list(self.scalar_thresholds),
                "sample_count": self.scalar_sample_count,
                "bin_count": self.scalar_bin_count,
            },
            "pair_delta": {
                "thresholds": list(self.delta_thresholds),
                "sample_count": self.delta_sample_count,
                "bin_count": self.delta_bin_count,
            },
            "equality": {
                "training_scale": self.training_scale,
                "relative_tolerance": self.relative_tolerance,
                "scale_tolerance": self.scale_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
            },
            "missing_labels": {
                "both": "unknown",
                "left_only": "left_known",
                "right_only": "right_known",
            },
        }

    def to_json(self) -> str:
        """Serialize metadata deterministically."""

        return json.dumps(
            self.to_metadata(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "QuantileNumericEvidence":
        """Restore and validate a frozen encoder from serialized metadata."""

        if metadata.get("schema") != _SCHEMA or metadata.get("version") != _VERSION:
            raise ValueError("unsupported numeric evidence schema")
        if metadata.get("quantile_method") != _QUANTILE_METHOD:
            raise ValueError("unsupported quantile method")
        scalar = metadata.get("scalar")
        pair_delta = metadata.get("pair_delta")
        equality = metadata.get("equality")
        if not isinstance(scalar, Mapping) or not isinstance(pair_delta, Mapping):
            raise ValueError("metadata is missing quantile sections")
        if not isinstance(equality, Mapping):
            raise ValueError("metadata is missing equality settings")

        try:
            requested_bins = int(metadata["requested_bins"])
            scalar_count = int(scalar["sample_count"])
            delta_count = int(pair_delta["sample_count"])
            training_scale = float(equality["training_scale"])
            relative_tolerance = float(equality["relative_tolerance"])
            scale_tolerance = float(equality["scale_tolerance"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid numeric evidence metadata") from exc

        encoder = cls(
            feature=str(metadata.get("feature", "")),
            requested_bins=requested_bins,
            scalar_thresholds=_validated_thresholds(
                scalar.get("thresholds"), "scalar"
            ),
            delta_thresholds=_validated_thresholds(
                pair_delta.get("thresholds"), "pair_delta"
            ),
            scalar_sample_count=scalar_count,
            delta_sample_count=delta_count,
            training_scale=training_scale,
            relative_tolerance=relative_tolerance,
            scale_tolerance=scale_tolerance,
        )
        if scalar.get("bin_count") != encoder.scalar_bin_count:
            raise ValueError("scalar bin count does not match thresholds")
        if pair_delta.get("bin_count") != encoder.delta_bin_count:
            raise ValueError("pair-delta bin count does not match thresholds")
        return encoder

    @classmethod
    def from_json(cls, payload: str) -> "QuantileNumericEvidence":
        """Restore an encoder from :meth:`to_json` output."""

        try:
            metadata = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid numeric evidence JSON") from exc
        if not isinstance(metadata, Mapping):
            raise ValueError("numeric evidence JSON must contain an object")
        return cls.from_metadata(metadata)


__all__ = ["QuantileNumericEvidence"]
