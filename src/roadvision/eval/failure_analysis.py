"""Scenario-sliced failure-mode analysis.

Real nuScenes scenario attributes (time of day, weather, scene description
tags) require the actual downloaded dataset/devkit to extract, so the
slicing utilities here are generic: they operate on a per-sample attribute
dict the caller supplies, however it was produced. `nuscenes_scene_attributes`
is the thin adapter that would populate that dict from real data -- it is
not exercised by this environment's test suite (no dataset to run it
against), same honesty bar as `data/bev_rasterize.py`'s devkit-dependent
classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SliceStats:
    attribute: str
    value: Any
    count: int
    mean: float
    worst_tokens: list[str] = field(default_factory=list)


def slice_metrics_by_attribute(
    sample_tokens: list[str],
    metric_values: list[float],
    attributes: list[dict[str, Any]],
    attribute_key: str,
    top_k_worst: int = 5,
) -> list[SliceStats]:
    """Group `metric_values` (higher = worse, e.g. a per-sample loss or
    AbsRel) by `attributes[i][attribute_key]`, returning one `SliceStats`
    per distinct value, sorted worst-mean-first.

    All three list arguments must be the same length and share index
    correspondence (sample i's token/metric/attributes).
    """
    if not (len(sample_tokens) == len(metric_values) == len(attributes)):
        raise ValueError("sample_tokens, metric_values, and attributes must have matching lengths")

    groups: dict[Any, list[tuple[str, float]]] = {}
    for token, value, attrs in zip(sample_tokens, metric_values, attributes):
        key = attrs.get(attribute_key, "unknown")
        groups.setdefault(key, []).append((token, value))

    stats = []
    for value, entries in groups.items():
        entries_sorted = sorted(entries, key=lambda e: e[1], reverse=True)
        mean = sum(v for _, v in entries) / len(entries)
        stats.append(
            SliceStats(
                attribute=attribute_key,
                value=value,
                count=len(entries),
                mean=mean,
                worst_tokens=[token for token, _ in entries_sorted[:top_k_worst]],
            )
        )
    return sorted(stats, key=lambda s: s.mean, reverse=True)


def worst_k_samples(sample_tokens: list[str], metric_values: list[float], k: int) -> list[tuple[str, float]]:
    """Top-`k` (token, value) pairs by descending metric value (higher =
    worse), for pulling up worst-case samples to inspect visually."""
    if len(sample_tokens) != len(metric_values):
        raise ValueError("sample_tokens and metric_values must have matching lengths")
    paired = sorted(zip(sample_tokens, metric_values), key=lambda p: p[1], reverse=True)
    return paired[:k]


def nuscenes_scene_attributes(nusc: Any, sample_token: str) -> dict[str, Any]:
    """Adapter from a real sample token to scenario attributes, for use with
    `slice_metrics_by_attribute`. Requires a live `nuscenes.NuScenes`
    instance and the actual dataset on disk -- not exercised by this
    repo's test suite, which has no real nuScenes data available.
    """
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    description = scene["description"].lower()

    time_of_day = "night" if "night" in description else "day"
    weather = "rain" if "rain" in description else ("wet" if "wet" in description else "clear")
    return {"time_of_day": time_of_day, "weather": weather, "scene_description": scene["description"]}
