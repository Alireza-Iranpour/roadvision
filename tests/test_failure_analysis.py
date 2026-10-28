import math

from roadvision.eval.failure_analysis import slice_metrics_by_attribute, worst_k_samples


def test_slice_metrics_by_attribute_groups_and_ranks():
    tokens = ["a", "b", "c", "d"]
    values = [0.1, 0.5, 0.3, 0.9]
    attributes = [
        {"time_of_day": "day"},
        {"time_of_day": "night"},
        {"time_of_day": "day"},
        {"time_of_day": "night"},
    ]

    stats = slice_metrics_by_attribute(tokens, values, attributes, "time_of_day", top_k_worst=1)

    assert [s.value for s in stats] == ["night", "day"]  # worst mean first
    night_stats = stats[0]
    assert night_stats.count == 2
    assert math.isclose(night_stats.mean, 0.7)
    assert night_stats.worst_tokens == ["d"]  # d (0.9) worse than b (0.5)

    day_stats = stats[1]
    assert math.isclose(day_stats.mean, 0.2)
    assert day_stats.worst_tokens == ["c"]


def test_slice_metrics_by_attribute_missing_key_defaults_to_unknown():
    stats = slice_metrics_by_attribute(["a"], [1.0], [{}], "weather")
    assert stats[0].value == "unknown"


def test_slice_metrics_by_attribute_rejects_mismatched_lengths():
    try:
        slice_metrics_by_attribute(["a", "b"], [1.0], [{}], "weather")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched lengths")


def test_worst_k_samples_sorted_descending():
    tokens = ["a", "b", "c"]
    values = [0.2, 0.9, 0.5]

    worst_two = worst_k_samples(tokens, values, k=2)

    assert worst_two == [("b", 0.9), ("c", 0.5)]


def test_worst_k_samples_k_larger_than_data_returns_all():
    result = worst_k_samples(["a"], [0.1], k=5)
    assert result == [("a", 0.1)]
