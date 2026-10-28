import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from visualize_bev import plot_bev_predictions  # noqa: E402


def test_plot_bev_predictions_saves_file(tmp_path):
    seg_logits = torch.randn(2, 20, 20)
    det_heatmap = torch.randn(3, 20, 20)
    det_offset = torch.rand(2, 20, 20)

    out_path = tmp_path / "bev.png"
    result = plot_bev_predictions(seg_logits, det_heatmap, det_offset, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_bev_predictions_handles_no_detections_above_threshold():
    seg_logits = torch.full((2, 10, 10), -10.0)
    det_heatmap = torch.full((1, 10, 10), -10.0)
    det_offset = torch.zeros(2, 10, 10)

    out_path = plot_bev_predictions(seg_logits, det_heatmap, det_offset, "/tmp/roadvision_test_bev_empty.png")

    assert out_path.exists()
