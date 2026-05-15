import numpy as np
import pytest
import torch

from ANDA import pipeline_ts


def _batch(N=4, C=1, T=32):
    return torch.arange(T, dtype=torch.float32).repeat(N, C, 1).clone(), torch.arange(N)


class TestPipelineStages:
    def test_no_op_returns_inputs(self):
        x, y = _batch()
        ox, oy = pipeline_ts.apply_ts_pipeline(x, y)
        assert torch.equal(ox, x)
        assert torch.equal(oy, y)

    def test_augment_then_window(self):
        x, y = _batch(N=2, C=1, T=16)
        np.random.seed(0)
        ox, oy = pipeline_ts.apply_ts_pipeline(
            x, y,
            transforms=[("gaussian_noise", {"sigma": 0.1})],
            window_size=4, stride=4,
            order=("augment", "window"),
        )
        # 2 series * (16/4) = 8 windows
        assert ox.shape == (8, 1, 4)
        assert oy.shape == (8,)
        assert not torch.allclose(ox[:, :, :], torch.arange(4, dtype=torch.float32).repeat(8, 1, 1))

    def test_window_then_augment(self):
        x, y = _batch(N=2, C=1, T=16)
        np.random.seed(0)
        ox, oy = pipeline_ts.apply_ts_pipeline(
            x, y,
            transforms=[("gaussian_noise", {"sigma": 0.1})],
            window_size=4, stride=4,
            order=("window", "augment"),
        )
        assert ox.shape == (8, 1, 4)
        assert oy.shape == (8,)

    def test_order_changes_result(self):
        x, y = _batch(N=2, C=1, T=16)
        # `permutation` iterates per-sample and shuffles segments inside the time
        # axis, so doing it before vs. after windowing produces materially
        # different outputs (whole-series shuffle vs. per-window shuffle).
        spec = [("permutation", {"max_segments": 4})]

        np.random.seed(0)
        a_x, _ = pipeline_ts.apply_ts_pipeline(
            x, y, transforms=spec, window_size=4, stride=4,
            order=("augment", "window"),
        )
        np.random.seed(0)
        b_x, _ = pipeline_ts.apply_ts_pipeline(
            x, y, transforms=spec, window_size=4, stride=4,
            order=("window", "augment"),
        )
        assert not torch.allclose(a_x, b_x)

    def test_transform_only(self):
        x, y = _batch(T=16)
        np.random.seed(1)
        ox, oy = pipeline_ts.apply_ts_pipeline(
            x, y, transforms=[("mean_offset", {"offset_std": 0.0})]
        )
        # zero-std mean_offset is identity
        assert torch.allclose(ox, x)
        assert torch.equal(oy, y)

    def test_window_only(self):
        x, y = _batch(N=2, C=1, T=8)
        ox, oy = pipeline_ts.apply_ts_pipeline(x, y, window_size=4, stride=4)
        assert ox.shape == (4, 1, 4)
        assert oy.shape == (4,)

    def test_unknown_stage_raises(self):
        x, y = _batch()
        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            pipeline_ts.apply_ts_pipeline(x, y, order=("augment", "bogus"))

    def test_string_entry_in_transforms_uses_defaults(self):
        x, y = _batch()
        np.random.seed(0)
        # bare string entry should be treated as (name, {}) and use the transform's defaults
        ox, _ = pipeline_ts.apply_ts_pipeline(x, y, transforms=["gaussian_noise"])
        assert ox.shape == x.shape
        assert not torch.allclose(ox, x)
