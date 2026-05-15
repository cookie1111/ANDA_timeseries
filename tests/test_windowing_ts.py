import pytest
import torch

from ANDA import windowing_ts


def _seq_batch(N=3, C=1, T=12):
    base = torch.arange(T, dtype=torch.float32)
    return base.repeat(N, C, 1) + torch.arange(N).reshape(N, 1, 1) * 100


class TestWindowSeries:
    def test_non_overlapping_default_stride(self):
        x = _seq_batch(N=2, C=1, T=12)
        y = torch.tensor([10, 20])
        w_x, w_y = windowing_ts.window_series(x, y, window_size=4)
        # T=12, ws=4, stride=4 -> 3 windows per series, 2 series -> 6 windows total
        assert w_x.shape == (6, 1, 4)
        assert w_y.shape == (6,)
        # labels repeat per parent series
        assert torch.equal(w_y, torch.tensor([10, 10, 10, 20, 20, 20]))

    def test_overlapping_stride(self):
        x = _seq_batch(N=1, C=1, T=10)
        y = torch.tensor([7])
        w_x, w_y = windowing_ts.window_series(x, y, window_size=4, stride=2)
        # (10-4)//2 + 1 = 4 windows
        assert w_x.shape == (4, 1, 4)
        assert torch.equal(w_y, torch.tensor([7, 7, 7, 7]))
        # first window covers indices 0..3, last covers 6..9
        assert torch.equal(w_x[0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert torch.equal(w_x[-1, 0], torch.tensor([6.0, 7.0, 8.0, 9.0]))

    def test_window_equals_series_length_returns_clone(self):
        x = _seq_batch(N=2, C=1, T=8)
        y = torch.tensor([0, 1])
        w_x, w_y = windowing_ts.window_series(x, y, window_size=8)
        assert torch.equal(w_x, x)
        assert torch.equal(w_y, y)

    def test_window_larger_than_series_raises(self):
        x = _seq_batch(T=8)
        y = torch.tensor([0, 0, 0])
        with pytest.raises(ValueError, match="larger than series length"):
            windowing_ts.window_series(x, y, window_size=16)

    def test_invalid_window_size_raises(self):
        x = _seq_batch()
        y = torch.tensor([0, 0, 0])
        with pytest.raises(ValueError):
            windowing_ts.window_series(x, y, window_size=0)

    def test_invalid_stride_raises(self):
        x = _seq_batch()
        y = torch.tensor([0, 0, 0])
        with pytest.raises(ValueError):
            windowing_ts.window_series(x, y, window_size=4, stride=0)

    def test_rejects_non_3d(self):
        with pytest.raises(ValueError, match="\\(N, C, T\\)"):
            windowing_ts.window_series(torch.randn(4, 12), torch.tensor([0, 1, 2, 3]), window_size=4)

    def test_label_count_mismatch_raises(self):
        x = _seq_batch(N=3)
        with pytest.raises(ValueError):
            windowing_ts.window_series(x, torch.tensor([0, 1]), window_size=4)

    def test_drop_last_false_appends_trailing(self):
        x = _seq_batch(N=1, C=1, T=11)
        y = torch.tensor([42])
        # T=11, ws=4, stride=4 -> 2 full windows ([0:4], [4:8]); remainder=3 so one trailing aligned to end ([7:11])
        w_x, w_y = windowing_ts.window_series(x, y, window_size=4, stride=4, drop_last=False)
        assert w_x.shape == (3, 1, 4)
        assert torch.equal(w_x[-1, 0], torch.tensor([7.0, 8.0, 9.0, 10.0]))
        assert torch.equal(w_y, torch.tensor([42, 42, 42]))

    def test_multichannel(self):
        x = torch.randn(2, 3, 16)
        y = torch.tensor([0, 1])
        w_x, _ = windowing_ts.window_series(x, y, window_size=8, stride=8)
        assert w_x.shape == (4, 3, 8)
