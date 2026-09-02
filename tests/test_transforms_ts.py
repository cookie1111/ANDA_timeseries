import numpy as np
import pytest
import torch

from ANDA import transforms_ts


BUILTINS = [
    "gaussian_noise",
    "mean_offset",
    "amplitude_scaling",
    "jitter",
    "magnitude_warp",
    "time_warp",
    "window_slice",
    "permutation",
]


def _sample_batch(N=4, C=1, T=64):
    return torch.linspace(0, 1, T).repeat(N, C, 1) + torch.randn(N, C, T) * 0.01


class TestRegistry:
    def test_all_builtins_registered(self):
        names = transforms_ts.list_transforms()
        for name in BUILTINS:
            assert name in names

    def test_all_builtins_have_default_buckets(self):
        for name in BUILTINS:
            assert name in transforms_ts.TRANSFORM_BUCKETS
            buckets = transforms_ts.TRANSFORM_BUCKETS[name]
            assert isinstance(buckets, list) and len(buckets) >= 1

    def test_register_user_transform(self):
        @transforms_ts.register_transform("noop_test_only", buckets=[{}, {}])
        def noop(x):
            return x.clone()

        try:
            assert "noop_test_only" in transforms_ts.list_transforms()
            x = _sample_batch()
            assert torch.equal(transforms_ts.apply_transform(x, "noop_test_only"), x)
        finally:
            transforms_ts.TRANSFORMS.pop("noop_test_only", None)
            transforms_ts.TRANSFORM_BUCKETS.pop("noop_test_only", None)

    def test_apply_transform_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown transform"):
            transforms_ts.apply_transform(_sample_batch(), "definitely_not_registered")

class TestShapePreservation:
    @pytest.mark.parametrize("name", BUILTINS)
    def test_output_shape_matches_input(self, name):
        x = _sample_batch(N=3, C=2, T=64)
        y = transforms_ts.apply_transform(x, name)
        assert y.shape == x.shape

    @pytest.mark.parametrize("name", BUILTINS)
    def test_output_is_torch_tensor(self, name):
        x = _sample_batch()
        y = transforms_ts.apply_transform(x, name)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == x.dtype


class TestDeterminism:
    @pytest.mark.parametrize("name", BUILTINS)
    def test_same_seed_same_output(self, name):
        x = _sample_batch()
        np.random.seed(0)
        a = transforms_ts.apply_transform(x, name)
        np.random.seed(0)
        b = transforms_ts.apply_transform(x, name)
        assert torch.allclose(a, b)


class TestShapeValidation:
    def test_rejects_2d_tensor(self):
        x = torch.randn(10, 64)
        with pytest.raises(ValueError, match="\\(N, C, T\\)"):
            transforms_ts.apply_transform(x, "gaussian_noise")


class TestSemanticInvariants:
    def test_gaussian_noise_zero_sigma_is_identity(self):
        x = _sample_batch()
        np.random.seed(0)
        assert torch.allclose(transforms_ts.gaussian_noise(x, sigma=0.0), x)

    def test_amplitude_scaling_with_zero_std_is_identity(self):
        x = _sample_batch()
        np.random.seed(0)
        # scales = 1 + N(0,0) = 1, so output == input
        assert torch.allclose(transforms_ts.amplitude_scaling(x, scale_std=0.0), x)

    def test_mean_offset_with_zero_std_is_identity(self):
        x = _sample_batch()
        np.random.seed(0)
        assert torch.allclose(transforms_ts.mean_offset(x, offset_std=0.0), x)

    def test_window_slice_full_ratio_is_clone(self):
        x = _sample_batch()
        y = transforms_ts.window_slice(x, reduce_ratio=1.0)
        assert y.shape == x.shape

    def test_window_slice_invalid_ratio_raises(self):
        x = _sample_batch()
        with pytest.raises(ValueError):
            transforms_ts.window_slice(x, reduce_ratio=0.0)

    def test_permutation_rejects_low_max_segments(self):
        x = _sample_batch()
        with pytest.raises(ValueError):
            transforms_ts.permutation(x, max_segments=1)

    def test_gaussian_noise_actually_perturbs(self):
        x = _sample_batch()
        np.random.seed(1)
        y = transforms_ts.gaussian_noise(x, sigma=0.5)
        assert not torch.allclose(y, x)
