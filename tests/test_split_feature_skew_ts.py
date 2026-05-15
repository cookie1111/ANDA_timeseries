from collections import Counter

import numpy as np
import pytest
import torch

from ANDA import anda, split_fn_ts


def _synthetic_dataset(N_train=60, N_test=20, C=1, T=32, num_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    train_x = torch.from_numpy(rng.standard_normal((N_train, C, T))).float()
    test_x = torch.from_numpy(rng.standard_normal((N_test, C, T))).float()
    train_y = torch.from_numpy(rng.integers(0, num_classes, size=N_train)).long()
    test_y = torch.from_numpy(rng.integers(0, num_classes, size=N_test)).long()
    return train_x, train_y, test_x, test_y


class TestPoolBuilder:
    def test_pool_contains_all_buckets(self):
        pool = split_fn_ts._build_pool(["gaussian_noise"], magnitude_buckets=3, custom_buckets=None)
        assert len(pool) == 3
        for name, kwargs in pool:
            assert name == "gaussian_noise"
            assert "sigma" in kwargs

    def test_pool_respects_magnitude_buckets_limit(self):
        pool = split_fn_ts._build_pool(["gaussian_noise"], magnitude_buckets=2, custom_buckets=None)
        assert len(pool) == 2
        # subsampling picks endpoints first
        sigmas = sorted(kw["sigma"] for _, kw in pool)
        assert sigmas[0] == 0.01
        assert sigmas[-1] == 0.15

    def test_pool_combines_multiple_transforms(self):
        pool = split_fn_ts._build_pool(["gaussian_noise", "jitter"], magnitude_buckets=3, custom_buckets=None)
        names = [n for n, _ in pool]
        assert Counter(names) == Counter({"gaussian_noise": 3, "jitter": 3})

    def test_custom_buckets_override(self):
        pool = split_fn_ts._build_pool(
            ["gaussian_noise"],
            magnitude_buckets=None,
            custom_buckets={"gaussian_noise": [{"sigma": 9.99}]},
        )
        assert pool == [("gaussian_noise", {"sigma": 9.99})]

    def test_unknown_transform_raises(self):
        with pytest.raises(KeyError, match="Unknown transform"):
            split_fn_ts._build_pool(["not_a_transform"], magnitude_buckets=3, custom_buckets=None)

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError, match="at least one transform"):
            split_fn_ts._build_pool([], magnitude_buckets=3, custom_buckets=None)


class TestAssigningTransformFeatures:
    def test_returns_one_entry_per_datapoint(self):
        np.random.seed(0)
        pool = [("a", {"x": 1}), ("b", {"x": 2}), ("c", {"x": 3})]
        out = split_fn_ts.assigning_transform_features(50, pool, scaling=0.0)
        assert len(out) == 50
        for name, kw in out:
            assert (name, kw) in pool

    def test_scaling_zero_is_uniform(self):
        np.random.seed(0)
        pool = [(c, {"x": i}) for i, c in enumerate("abcd")]
        out = split_fn_ts.assigning_transform_features(8000, pool, scaling=0.0, random_order=False)
        counts = Counter(name for name, _ in out)
        # roughly uniform: each ~2000 / 8000 = 25%; allow generous slack
        for c in "abcd":
            assert 0.20 < counts[c] / 8000 < 0.30

    def test_scaling_high_concentrates(self):
        np.random.seed(0)
        pool = [(c, {"x": i}) for i, c in enumerate("abcd")]
        # With random_order=False the first pool entry has the highest softmax weight.
        out = split_fn_ts.assigning_transform_features(4000, pool, scaling=1.0, random_order=False)
        counts = Counter(name for name, _ in out)
        assert counts["a"] > counts["b"] > 0
        assert counts["a"] / 4000 > 0.4

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValueError):
            split_fn_ts.assigning_transform_features(10, [("a", {})], scaling=1.5)

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError):
            split_fn_ts.assigning_transform_features(10, [], scaling=0.0)


class TestSplitFeatureSkewTS:
    def test_basic_shape_and_count(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _synthetic_dataset(N_train=50, N_test=20, T=32)
        clients = split_fn_ts.split_feature_skew_ts(
            tr_x, tr_y, te_x, te_y,
            client_number=5,
            transforms_pool=["gaussian_noise", "amplitude_scaling"],
            magnitude_buckets=3,
        )
        assert len(clients) == 5
        total_train = sum(c["train_features"].shape[0] for c in clients)
        total_test = sum(c["test_features"].shape[0] for c in clients)
        assert total_train == 50
        assert total_test == 20
        for c in clients:
            assert c["train_features"].ndim == 3  # (n, C, T)
            assert c["test_features"].ndim == 3
            assert c["train_features"].shape[1] == 1
            assert c["train_features"].shape[2] == 32
            assert c["cluster"] == -1
            assert isinstance(c["train_features"], np.ndarray)
            assert isinstance(c["train_labels"], np.ndarray)

    def test_seed_reproducible(self):
        kwargs = dict(
            client_number=4,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            scaling_low=0.2,
            scaling_high=0.4,
        )

        anda.set_seed(123)
        tr_x, tr_y, te_x, te_y = _synthetic_dataset(seed=0)
        a = split_fn_ts.split_feature_skew_ts(tr_x, tr_y, te_x, te_y, **kwargs)

        anda.set_seed(123)
        tr_x2, tr_y2, te_x2, te_y2 = _synthetic_dataset(seed=0)
        b = split_fn_ts.split_feature_skew_ts(tr_x2, tr_y2, te_x2, te_y2, **kwargs)

        for ca, cb in zip(a, b):
            np.testing.assert_array_equal(ca["train_features"], cb["train_features"])
            np.testing.assert_array_equal(ca["train_labels"], cb["train_labels"])

    def test_with_windowing(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _synthetic_dataset(N_train=20, N_test=8, T=32)
        clients = split_fn_ts.split_feature_skew_ts(
            tr_x, tr_y, te_x, te_y,
            client_number=2,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            window_size=8, stride=8,
        )
        for c in clients:
            assert c["train_features"].shape[1:] == (1, 8)
            assert c["test_features"].shape[1:] == (1, 8)
            # (32/8) = 4 windows per parent series
            assert c["train_features"].shape[0] == c["train_labels"].shape[0]
            assert c["test_features"].shape[0] == c["test_labels"].shape[0]

    def test_transforms_actually_modify_data(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _synthetic_dataset(N_train=20, N_test=8, T=32)
        original = tr_x.clone()
        clients = split_fn_ts.split_feature_skew_ts(
            tr_x, tr_y, te_x, te_y,
            client_number=2,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            scaling_low=0.0, scaling_high=0.0,  # uniform; every sample gets some bucket
        )
        # Re-aggregate client features and compare to the original. They should differ
        # because every sample was transformed.
        all_train = np.concatenate([c["train_features"] for c in clients], axis=0)
        assert all_train.shape == original.shape
        # Some samples should differ from any row of the original.
        assert not np.allclose(np.sort(all_train.reshape(-1)), np.sort(original.numpy().reshape(-1)))

    def test_empty_pool_raises(self):
        tr_x, tr_y, te_x, te_y = _synthetic_dataset()
        with pytest.raises(ValueError):
            split_fn_ts.split_feature_skew_ts(
                tr_x, tr_y, te_x, te_y, client_number=2, transforms_pool=[],
            )

    def test_scaling_bounds_check(self):
        tr_x, tr_y, te_x, te_y = _synthetic_dataset()
        with pytest.raises(ValueError):
            split_fn_ts.split_feature_skew_ts(
                tr_x, tr_y, te_x, te_y, client_number=2,
                transforms_pool=["gaussian_noise"],
                scaling_low=0.5, scaling_high=0.1,
            )

    def test_rejects_2d_features(self):
        tr_x, tr_y, te_x, te_y = _synthetic_dataset()
        with pytest.raises(ValueError):
            split_fn_ts.split_feature_skew_ts(
                tr_x.squeeze(1), tr_y, te_x.squeeze(1), te_y, client_number=2,
                transforms_pool=["gaussian_noise"],
            )


class TestAndaDispatch:
    def test_load_split_datasets_dispatches_to_ts(self, monkeypatch):
        # Bypass the actual UCR download by stubbing load_full_datasets.
        tr_x, tr_y, te_x, te_y = _synthetic_dataset(seed=0)
        monkeypatch.setattr(anda, "load_full_datasets",
                            lambda name: [tr_x, tr_y, te_x, te_y])

        result = anda.load_split_datasets(
            dataset_name="UCR:ECG200",
            client_number=3,
            non_iid_type="feature_skew",
            mode="manual",
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
        )
        assert isinstance(result, list)
        assert len(result) == 3
        assert "train_features" in result[0]

    def test_load_split_datasets_rejects_auto_mode_for_ucr(self):
        with pytest.raises(NotImplementedError, match="manual"):
            anda.load_split_datasets(
                dataset_name="UCR:ECG200",
                non_iid_type="feature_skew",
                mode="auto",
            )

    def test_load_split_datasets_rejects_unknown_non_iid_type(self):
        with pytest.raises(NotImplementedError, match="totally_made_up"):
            anda.load_split_datasets(
                dataset_name="UCR:ECG200",
                non_iid_type="totally_made_up",
                mode="manual",
            )

    def test_load_split_datasets_rejects_multi_dataset(self):
        with pytest.raises(ValueError, match="single UCR dataset"):
            anda.load_split_datasets(
                dataset_name="UCR:[ECG200, Coffee]",
                non_iid_type="feature_skew",
                mode="manual",
                transforms_pool=["gaussian_noise"],
            )
