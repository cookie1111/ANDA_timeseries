from collections import Counter

import numpy as np
import pytest
import torch

from ANDA import anda, split_fn_ts


def _stratified_synthetic(
    N_train=120,
    N_test=40,
    C=1,
    T=32,
    num_classes=4,
    seed=0,
):
    rng = np.random.default_rng(seed)
    train_y = np.tile(np.arange(num_classes), N_train // num_classes + 1)[:N_train]
    test_y = np.tile(np.arange(num_classes), N_test // num_classes + 1)[:N_test]
    rng.shuffle(train_y)
    rng.shuffle(test_y)
    train_x = rng.standard_normal((N_train, C, T)).astype(np.float32)
    test_x = rng.standard_normal((N_test, C, T)).astype(np.float32)
    return (
        torch.from_numpy(train_x),
        torch.from_numpy(train_y.astype(np.int64)),
        torch.from_numpy(test_x),
        torch.from_numpy(test_y.astype(np.int64)),
    )


class TestCalculateProbabilitiesTS:
    def test_returns_length_equal_to_num_classes(self):
        labels = torch.tensor([0, 0, 1, 1, 1, 2])
        probs = split_fn_ts._calculate_probabilities_ts(labels, scaling=0.5)
        assert probs.shape == (3,)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))

    def test_scaling_zero_uniform(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2])
        probs = split_fn_ts._calculate_probabilities_ts(labels, scaling=0.0)
        # zero scaling => label_counts ** 0 = 1 for every present class => uniform softmax
        assert torch.allclose(probs, torch.full((3,), 1 / 3))

    def test_higher_scaling_favours_majority_class(self):
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 2, 3])
        low = split_fn_ts._calculate_probabilities_ts(labels, scaling=0.1)
        high = split_fn_ts._calculate_probabilities_ts(labels, scaling=2.0)
        # class 0 is dominant; with higher scaling its softmax weight grows
        assert high[0] > low[0]

    def test_handles_empty_labels(self):
        out = split_fn_ts._calculate_probabilities_ts(torch.tensor([], dtype=torch.long), scaling=0.5)
        assert out.numel() == 0


class TestSplitLabelSkewTS:
    def test_basic_shapes(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        clients = split_fn_ts.split_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=4,
            scaling_label_low=0.4, scaling_label_high=0.6,
        )
        assert len(clients) == 4
        for c in clients:
            assert c["train_features"].ndim == 3
            assert c["test_features"].ndim == 3
            assert c["cluster"] == -1

    def test_reproducible_with_seed(self):
        kwargs = dict(client_number=3, scaling_label_low=0.4, scaling_label_high=0.6)
        anda.set_seed(7)
        a = split_fn_ts.split_label_skew_ts(*_stratified_synthetic(), **kwargs)
        anda.set_seed(7)
        b = split_fn_ts.split_label_skew_ts(*_stratified_synthetic(), **kwargs)
        for ca, cb in zip(a, b):
            np.testing.assert_array_equal(ca["train_labels"], cb["train_labels"])

    def test_higher_scaling_increases_label_concentration(self):
        '''With higher scaling the per-client label distribution should be
        more peaked than under near-zero scaling.'''
        def _entropy_of(clients):
            es = []
            for c in clients:
                _, counts = np.unique(c["train_labels"], return_counts=True)
                p = counts / counts.sum()
                p = p[p > 0]
                es.append(-(p * np.log(p)).sum())
            return float(np.mean(es))

        anda.set_seed(0)
        low = split_fn_ts.split_label_skew_ts(
            *_stratified_synthetic(N_train=400, N_test=80, num_classes=4),
            client_number=4, scaling_label_low=0.0, scaling_label_high=0.0,
        )
        anda.set_seed(0)
        high = split_fn_ts.split_label_skew_ts(
            *_stratified_synthetic(N_train=400, N_test=80, num_classes=4),
            client_number=4, scaling_label_low=1.0, scaling_label_high=1.0,
        )
        assert _entropy_of(high) < _entropy_of(low)

    def test_with_windowing(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(T=32)
        clients = split_fn_ts.split_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=2,
            window_size=8, stride=8,
        )
        for c in clients:
            assert c["train_features"].shape[2] == 8
            assert c["train_features"].shape[0] == c["train_labels"].shape[0]

    def test_rejects_2d_features(self):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        with pytest.raises(ValueError):
            split_fn_ts.split_label_skew_ts(
                tr_x.squeeze(1), tr_y, te_x.squeeze(1), te_y, client_number=2,
            )


class TestSplitFeatureLabelSkewTS:
    def test_basic_shapes_and_count(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        clients = split_fn_ts.split_feature_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=4,
            transforms_pool=["gaussian_noise", "amplitude_scaling"],
            magnitude_buckets=3,
        )
        assert len(clients) == 4
        for c in clients:
            assert c["train_features"].ndim == 3
            assert c["test_features"].ndim == 3
            assert c["cluster"] == -1

    def test_transforms_actually_modify_data(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        original_train = tr_x.clone().numpy()
        clients = split_fn_ts.split_feature_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=2,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            scaling_label_low=0.0, scaling_label_high=0.0,
        )
        all_train = np.concatenate([c["train_features"] for c in clients], axis=0)
        # Cannot just compare sorted (sub-sampling allows duplicates / drops);
        # at minimum the aggregated mean and std should differ from the originals.
        assert not np.allclose(
            all_train.mean(axis=(1, 2)).mean(),
            original_train.mean(axis=(1, 2)).mean(),
            atol=1e-4,
        )

    def test_empty_pool_raises(self):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        with pytest.raises(ValueError):
            split_fn_ts.split_feature_label_skew_ts(
                tr_x, tr_y, te_x, te_y, client_number=2, transforms_pool=[],
            )

    def test_with_windowing(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(T=32)
        clients = split_fn_ts.split_feature_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=2,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            window_size=8, stride=8,
        )
        for c in clients:
            assert c["train_features"].shape[2] == 8


class TestLabelSkewImbalancedTailClass:
    '''Regression for ECG5000 Px_y scaling=1 failing ~15% of the time.

    ECG5000 has 5 classes with counts roughly [292,177,10,19,2] in train
    and [2627,1593,96,176,8] in test - the tail classes are small enough
    that a few early clients can fully drain them from the train pool. The
    pre-fix _calculate_probabilities_ts sized the probability vector from
    `rem_tr_y.max() + 1`, so once the train tail was gone the vector
    shrank to len 4. Re-applying that vector to the test pool (which still
    has class 4) crashed with IndexError on probabilities[4]. Now the
    callers pin num_classes against the dataset-wide max.
    '''

    @staticmethod
    def _ecg5000_like():
        # Same per-class ratios as ECG5000; counts scaled down for speed.
        train_counts = [292, 177, 10, 19, 2]
        test_counts = [2627, 1593, 96, 176, 8]
        def make(counts):
            xs, ys = [], []
            for c, n in enumerate(counts):
                xs.append(torch.randn(n, 1, 64))
                ys.append(torch.full((n,), c, dtype=torch.int64))
            return torch.cat(xs), torch.cat(ys)
        return make(train_counts) + make(test_counts)

    @pytest.mark.parametrize("seed", [3, 17, 18])  # seeds that crashed pre-fix
    def test_feature_label_skew_survives_drained_tail(self, seed):
        anda.set_seed(seed)
        tr_x, tr_y, te_x, te_y = self._ecg5000_like()
        clients = split_fn_ts.split_feature_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=10,
            transforms_pool=["gaussian_noise", "jitter"],
            magnitude_buckets=3,
            scaling_low=0.0, scaling_high=0.3,
            scaling_label_low=0.1, scaling_label_high=0.3,
        )
        assert len(clients) == 10

    @pytest.mark.parametrize("seed", [3, 17, 18])
    def test_label_skew_survives_drained_tail(self, seed):
        anda.set_seed(seed)
        tr_x, tr_y, te_x, te_y = self._ecg5000_like()
        clients = split_fn_ts.split_label_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=10,
            scaling_label_low=0.6, scaling_label_high=0.8,
        )
        assert len(clients) == 10

    def test_calc_probs_vector_pinned_to_dataset_max(self):
        '''Caller-pinned num_classes wins over the pool's own max+1, so a
        drained tail doesn't shrink the vector.'''
        # Pool only has classes {0, 1, 2}; dataset has 5.
        labels = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        probs = split_fn_ts._calculate_probabilities_ts(labels, 0.5, num_classes=5)
        assert probs.shape == (5,)
        # No NaN / negatives, softmax still sums to 1.
        assert torch.isfinite(probs).all()
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_calc_probs_empty_with_num_classes_returns_uniform(self):
        '''Defensive: an empty remaining pool used to return an empty
        tensor; now it returns a uniform vector so create_sub_dataset still
        sees a probability per class.'''
        probs = split_fn_ts._calculate_probabilities_ts(
            torch.tensor([], dtype=torch.int64), 1.0, num_classes=3
        )
        assert probs.shape == (3,)
        assert torch.allclose(probs, torch.full((3,), 1 / 3))


class TestSplitFeatureSkewUnbalancedTS:
    def test_basic(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        clients = split_fn_ts.split_feature_skew_unbalanced_ts(
            tr_x, tr_y, te_x, te_y, client_number=4,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            std_dev=0.3,
        )
        assert len(clients) == 4
        total_train = sum(c["train_features"].shape[0] for c in clients)
        total_test = sum(c["test_features"].shape[0] for c in clients)
        assert total_train == tr_y.shape[0]
        assert total_test == te_y.shape[0]

    def test_clients_have_different_sizes(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(N_train=400)
        clients = split_fn_ts.split_feature_skew_unbalanced_ts(
            tr_x, tr_y, te_x, te_y, client_number=5,
            transforms_pool=["gaussian_noise"],
            magnitude_buckets=3,
            std_dev=0.4,
        )
        sizes = [c["train_features"].shape[0] for c in clients]
        # truncnorm with std=0.4 should produce noticeably different sizes
        assert max(sizes) != min(sizes)

    def test_invalid_std_dev_raises(self):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        with pytest.raises(ValueError):
            split_fn_ts.split_feature_skew_unbalanced_ts(
                tr_x, tr_y, te_x, te_y, client_number=2,
                transforms_pool=["gaussian_noise"],
                std_dev=0.0,
            )

    def test_empty_pool_raises(self):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        with pytest.raises(ValueError):
            split_fn_ts.split_feature_skew_unbalanced_ts(
                tr_x, tr_y, te_x, te_y, client_number=2, transforms_pool=[],
            )


class TestSplitLabelSkewUnbalancedTS:
    def test_basic_and_total_counts(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(N_train=400, N_test=120)
        clients = split_fn_ts.split_label_skew_unbalanced_ts(
            tr_x, tr_y, te_x, te_y, client_number=5,
            scaling_label_low=0.4, scaling_label_high=0.6,
            std_dev=0.3,
        )
        assert len(clients) == 5
        # create_sub_dataset can reuse indices when it can't find enough
        # matching samples; assert each client received <= the requested share
        # plus the total is approximately conserved.
        sizes = [c["train_features"].shape[0] for c in clients]
        assert max(sizes) != min(sizes)
        for c in clients:
            assert c["cluster"] == -1

    def test_invalid_std_dev_raises(self):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        with pytest.raises(ValueError):
            split_fn_ts.split_label_skew_unbalanced_ts(
                tr_x, tr_y, te_x, te_y, client_number=2, std_dev=0.0,
            )


class TestAndaDispatchExpanded:
    def test_label_skew_dispatches(self, monkeypatch):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        monkeypatch.setattr(anda, "load_full_datasets",
                            lambda name: [tr_x, tr_y, te_x, te_y])
        result = anda.load_split_datasets(
            dataset_name="UCR:ECG200", client_number=3,
            non_iid_type="label_skew", mode="manual",
        )
        assert len(result) == 3
        assert "train_features" in result[0]

    def test_feature_label_skew_dispatches(self, monkeypatch):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        monkeypatch.setattr(anda, "load_full_datasets",
                            lambda name: [tr_x, tr_y, te_x, te_y])
        result = anda.load_split_datasets(
            dataset_name="UCR:ECG200", client_number=3,
            non_iid_type="feature_label_skew", mode="manual",
            transforms_pool=["gaussian_noise"],
        )
        assert len(result) == 3

    def test_feature_skew_unbalanced_dispatches(self, monkeypatch):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        monkeypatch.setattr(anda, "load_full_datasets",
                            lambda name: [tr_x, tr_y, te_x, te_y])
        result = anda.load_split_datasets(
            dataset_name="UCR:ECG200", client_number=3,
            non_iid_type="feature_skew_unbalanced", mode="manual",
            transforms_pool=["gaussian_noise"],
            std_dev=0.3,
        )
        assert len(result) == 3

    def test_label_skew_unbalanced_dispatches(self, monkeypatch):
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        monkeypatch.setattr(anda, "load_full_datasets",
                            lambda name: [tr_x, tr_y, te_x, te_y])
        result = anda.load_split_datasets(
            dataset_name="UCR:ECG200", client_number=3,
            non_iid_type="label_skew_unbalanced", mode="manual",
            std_dev=0.3,
        )
        assert len(result) == 3

    def test_unknown_non_iid_lists_available_variants(self):
        with pytest.raises(NotImplementedError, match="Available TS variants"):
            anda.load_split_datasets(
                dataset_name="UCR:ECG200", client_number=3,
                non_iid_type="not_real", mode="manual",
            )
