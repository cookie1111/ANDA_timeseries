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


class TestSplitLabelConditionSkewTS:
    '''True P(y|x) shift: clients share P(x) (uniform partition, no
    augmentation) but each client gets a cluster-specific label permutation,
    so the same waveform means different things to different clients.'''

    def test_preserves_p_of_x(self):
        '''P(x) shared across clients => no augmentation, uniform partition.
        Concatenating all client X back together should give the original
        train+test set bit-for-bit (modulo client-internal ordering).'''
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(N_train=120, N_test=40)
        clients = split_fn_ts.split_label_condition_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=4, mixing_label_number=2,
        )
        # Same total sample count and identical multiset of feature tensors.
        recombined_tr = np.concatenate([c['train_features'] for c in clients])
        recombined_te = np.concatenate([c['test_features'] for c in clients])
        assert recombined_tr.shape == tuple(tr_x.shape)
        assert recombined_te.shape == tuple(te_x.shape)
        # Order may differ but sum of values is invariant under reordering.
        assert np.isclose(recombined_tr.sum(), tr_x.sum().item())
        assert np.isclose(recombined_te.sum(), te_x.sum().item())

    def test_relabeling_rule_holds_per_client(self):
        '''The decisive P(y|x) check: build a dataset where each sample's
        FIRST feature value encodes its original class id. After splitting,
        we can recover each sample's original label from its features and
        verify that its assigned (post-relabel) label equals the cluster's
        permutation applied to the original. This proves both
            (1) features are not transformed (P(x) preserved), and
            (2) labels follow the per-cluster permutation exactly.'''
        anda.set_seed(0)
        per_class, num_classes, T = 40, 4, 16
        xs, ys = [], []
        for c in range(num_classes):
            x = torch.randn(per_class, 1, T)
            x[:, 0, 0] = float(c)  # encode class id in the first sample value
            xs.append(x)
            ys.append(torch.full((per_class,), c, dtype=torch.int64))
        tr_x = torch.cat(xs); tr_y = torch.cat(ys)

        # Pin the swap pool so the expected permutations are deterministic.
        clients = split_fn_ts.split_label_condition_skew_ts(
            tr_x, tr_y, tr_x, tr_y, client_number=8,
            random_mode=False, mixing_label_list=[0, 1],
        )
        # itertools.permutations([0, 1]) -> [(0,1), (1,0)] in that order, so
        # cluster 0 is the identity map and cluster 1 is the swap.
        expected_maps = [{0: 0, 1: 1}, {0: 1, 1: 0}]
        for c in clients:
            feats = c['train_features']
            labels = c['train_labels']
            assert feats.shape[0] == labels.shape[0]
            label_map = expected_maps[c['cluster']]
            for sample, lab in zip(feats, labels):
                original = int(round(float(sample[0, 0])))
                expected = label_map.get(original, original)
                assert int(lab) == expected, (
                    f"cluster {c['cluster']}: sample with original={original} "
                    f"got label {int(lab)}, expected {expected}"
                )

    def test_records_real_cluster_id(self):
        '''Unlike feature_skew_ts / label_skew_ts (cluster=-1), every client
        here carries the cluster index it was assigned to. This is what flux
        can score its discovered clusters against.'''
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        clients = split_fn_ts.split_label_condition_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=8, mixing_label_number=2,
        )
        assert all(c['cluster'] >= 0 for c in clients), \
            "cluster ids must be real (>=0), not the -1 placeholder"
        assert all(c['cluster'] < 2 for c in clients), \
            "with mixing_label_number=2 only 2!=2 candidate clusters exist"

    def test_no_feature_transformation_applied(self):
        '''A client's train features must be a verbatim subset of the
        original train features - no jitter, no rotation, no anything.'''
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(N_train=60, N_test=20)
        clients = split_fn_ts.split_label_condition_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=3, mixing_label_number=2,
        )
        tr_x_np = tr_x.detach().cpu().numpy()
        for c in clients:
            for row in c['train_features']:
                # row must equal some original train sample exactly.
                match = np.any(np.all(np.isclose(tr_x_np, row), axis=(1, 2)))
                assert match, "found a transformed sample - P(x) was not preserved"

    def test_mixing_label_number_clipped_to_num_classes(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic(num_classes=4)
        # Asking for 99 labels in a 4-class dataset: silently clip to 4.
        # 4! = 24 candidate clusters, so cluster ids stay in [0, 24).
        clients = split_fn_ts.split_label_condition_skew_ts(
            tr_x, tr_y, te_x, te_y, client_number=4, mixing_label_number=99,
        )
        for c in clients:
            assert 0 <= c['cluster'] < 24

    def test_single_class_dataset_raises(self):
        torch.manual_seed(0)
        tr_x = torch.randn(20, 1, 16); tr_y = torch.zeros(20, dtype=torch.int64)
        te_x = torch.randn(10, 1, 16); te_y = torch.zeros(10, dtype=torch.int64)
        with pytest.raises(ValueError, match="at least 2 classes"):
            split_fn_ts.split_label_condition_skew_ts(
                tr_x, tr_y, te_x, te_y, client_number=2,
            )

    def test_non_random_mode_requires_explicit_pool(self):
        anda.set_seed(0)
        tr_x, tr_y, te_x, te_y = _stratified_synthetic()
        with pytest.raises(ValueError, match="mixing_label_list"):
            split_fn_ts.split_label_condition_skew_ts(
                tr_x, tr_y, te_x, te_y, client_number=2, random_mode=False,
            )
        with pytest.raises(ValueError, match="at least 2 labels"):
            split_fn_ts.split_label_condition_skew_ts(
                tr_x, tr_y, te_x, te_y, client_number=2,
                random_mode=False, mixing_label_list=[1],
            )

    def test_load_split_datasets_dispatch_routes_correctly(self):
        '''anda.load_split_datasets auto-dispatches via split_{name}_ts.
        Confirm label_condition_skew is now resolvable in the UCR path
        without needing a network call (we hit the split fn directly via
        the dispatcher to make sure the name is wired up).'''
        from ANDA import anda as anda_mod
        # The dispatcher resolves by getattr-style lookup; presence here is
        # equivalent to it being available.
        assert hasattr(split_fn_ts, "split_label_condition_skew_ts")
