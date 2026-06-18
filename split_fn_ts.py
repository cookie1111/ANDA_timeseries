from collections import Counter, defaultdict
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm

from .transforms_ts import TRANSFORM_BUCKETS, TRANSFORMS, apply_transform
from .utils import create_sub_dataset, split_basic, split_unbalanced
from .windowing_ts import window_series


def set_seed(RANDOM_SEED: int = 42):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def _resolve_buckets(
    name: str,
    magnitude_buckets,
    custom_buckets: dict,
) -> list:
    '''Resolve which kwargs presets to use for a transform.

    Lookup order: custom_buckets[name] -> TRANSFORM_BUCKETS[name] -> [{}].
    If magnitude_buckets is an int and smaller than the resolved bucket list,
    evenly-spaced indices are picked (preserving low / high coverage).
    '''
    if name not in TRANSFORMS:
        raise KeyError(
            f"Unknown transform {name!r}. Known: {sorted(TRANSFORMS)}"
        )

    if custom_buckets and name in custom_buckets:
        buckets = [dict(b) for b in custom_buckets[name]]
    elif name in TRANSFORM_BUCKETS:
        buckets = [dict(b) for b in TRANSFORM_BUCKETS[name]]
    else:
        buckets = [{}]

    if isinstance(magnitude_buckets, int) and magnitude_buckets > 0 and len(buckets) > magnitude_buckets:
        idx = np.linspace(0, len(buckets) - 1, magnitude_buckets).round().astype(int)
        buckets = [buckets[i] for i in idx]

    return buckets


def _build_pool(
    transforms_pool: list,
    magnitude_buckets,
    custom_buckets: dict,
) -> list:
    '''Expand a list of transform names into a flat list of (name, kwargs) entries.'''
    if not transforms_pool:
        raise ValueError("transforms_pool must contain at least one transform name.")
    pool = []
    for name in transforms_pool:
        for kwargs in _resolve_buckets(name, magnitude_buckets, custom_buckets):
            pool.append((name, kwargs))
    return pool


def assigning_transform_features(
    datapoint_number: int,
    pool: list,
    scaling: float,
    random_order: bool = True,
) -> list:
    '''
    Assign one (transform_name, kwargs) entry to each datapoint by sampling from
    a softmax distribution over `pool`. Mirrors `assigning_rotation_features`
    but works on an arbitrary discrete (transform, bucket) pool.

    Args:
        datapoint_number (int): Number of per-sample assignments to draw.
        pool (list[tuple[str, dict]]): The discrete pool to draw from.
        scaling (float): Softmax peakiness in [0, 1]. 0 = uniform over pool.
        random_order (bool): Shuffle the pool before computing probabilities so
            that different clients can favour different pool entries.

    Returns:
        list[tuple[str, dict]]: One pool entry per datapoint.
    '''
    if not 0.0 <= scaling <= 1.0:
        raise ValueError("scaling must be in [0, 1].")
    if len(pool) == 0:
        raise ValueError("pool must be non-empty.")

    pool = list(pool)
    if random_order:
        order = np.random.permutation(len(pool))
        pool = [pool[i] for i in order]

    values = np.arange(len(pool), 0, -1, dtype=np.float64) * scaling
    probs = np.exp(values)
    probs = probs / probs.sum()

    idx = np.random.choice(len(pool), size=datapoint_number, p=probs)
    return [pool[i] for i in idx]


def _apply_per_sample_transforms(features: torch.Tensor, assignments: list) -> torch.Tensor:
    '''Apply per-sample (name, kwargs) transforms efficiently by grouping
    samples that share the same (name, kwargs) and running a single batched
    call per group.'''
    if len(assignments) != features.shape[0]:
        raise ValueError("assignments must have one entry per sample.")

    groups = defaultdict(list)
    for i, (name, kwargs) in enumerate(assignments):
        key = (name, tuple(sorted(kwargs.items())))
        groups[key].append(i)

    out = features.clone()
    for (name, kw_items), indices in groups.items():
        idx_t = torch.as_tensor(indices, dtype=torch.long)
        sub = out.index_select(0, idx_t)
        sub_t = apply_transform(sub, name, dict(kw_items))
        out.index_copy_(0, idx_t, sub_t)
    return out


def _per_client_transform_pair(
    tr_x: torch.Tensor,
    te_x: torch.Tensor,
    transforms_pool: list,
    magnitude_buckets,
    custom_buckets: dict,
    scaling_low: float,
    scaling_high: float,
    random_order: bool,
    verbose: bool,
    client_id: int,
):
    '''Sample one (transform, bucket) assignment per row over the combined
    train+test batch (so a client gets the same pool ordering and scaling for
    both), then apply transforms.'''
    pool = _build_pool(transforms_pool, magnitude_buckets, custom_buckets)
    client_scaling = float(np.random.uniform(scaling_low, scaling_high))
    n_tr = int(tr_x.shape[0])
    n_te = int(te_x.shape[0])
    total = assigning_transform_features(
        n_tr + n_te, pool, client_scaling, random_order=random_order
    )
    tr_assign = total[:n_tr]
    te_assign = total[n_tr:]

    if verbose:
        counts = Counter(name for name, _ in total)
        print(
            f"Client {client_id} | scaling={client_scaling:.3f} | "
            f"transform counts: {dict(counts)}"
        )

    tr_x = _apply_per_sample_transforms(tr_x, tr_assign) if n_tr else tr_x
    te_x = _apply_per_sample_transforms(te_x, te_assign) if n_te else te_x
    return tr_x, te_x


def _window_pair(
    tr_x: torch.Tensor,
    tr_y: torch.Tensor,
    te_x: torch.Tensor,
    te_y: torch.Tensor,
    window_size: int,
    stride: int,
    drop_last: bool,
):
    '''Window a (train, test) feature/label pair if window_size is set.'''
    if window_size is None:
        return tr_x, tr_y, te_x, te_y
    if tr_x.shape[0] > 0:
        tr_x, tr_y = window_series(tr_x, tr_y, window_size, stride=stride, drop_last=drop_last)
    if te_x.shape[0] > 0:
        te_x, te_y = window_series(te_x, te_y, window_size, stride=stride, drop_last=drop_last)
    return tr_x, tr_y, te_x, te_y


def _calculate_probabilities_ts(labels: torch.Tensor, scaling: float,
                                num_classes: int = None) -> torch.Tensor:
    '''Per-class softmax probabilities scaled by class frequency.

    Equivalent to `utils.calculate_probabilities` but auto-sizes the probability
    vector to the number of classes present in `labels` instead of hardcoding 10
    (which is correct for MNIST/CIFAR10 but wrong for UCR datasets with K != 10
    classes - phantom classes would dilute the softmax).

    Args:
        labels: Current label pool (may be a drained remainder).
        scaling: Softmax temperature exponent.
        num_classes: Pin the probability-vector length to the dataset-wide
            class count. When callers iterate per-client and the pool drains
            tail classes, sizing from `labels.max() + 1` shrinks the vector,
            and any downstream user that indexes by the original label space
            (e.g. the test pool which still has those classes) goes out of
            bounds. Always pass this from the caller's dataset-wide max.
    '''
    if labels.numel() == 0:
        if num_classes is None:
            return torch.tensor([], dtype=torch.float32)
        return torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32)
    pool_max = int(labels.max().item()) + 1
    if num_classes is None:
        num_classes = pool_max
    else:
        num_classes = max(num_classes, pool_max)
    label_counts = torch.bincount(labels, minlength=num_classes).float()
    scaled_counts = label_counts ** scaling
    return F.softmax(scaled_counts, dim=0)


def _pack_client(tr_x, tr_y, te_x, te_y, cluster=-1) -> dict:
    return {
        'train_features': tr_x.detach().cpu().numpy(),
        'train_labels': tr_y.detach().cpu().numpy(),
        'test_features': te_x.detach().cpu().numpy(),
        'test_labels': te_y.detach().cpu().numpy(),
        'cluster': cluster,
    }


def _validate_ts_inputs(train_features, train_labels, test_features, test_labels):
    if train_features.dim() != 3 or test_features.dim() != 3:
        raise ValueError("Features must be 3D tensors of shape (N, C, T).")
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features and train_labels must have matching N.")
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test_features and test_labels must have matching N.")


def split_feature_skew_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    transforms_pool: list = None,
    magnitude_buckets: int = 3,
    custom_buckets: dict = None,
    scaling_low: float = 0.0,
    scaling_high: float = 0.4,
    random_order: bool = True,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    permute: bool = True,
    verbose: bool = False,
) -> list:
    '''
    Time-series counterpart of `split_feature_skew`: split into `client_number`
    clusters and inject feature heterogeneity by giving each client its own
    softmax distribution over a discrete (transform, bucket) pool.

    Args:
        train_features, train_labels, test_features, test_labels (torch.Tensor):
            Inputs of shape (N, C, T) and (N,).
        client_number (int): Number of clients to produce.
        transforms_pool (list[str]): Names of registered transforms to expose
            to clients. Required.
        magnitude_buckets (int): Number of magnitude buckets to keep per
            transform.
        custom_buckets (dict[str, list[dict]] | None): Override default buckets
            for specific transforms.
        scaling_low, scaling_high (float): Bounds on the per-client softmax
            peakiness.
        random_order (bool): Permute the pool per client.
        window_size, stride, drop_last: Forwarded to `window_series` after
            transforms are applied.
        permute (bool): Shuffle samples before basic split.
        verbose (bool): Print per-client transform counts.

    Returns:
        list[dict]: One dict per client with `train_features`, `train_labels`,
            `test_features`, `test_labels`, `cluster=-1`.
    '''
    if transforms_pool is None or len(transforms_pool) == 0:
        raise ValueError("transforms_pool must be a non-empty list of registered transform names.")
    if scaling_high < scaling_low:
        raise ValueError("scaling_high must be >= scaling_low.")
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)

    train_clients = split_basic(train_features, train_labels, client_number, permute=permute)
    test_clients = split_basic(test_features, test_labels, client_number, permute=permute)

    rearranged = []
    for cid, (tr, te) in enumerate(zip(train_clients, test_clients)):
        tr_x, te_x = _per_client_transform_pair(
            tr['features'], te['features'], transforms_pool, magnitude_buckets,
            custom_buckets, scaling_low, scaling_high, random_order, verbose, cid,
        )
        tr_x, tr_y, te_x, te_y = _window_pair(
            tr_x, tr['labels'], te_x, te['labels'], window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(tr_x, tr_y, te_x, te_y))

    return rearranged


def split_label_skew_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    verbose: bool = False,
) -> list:
    '''
    Time-series counterpart of `split_label_skew`: per-client sub-sample the
    training and test pools using a softmax over class frequency so each client
    sees a skewed class distribution. No X-side augmentation.

    The softmax is computed against the actual class count of the TS labels
    (via `_calculate_probabilities_ts`) so it is correct for any K-class
    dataset, unlike the image-side helper which hardcodes K=10.
    '''
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)
    if scaling_label_high < scaling_label_low:
        raise ValueError("scaling_label_high must be >= scaling_label_low.")

    avg_tr = train_labels.shape[0] // client_number
    avg_te = test_labels.shape[0] // client_number

    rem_tr_x, rem_tr_y = train_features, train_labels
    rem_te_x, rem_te_y = test_features, test_labels

    # Dataset-wide class count, pinned now so the probability vector stays
    # the right size even after the train pool drains tail classes.
    num_classes_total = int(max(train_labels.max().item(),
                                 test_labels.max().item())) + 1

    rearranged = []
    for cid in range(client_number):
        client_scaling = float(np.random.uniform(scaling_label_low, scaling_label_high))
        # The 0.6 dampening factor here matches split_label_skew's calling convention;
        # it keeps the effective per-client skew comparable to the image version.
        probs = _calculate_probabilities_ts(rem_tr_y, client_scaling * 0.6,
                                            num_classes=num_classes_total)

        sub_tr_x, sub_tr_y, rem_tr_x, rem_tr_y = create_sub_dataset(rem_tr_x, rem_tr_y, probs, avg_tr)
        sub_te_x, sub_te_y, rem_te_x, rem_te_y = create_sub_dataset(rem_te_x, rem_te_y, probs, avg_te)

        if verbose:
            counts = Counter(sub_tr_y.tolist())
            print(f"Client {cid} | scaling={client_scaling:.3f} | label counts: {dict(counts)}")

        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = _window_pair(
            sub_tr_x, sub_tr_y, sub_te_x, sub_te_y, window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(sub_tr_x, sub_tr_y, sub_te_x, sub_te_y))

    return rearranged


def split_feature_label_skew_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    transforms_pool: list = None,
    magnitude_buckets: int = 3,
    custom_buckets: dict = None,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    scaling_low: float = 0.0,
    scaling_high: float = 0.4,
    random_order: bool = True,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    verbose: bool = False,
) -> list:
    '''
    Time-series counterpart of `split_feature_label_skew`: combines label-based
    sub-sampling with per-sample (transform, bucket) assignment. Each client
    gets both a class-frequency softmax and a transforms-pool softmax.
    '''
    if transforms_pool is None or len(transforms_pool) == 0:
        raise ValueError("transforms_pool must be a non-empty list of registered transform names.")
    if scaling_high < scaling_low or scaling_label_high < scaling_label_low:
        raise ValueError("scaling_*_high must be >= scaling_*_low.")
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)

    avg_tr = train_labels.shape[0] // client_number
    avg_te = test_labels.shape[0] // client_number

    rem_tr_x, rem_tr_y = train_features, train_labels
    rem_te_x, rem_te_y = test_features, test_labels

    # See split_label_skew_ts: pin num_classes so the probs vector keeps the
    # right length once the train pool drains tail classes.
    num_classes_total = int(max(train_labels.max().item(),
                                 test_labels.max().item())) + 1

    rearranged = []
    for cid in range(client_number):
        label_scale = float(np.random.uniform(scaling_label_low, scaling_label_high))
        probs = _calculate_probabilities_ts(rem_tr_y, label_scale,
                                            num_classes=num_classes_total)

        sub_tr_x, sub_tr_y, rem_tr_x, rem_tr_y = create_sub_dataset(rem_tr_x, rem_tr_y, probs, avg_tr)
        sub_te_x, sub_te_y, rem_te_x, rem_te_y = create_sub_dataset(rem_te_x, rem_te_y, probs, avg_te)

        sub_tr_x, sub_te_x = _per_client_transform_pair(
            sub_tr_x, sub_te_x, transforms_pool, magnitude_buckets, custom_buckets,
            scaling_low, scaling_high, random_order, verbose, cid,
        )
        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = _window_pair(
            sub_tr_x, sub_tr_y, sub_te_x, sub_te_y, window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(sub_tr_x, sub_tr_y, sub_te_x, sub_te_y))

    return rearranged


def split_feature_skew_unbalanced_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    transforms_pool: list = None,
    magnitude_buckets: int = 3,
    custom_buckets: dict = None,
    scaling_low: float = 0.0,
    scaling_high: float = 0.4,
    random_order: bool = True,
    std_dev: float = 0.1,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    permute: bool = True,
    verbose: bool = False,
) -> list:
    '''
    Unbalanced variant of `split_feature_skew_ts`: each client receives an
    unequal number of samples drawn via a truncated-normal share of the total
    (per `split_unbalanced`), then feature heterogeneity is applied as in
    `split_feature_skew_ts`.
    '''
    if transforms_pool is None or len(transforms_pool) == 0:
        raise ValueError("transforms_pool must be a non-empty list of registered transform names.")
    if scaling_high < scaling_low:
        raise ValueError("scaling_high must be >= scaling_low.")
    if std_dev <= 0:
        raise ValueError("std_dev must be > 0.")
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)

    train_clients = split_unbalanced(train_features, train_labels, client_number, std_dev, permute)
    test_clients = split_unbalanced(test_features, test_labels, client_number, std_dev, permute)

    if verbose:
        for i, (tr, te) in enumerate(zip(train_clients, test_clients)):
            print(f"Client {i} | train n={tr['labels'].shape[0]} | test n={te['labels'].shape[0]}")

    rearranged = []
    for cid, (tr, te) in enumerate(zip(train_clients, test_clients)):
        tr_x, te_x = _per_client_transform_pair(
            tr['features'], te['features'], transforms_pool, magnitude_buckets,
            custom_buckets, scaling_low, scaling_high, random_order, verbose, cid,
        )
        tr_x, tr_y, te_x, te_y = _window_pair(
            tr_x, tr['labels'], te_x, te['labels'], window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(tr_x, tr_y, te_x, te_y))

    return rearranged


def split_label_skew_unbalanced_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    scaling_label_low: float = 0.4,
    scaling_label_high: float = 0.6,
    std_dev: float = 0.1,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    verbose: bool = False,
) -> list:
    '''
    Unbalanced variant of `split_label_skew_ts`: each client gets an unequal
    number of samples drawn via truncated-normal shares, then per-client label
    softmax skew is applied via class-frequency-aware probabilities.
    '''
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)
    if scaling_label_high < scaling_label_low:
        raise ValueError("scaling_label_high must be >= scaling_label_low.")
    if std_dev <= 0:
        raise ValueError("std_dev must be > 0.")

    def _shares(total, n_clients):
        pct = truncnorm.rvs(-0.5 / std_dev, 0.5 / std_dev, loc=0.5, scale=std_dev, size=n_clients)
        normalized = pct / pct.sum()
        counts = (normalized * total).astype(int)
        # Patch rounding drift so we land exactly on `total`.
        diff = total - counts.sum()
        for i in range(abs(diff)):
            counts[i % n_clients] += np.sign(diff)
        return counts

    train_counts = _shares(train_labels.shape[0], client_number)
    test_counts = _shares(test_labels.shape[0], client_number)

    rem_tr_x, rem_tr_y = train_features, train_labels
    rem_te_x, rem_te_y = test_features, test_labels

    # See split_label_skew_ts: pin num_classes against the dataset-wide max.
    num_classes_total = int(max(train_labels.max().item(),
                                 test_labels.max().item())) + 1

    rearranged = []
    for cid in range(client_number):
        client_scaling = float(np.random.uniform(scaling_label_low, scaling_label_high))
        probs = _calculate_probabilities_ts(rem_tr_y, client_scaling,
                                            num_classes=num_classes_total)

        n_tr = int(train_counts[cid])
        n_te = int(test_counts[cid])
        sub_tr_x, sub_tr_y, rem_tr_x, rem_tr_y = create_sub_dataset(rem_tr_x, rem_tr_y, probs, n_tr)
        sub_te_x, sub_te_y, rem_te_x, rem_te_y = create_sub_dataset(rem_te_x, rem_te_y, probs, n_te)

        if verbose:
            counts = Counter(sub_tr_y.tolist())
            print(
                f"Client {cid} | train n={n_tr} test n={n_te} | "
                f"scaling={client_scaling:.3f} | label counts: {dict(counts)}"
            )

        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = _window_pair(
            sub_tr_x, sub_tr_y, sub_te_x, sub_te_y, window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(sub_tr_x, sub_tr_y, sub_te_x, sub_te_y))

    return rearranged


def split_label_condition_skew_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    mixing_label_number: int = 2,
    mixing_label_list: list = None,
    random_mode: bool = True,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    verbose: bool = False,
) -> list:
    '''Time-series counterpart of `split_label_condition_skew_strict`: a true
    P(y|x) shift.

    Clients share P(x): the data is partitioned uniformly across clients with
    no feature augmentation. They differ in P(y|x): each client is assigned to
    a cluster, and each cluster owns a deterministic permutation of a subset
    of labels (the "swap pool"). All samples in a client whose label lies in
    the pool get relabeled according to that permutation; labels outside the
    pool are untouched. Different clients with the same cluster see the same
    re-labeling rule (real ground-truth cluster structure - finally
    addresses the missing-cluster-labels gap from docs §4.2 for the Py_x
    axis).

    Strength is controlled by `mixing_label_number`: with K labels in the
    pool there are K! permutations, so K! candidate clusters. K=1 yields a
    single identity cluster (no shift -- the IID baseline). K=2 yields the
    minimal binary swap (identity + transposition); higher K gives more
    diverse re-labeling rules, growing factorially.

    No feature augmentation is applied (P(x) is preserved across clients),
    which is what makes this a genuine P(y|x) shift rather than a joint
    label-dominant skew. If you want P(x) variation on top, layer a feature
    augmentation pass post-hoc or use feature_label_skew_ts.

    Args:
        train_features, train_labels: TS train pool of shape (N, C, T) and (N,).
        test_features, test_labels: TS test pool, same shapes.
        client_number: Number of clients to produce.
        mixing_label_number: Size of the swap pool. Clamped to
            [1, num_classes_total]; 1 is the IID baseline (single identity
            cluster) and the pool never exceeds the label space.
        mixing_label_list: Explicit swap pool (list of label ids). Overrides
            random sampling. Useful for reproducible non-IID setups across
            datasets with different class IDs.
        random_mode: If True, sample mixing_label_list uniformly from the
            label space; if False, mixing_label_list must be provided.
        window_size, stride, drop_last: Optional windowing pass applied to
            each client's (X, y) at the end - same semantics as the other
            *_ts splits.

    Returns:
        list of n_clients dicts with keys train_features, train_labels,
        test_features, test_labels, cluster. The cluster field is the index
        into the enumerated permutation list, which means callers can
        compare flux's inferred clusters against a real ground truth here
        (unlike feature_skew_ts / label_skew_ts / feature_label_skew_ts
        where cluster=-1).
    '''
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)
    if client_number < 1:
        raise ValueError("client_number must be >= 1.")

    num_classes_total = int(max(train_labels.max().item(),
                                 test_labels.max().item())) + 1
    if num_classes_total < 2:
        raise ValueError("Need at least 2 classes for a P(y|x) shift.")

    # Resolve the swap pool. Cap at num_classes_total so we don't ask for more
    # labels than exist; floor at 1 so mixing_label_number=1 is the IID
    # baseline (1! = 1 identity permutation = no shift), matching the
    # image-side label_condition_skew where scaling=1 -> mixing_label_number=1.
    # Datasets with very few classes simply get a smaller pool.
    if random_mode:
        pool_size = max(1, min(int(mixing_label_number), num_classes_total))
        mixing_label_list = np.random.choice(
            num_classes_total, size=pool_size, replace=False
        ).tolist()
    else:
        if not mixing_label_list:
            raise ValueError("Non-random mode requires a mixing_label_list.")
        if len(mixing_label_list) != len(set(mixing_label_list)):
            raise ValueError("mixing_label_list must not contain duplicates.")
        if any(not (0 <= int(v) < num_classes_total) for v in mixing_label_list):
            raise ValueError(
                f"mixing_label_list values must lie in [0, {num_classes_total})."
            )
        if len(mixing_label_list) < 2:
            raise ValueError("mixing_label_list must contain at least 2 labels.")
        mixing_label_list = [int(v) for v in mixing_label_list]

    # Enumerate all permutations of the pool. Each becomes one cluster.
    all_label_maps = [
        dict(zip(mixing_label_list, perm))
        for perm in itertools.permutations(mixing_label_list)
    ]

    if verbose:
        print(f"label_condition_skew_ts: pool={mixing_label_list}, "
              f"{len(all_label_maps)} candidate clusters.")

    # Partition the data uniformly across clients (no skew on P(x) or P(y)).
    basic_train = split_basic(train_features, train_labels, client_number)
    basic_test = split_basic(test_features, test_labels, client_number)

    rearranged = []
    for cid in range(client_number):
        cluster_id = int(np.random.randint(0, len(all_label_maps)))
        label_map = all_label_maps[cluster_id]

        sub_tr_x = basic_train[cid]['features']
        sub_te_x = basic_test[cid]['features']
        sub_tr_y = basic_train[cid]['labels'].clone()
        sub_te_y = basic_test[cid]['labels'].clone()

        # Apply the cluster's permutation to in-pool labels only.
        for original, permuted in label_map.items():
            sub_tr_y[basic_train[cid]['labels'] == original] = permuted
            sub_te_y[basic_test[cid]['labels'] == original] = permuted

        if verbose:
            print(f"  client {cid} -> cluster {cluster_id} (map={label_map})")

        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = _window_pair(
            sub_tr_x, sub_tr_y, sub_te_x, sub_te_y, window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(sub_tr_x, sub_tr_y, sub_te_x, sub_te_y,
                                       cluster=cluster_id))

    return rearranged


def split_feature_condition_skew_ts(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    client_number: int = 10,
    targeted_label_number: int = 1,
    targeted_label_list: list = None,
    transforms_pool: list = None,
    magnitude_buckets: int = 3,
    custom_buckets: dict = None,
    random_mode: bool = True,
    window_size: int = None,
    stride: int = None,
    drop_last: bool = True,
    verbose: bool = False,
) -> list:
    '''Time-series counterpart of `split_feature_condition_skew_strict`: a true
    P(x|y) shift.

    Clients share P(y) (uniform partition via split_basic, which permutes
    before slicing). For each cluster a deterministic (transform, bucket)
    signature is fixed: every client in cluster k applies that signature to
    samples whose label is in the targeted pool, leaving non-targeted samples
    untouched. So P(x | y not in pool) is shared across clusters, while
    P(x | y in pool) differs by cluster.

    Image-side analog (`split_feature_condition_skew_strict`) factors the
    cluster signature over two orthogonal axes -- rotation X colour -- and
    targets a separate label list for each. TS has only one augmentation
    axis (the transform pool), so this implementation collapses that to one
    pool + one targeted-label list. The defining property (P(y) shared,
    P(x | y in pool) differs by cluster) is preserved exactly.

    Strength is controlled by the cluster count, which equals len(_build_pool
    (transforms_pool, magnitude_buckets, custom_buckets)). More transforms
    or more buckets => more candidate clusters => more diverse P(x|y) maps.

    Args:
        train_features, train_labels: TS train pool of shape (N, C, T)/(N,).
        test_features, test_labels: TS test pool, same shapes.
        client_number: Number of clients.
        targeted_label_number: Size of the targeted-label pool. Clipped to
            min(targeted_label_number, num_classes_total - 1) so at least
            one label remains untargeted (otherwise the shift collapses to
            split_feature_skew_ts).
        targeted_label_list: Explicit targeted pool; overrides random_mode.
        transforms_pool: Names of registered transforms. Cluster signatures
            are the (transform, bucket) pairs expanded from this pool.
        magnitude_buckets: How many bucket presets per transform to keep.
        custom_buckets: Override the bucket dict per transform.
        random_mode: If True, sample targeted_label_list uniformly.
        window_size, stride, drop_last: Optional windowing pass.

    Returns:
        list of n_clients dicts with train/test features+labels and a real
        cluster index (>=0) - same convention as split_label_condition_skew_ts,
        which means flux/feroma can compute extrinsic clustering metrics
        (ARI / NMI / purity) against ground truth on this shift too.
    '''
    _validate_ts_inputs(train_features, train_labels, test_features, test_labels)
    if client_number < 1:
        raise ValueError("client_number must be >= 1.")
    if not transforms_pool:
        raise ValueError("transforms_pool must be a non-empty list of transform names.")

    num_classes_total = int(max(train_labels.max().item(),
                                 test_labels.max().item())) + 1
    if num_classes_total < 2:
        raise ValueError("Need at least 2 classes for a P(x|y) shift.")

    # Resolve targeted pool. Clip below num_classes_total so at least one
    # untouched label remains; clip at >=1 so there's something to skew.
    max_targeted = max(1, num_classes_total - 1)
    if random_mode:
        pool_size = max(1, min(int(targeted_label_number), max_targeted))
        targeted_label_list = np.random.choice(
            num_classes_total, size=pool_size, replace=False
        ).tolist()
    else:
        if not targeted_label_list:
            raise ValueError("Non-random mode requires a targeted_label_list.")
        if len(targeted_label_list) != len(set(targeted_label_list)):
            raise ValueError("targeted_label_list must not contain duplicates.")
        if any(not (0 <= int(v) < num_classes_total) for v in targeted_label_list):
            raise ValueError(
                f"targeted_label_list values must lie in [0, {num_classes_total})."
            )
        if len(targeted_label_list) >= num_classes_total:
            raise ValueError(
                "targeted_label_list must leave at least one untouched label; "
                f"got {len(targeted_label_list)} of {num_classes_total} classes."
            )
        targeted_label_list = [int(v) for v in targeted_label_list]

    # Enumerate cluster signatures from the (transform, bucket) pool.
    pool = _build_pool(transforms_pool, magnitude_buckets, custom_buckets)
    if len(pool) < 2:
        raise ValueError(
            "Need >= 2 cluster signatures. Expand transforms_pool or raise "
            "magnitude_buckets."
        )

    if verbose:
        print(f"feature_condition_skew_ts: targeted_labels={targeted_label_list}, "
              f"{len(pool)} candidate clusters.")

    # Uniform partition (split_basic permutes by default => P(y) shared).
    basic_train = split_basic(train_features, train_labels, client_number)
    basic_test = split_basic(test_features, test_labels, client_number)

    rearranged = []
    targeted_set = set(targeted_label_list)
    for cid in range(client_number):
        cluster_id = int(np.random.randint(0, len(pool)))
        name, kwargs = pool[cluster_id]

        sub_tr_x = basic_train[cid]['features'].clone()
        sub_te_x = basic_test[cid]['features'].clone()
        sub_tr_y = basic_train[cid]['labels']
        sub_te_y = basic_test[cid]['labels']

        # Apply the cluster's transform ONLY to in-pool labels. Out-of-pool
        # samples pass through unchanged - that's the defining property.
        for label in targeted_set:
            tr_mask = (sub_tr_y == label)
            te_mask = (sub_te_y == label)
            if tr_mask.any():
                sub_tr_x[tr_mask] = apply_transform(sub_tr_x[tr_mask], name, kwargs)
            if te_mask.any():
                sub_te_x[te_mask] = apply_transform(sub_te_x[te_mask], name, kwargs)

        if verbose:
            print(f"  client {cid} -> cluster {cluster_id} "
                  f"(transform={name}, kwargs={kwargs})")

        sub_tr_x, sub_tr_y, sub_te_x, sub_te_y = _window_pair(
            sub_tr_x, sub_tr_y, sub_te_x, sub_te_y, window_size, stride, drop_last,
        )
        rearranged.append(_pack_client(sub_tr_x, sub_tr_y, sub_te_x, sub_te_y,
                                       cluster=cluster_id))

    return rearranged
