from collections import Counter, defaultdict

import numpy as np
import torch

from .transforms_ts import TRANSFORM_BUCKETS, TRANSFORMS, apply_transform
from .utils import split_basic
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

    The flow per client is: basic random split -> per-sample (transform, bucket)
    assignment via softmax -> apply transforms in batches -> optional windowing.

    Args:
        train_features, train_labels, test_features, test_labels (torch.Tensor):
            Inputs of shape (N, C, T) and (N,).
        client_number (int): Number of clients to produce.
        transforms_pool (list[str]): Names of registered transforms to expose
            to clients. Required.
        magnitude_buckets (int): Number of magnitude buckets to keep per
            transform. Evenly samples from the transform's default bucket list.
        custom_buckets (dict[str, list[dict]] | None): Override the default
            buckets for specific transforms.
        scaling_low, scaling_high (float): Bounds on the per-client softmax
            peakiness (drawn uniformly per client).
        random_order (bool): Permute the pool per client so different clients
            favour different pool entries.
        window_size, stride, drop_last: Forwarded to `window_series` after the
            transforms are applied. window_size=None skips windowing.
        permute (bool): Shuffle samples before splitting into clients.
        verbose (bool): Print per-client transform assignment counts.

    Returns:
        list[dict]: One dict per client with keys `train_features`,
            `train_labels`, `test_features`, `test_labels`, `cluster`. Feature
            and label values are numpy arrays, matching the image split_fn
            output convention.
    '''
    if transforms_pool is None or len(transforms_pool) == 0:
        raise ValueError("transforms_pool must be a non-empty list of registered transform names.")
    if scaling_high < scaling_low:
        raise ValueError("scaling_high must be >= scaling_low.")
    if train_features.dim() != 3 or test_features.dim() != 3:
        raise ValueError("Features must be 3D tensors of shape (N, C, T).")
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("train_features and train_labels must have matching N.")
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("test_features and test_labels must have matching N.")

    train_clients = split_basic(train_features, train_labels, client_number, permute=permute)
    test_clients = split_basic(test_features, test_labels, client_number, permute=permute)

    rearranged = []
    for cid, (tr, te) in enumerate(zip(train_clients, test_clients)):
        pool = _build_pool(transforms_pool, magnitude_buckets, custom_buckets)
        client_scaling = float(np.random.uniform(scaling_low, scaling_high))

        len_tr = int(tr['labels'].shape[0])
        len_te = int(te['labels'].shape[0])
        total = assigning_transform_features(
            len_tr + len_te, pool, client_scaling, random_order=random_order
        )
        tr_assign = total[:len_tr]
        te_assign = total[len_tr:]

        if verbose:
            counts = Counter(name for name, _ in total)
            print(
                f"Client {cid} | scaling={client_scaling:.3f} | "
                f"transform counts: {dict(counts)}"
            )

        tr_x = _apply_per_sample_transforms(tr['features'], tr_assign)
        te_x = _apply_per_sample_transforms(te['features'], te_assign)
        tr_y = tr['labels']
        te_y = te['labels']

        if window_size is not None:
            tr_x, tr_y = window_series(tr_x, tr_y, window_size, stride=stride, drop_last=drop_last)
            te_x, te_y = window_series(te_x, te_y, window_size, stride=stride, drop_last=drop_last)

        rearranged.append({
            'train_features': tr_x.detach().cpu().numpy(),
            'train_labels': tr_y.detach().cpu().numpy(),
            'test_features': te_x.detach().cpu().numpy(),
            'test_labels': te_y.detach().cpu().numpy(),
            'cluster': -1,
        })

    return rearranged
