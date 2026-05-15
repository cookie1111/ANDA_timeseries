import torch

from .transforms_ts import apply_transform
from .windowing_ts import window_series


_VALID_STAGES = ("augment", "window")


def apply_ts_pipeline(
    features: torch.Tensor,
    labels: torch.Tensor,
    transforms: list = None,
    window_size: int = None,
    stride: int = None,
    order: tuple = ("augment", "window"),
    drop_last: bool = True,
):
    '''
    Run an optional sequence of transforms and an optional windowing stage on a
    time series batch, in a configurable order.

    Args:
        features (torch.Tensor): Tensor of shape (N, C, T).
        labels (torch.Tensor): Tensor of shape (N,).
        transforms (list[tuple[str, dict]] | None): Ordered list of
            (transform_name, kwargs) pairs to apply at the augment stage. None
            or [] skips augmentation.
        window_size (int | None): Window length for the window stage. None skips
            windowing.
        stride (int | None): Stride for windowing. Defaults to window_size.
        order (tuple[str, ...]): Sequence of stages to run. Each entry must be
            one of "augment" or "window".
        drop_last (bool): Forwarded to window_series.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Possibly-augmented and/or windowed
            (features, labels).
    '''
    for stage in order:
        if stage not in _VALID_STAGES:
            raise ValueError(
                f"Unknown pipeline stage {stage!r}. Valid stages: {_VALID_STAGES}"
            )

    for stage in order:
        if stage == "augment" and transforms:
            for entry in transforms:
                name, kwargs = entry if isinstance(entry, tuple) else (entry, {})
                features = apply_transform(features, name, kwargs)
        elif stage == "window" and window_size is not None:
            features, labels = window_series(
                features, labels, window_size, stride=stride, drop_last=drop_last
            )

    return features, labels
