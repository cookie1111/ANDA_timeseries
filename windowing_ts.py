import torch


def window_series(
    features: torch.Tensor,
    labels: torch.Tensor,
    window_size: int,
    stride: int = None,
    drop_last: bool = True,
):
    '''
    Slice each series along the time axis into fixed-length windows. Each window
    inherits the parent series' label.

    Args:
        features (torch.Tensor): Tensor of shape (N, C, T).
        labels (torch.Tensor): Tensor of shape (N,).
        window_size (int): Length of each window along the time axis.
        stride (int, optional): Step between consecutive window starts. Defaults
            to `window_size` (non-overlapping windows).
        drop_last (bool): If True, any trailing partial window is dropped. If
            False and (T - window_size) % stride != 0, one extra window aligned
            to the end of the series is appended per sample.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            features of shape (N * W, C, window_size), labels of shape (N * W,),
            where W is the per-series window count after the drop_last policy.
    '''
    if features.dim() != 3:
        raise ValueError(
            f"window_series expects features of shape (N, C, T); got {tuple(features.shape)}"
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same number of samples.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    N, C, T = features.shape
    if window_size > T:
        raise ValueError(
            f"window_size ({window_size}) is larger than series length ({T})."
        )
    if window_size == T:
        return features.clone(), labels.clone()

    if stride is None:
        stride = window_size
    if stride <= 0:
        raise ValueError("stride must be positive.")

    full_W = (T - window_size) // stride + 1
    windows = features.unfold(2, window_size, stride)  # (N, C, full_W, window_size)
    windows = windows.permute(0, 2, 1, 3).contiguous().reshape(N * full_W, C, window_size)
    out_labels = labels.repeat_interleave(full_W)

    if not drop_last and (T - window_size) % stride != 0:
        tail = features[:, :, T - window_size:T].contiguous()  # (N, C, window_size)
        windows = torch.cat([windows.reshape(N, full_W, C, window_size), tail.unsqueeze(1)], dim=1)
        windows = windows.reshape(N * (full_W + 1), C, window_size)
        out_labels = labels.repeat_interleave(full_W + 1)

    return windows, out_labels
