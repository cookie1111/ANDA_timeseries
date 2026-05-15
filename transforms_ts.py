import numpy as np
import torch

TRANSFORMS = {}
TRANSFORM_BUCKETS = {}


def register_transform(name: str, buckets: list = None):
    '''
    Decorator that registers a time series augmentation under a string name and
    optionally declares its default magnitude buckets.

    Args:
        name (str): The string key the transform will be referenced by in the
            registry and in split_fn_ts pool specs.
        buckets (list[dict], optional): Ordered list of kwargs presets, typically
            low -> medium -> high magnitude. Used by split_feature_skew_ts to
            build a discrete (transform, bucket) softmax pool. If omitted, the
            transform is registered without bucket presets and callers must
            supply kwargs explicitly.

    Returns:
        Callable: Decorator that stores the function and returns it unchanged.
    '''
    def decorator(fn):
        TRANSFORMS[name] = fn
        if buckets is not None:
            TRANSFORM_BUCKETS[name] = [dict(b) for b in buckets]
        return fn
    return decorator


def list_transforms() -> list:
    '''Return the list of currently registered transform names.'''
    return sorted(TRANSFORMS)


def apply_transform(x: torch.Tensor, name: str, kwargs: dict = None) -> torch.Tensor:
    '''
    Look up a registered transform by name and run it on x.

    Args:
        x (torch.Tensor): Input batch of shape (N, C, T).
        name (str): Registered transform name.
        kwargs (dict, optional): Keyword arguments forwarded to the transform.

    Returns:
        torch.Tensor: Transformed batch, same shape (N, C, T).
    '''
    if name not in TRANSFORMS:
        raise KeyError(
            f"Unknown transform {name!r}. Known: {list_transforms()}"
        )
    return TRANSFORMS[name](x, **(kwargs or {}))


def _validate_shape(x: torch.Tensor) -> None:
    if x.dim() != 3:
        raise ValueError(
            f"Time series transforms expect tensors of shape (N, C, T); got {tuple(x.shape)}"
        )


@register_transform(
    "gaussian_noise",
    buckets=[{"sigma": 0.01}, {"sigma": 0.05}, {"sigma": 0.15}],
)
def gaussian_noise(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    '''Add iid Gaussian noise N(0, sigma) to every time step of every series.'''
    _validate_shape(x)
    noise = np.random.standard_normal(tuple(x.shape)) * sigma
    return x + torch.as_tensor(noise, dtype=x.dtype, device=x.device)


@register_transform(
    "mean_offset",
    buckets=[{"offset_std": 0.1}, {"offset_std": 0.5}, {"offset_std": 1.0}],
)
def mean_offset(x: torch.Tensor, offset_std: float = 0.5) -> torch.Tensor:
    '''Shift each series by a per-(sample,channel) constant drawn from N(0, offset_std).'''
    _validate_shape(x)
    N, C, _ = x.shape
    offsets = np.random.standard_normal((N, C, 1)) * offset_std
    return x + torch.as_tensor(offsets, dtype=x.dtype, device=x.device)


@register_transform(
    "amplitude_scaling",
    buckets=[{"scale_std": 0.05}, {"scale_std": 0.2}, {"scale_std": 0.5}],
)
def amplitude_scaling(x: torch.Tensor, scale_std: float = 0.2) -> torch.Tensor:
    '''Multiply each series by a per-(sample,channel) scalar drawn from N(1, scale_std).'''
    _validate_shape(x)
    N, C, _ = x.shape
    scales = 1.0 + np.random.standard_normal((N, C, 1)) * scale_std
    return x * torch.as_tensor(scales, dtype=x.dtype, device=x.device)


@register_transform(
    "jitter",
    buckets=[{"sigma": 0.005}, {"sigma": 0.02}, {"sigma": 0.05}],
)
def jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    '''Small-magnitude per-step Gaussian noise. Convention from the TS aug literature.'''
    return gaussian_noise(x, sigma=sigma)


def _cubic_spline_curve(knot_steps, knot_values, eval_steps):
    try:
        from scipy.interpolate import CubicSpline
    except ImportError as e:
        raise ImportError(
            "magnitude_warp and time_warp require scipy. Install with `pip install scipy`."
        ) from e
    return CubicSpline(knot_steps, knot_values)(eval_steps)


@register_transform(
    "magnitude_warp",
    buckets=[
        {"sigma": 0.1, "knot": 4},
        {"sigma": 0.2, "knot": 4},
        {"sigma": 0.4, "knot": 4},
    ],
)
def magnitude_warp(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
    '''Multiply each (sample, channel) series by a smooth random curve (cubic spline
    through `knot`+2 control points drawn from N(1, sigma)).'''
    _validate_shape(x)
    N, C, T = x.shape
    eval_steps = np.arange(T)
    knot_steps = np.linspace(0, T - 1, knot + 2)
    arr = x.detach().cpu().numpy()
    out = np.empty_like(arr)
    for i in range(N):
        for c in range(C):
            controls = np.random.standard_normal(knot + 2) * sigma + 1.0
            curve = _cubic_spline_curve(knot_steps, controls, eval_steps)
            out[i, c] = arr[i, c] * curve
    return torch.as_tensor(out, dtype=x.dtype, device=x.device)


@register_transform(
    "time_warp",
    buckets=[
        {"sigma": 0.1, "knot": 4},
        {"sigma": 0.2, "knot": 4},
        {"sigma": 0.4, "knot": 4},
    ],
)
def time_warp(x: torch.Tensor, sigma: float = 0.2, knot: int = 4) -> torch.Tensor:
    '''Warp the time axis non-uniformly using a smooth random curve, then resample.'''
    _validate_shape(x)
    N, C, T = x.shape
    eval_steps = np.arange(T)
    knot_steps = np.linspace(0, T - 1, knot + 2)
    arr = x.detach().cpu().numpy()
    out = np.empty_like(arr)
    for i in range(N):
        controls = np.random.standard_normal(knot + 2) * sigma + 1.0
        curve = _cubic_spline_curve(knot_steps, controls, eval_steps)
        warped_steps = np.cumsum(np.clip(curve, 1e-3, None))
        warped_steps = warped_steps * ((T - 1) / warped_steps[-1])
        for c in range(C):
            out[i, c] = np.interp(eval_steps, warped_steps, arr[i, c])
    return torch.as_tensor(out, dtype=x.dtype, device=x.device)


@register_transform(
    "window_slice",
    buckets=[
        {"reduce_ratio": 0.95},
        {"reduce_ratio": 0.9},
        {"reduce_ratio": 0.8},
    ],
)
def window_slice(x: torch.Tensor, reduce_ratio: float = 0.9) -> torch.Tensor:
    '''Take a random contiguous sub-window of length `reduce_ratio`*T per sample
    and linearly resample it back to T.'''
    _validate_shape(x)
    if not 0 < reduce_ratio <= 1.0:
        raise ValueError("reduce_ratio must be in (0, 1].")
    N, C, T = x.shape
    target = max(2, int(round(T * reduce_ratio)))
    if target >= T:
        return x.clone()
    eval_steps = np.arange(T)
    sub_steps = np.linspace(0, T - 1, target)
    arr = x.detach().cpu().numpy()
    out = np.empty_like(arr)
    for i in range(N):
        start = int(np.random.randint(0, T - target + 1))
        for c in range(C):
            sub = arr[i, c, start:start + target]
            out[i, c] = np.interp(eval_steps, sub_steps, sub)
    return torch.as_tensor(out, dtype=x.dtype, device=x.device)


@register_transform(
    "permutation",
    buckets=[
        {"max_segments": 3},
        {"max_segments": 5},
        {"max_segments": 8},
    ],
)
def permutation(x: torch.Tensor, max_segments: int = 5) -> torch.Tensor:
    '''Split each series into [2, max_segments] roughly equal contiguous segments
    and reorder them.'''
    _validate_shape(x)
    if max_segments < 2:
        raise ValueError("max_segments must be >= 2.")
    N, C, T = x.shape
    arr = x.detach().cpu().numpy()
    out = np.empty_like(arr)
    for i in range(N):
        num_segs = int(np.random.randint(2, max_segments + 1))
        seg_size = max(1, T // num_segs)
        boundaries = list(range(0, num_segs * seg_size, seg_size)) + [T]
        boundaries = boundaries[: num_segs + 1]
        boundaries[-1] = T
        order = np.random.permutation(num_segs)
        pos = 0
        for s in order:
            seg = arr[i, :, boundaries[s]:boundaries[s + 1]]
            seg_len = seg.shape[-1]
            out[i, :, pos:pos + seg_len] = seg
            pos += seg_len
    return torch.as_tensor(out, dtype=x.dtype, device=x.device)
