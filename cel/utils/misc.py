from collections.abc import Mapping 

def get_x(sample):
    """
    Return the actual image / feature tensor from a single dataset sample.

    Works for:
    • Tuple-style datasets that return (x, y, …)
    • Dict-style datasets that return {"x": x, …}
    • Namedtuples / dataclasses that expose an `.x` attribute
    """
    if isinstance(sample, Mapping):          # dict or similar
        return sample["x"]
    if hasattr(sample, "x"):                 # namedtuple, dataclass
        return sample.x
    # fallback: old tuple/list convention
    return sample[0]

def infer_shape(dataloader, device=None):
    x = get_x(dataloader.dataset[0])
    if device is not None:
        x = x.to(device)
    return x.shape #, x.shape[0], x.shape[1:] 