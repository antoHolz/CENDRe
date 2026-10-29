import numpy as np
import torch
import torch.nn.functional as F
import omegaconf
from scipy.interpolate import interp1d, CubicSpline

class HFToTensor:
    """
    Convert the content of specified fields in a sample to torch.Tensors.
    
    The sample can be:
      - A dict: If fields is None, attempt to convert every convertible field.
                Otherwise, only convert the specified fields.
      - A list or tuple: Convert each convertible item to a torch.Tensor.
      - A single np.ndarray: Convert it to a torch.Tensor.
      - A single torch.Tensor: Returned as is.
    
    Parameters
    ----------
    fields : str, list of str, or None
        The key or keys in the sample dict whose content will be converted.
        This parameter is ignored if the sample is a list, tuple, np.ndarray, or torch.Tensor.
    """
    def __init__(self, fields=None):
        if fields is None:
            self.fields = None
        elif isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
    
    def __call__(self, sample):        
        if isinstance(sample, dict):
            if self.fields is None:
                # Convert every field that is not already a torch.Tensor and is convertible.
                for key, value in sample.items():
                    if not isinstance(value, torch.Tensor) and isinstance(value, (int, float, bool, list, tuple, np.ndarray)):
                        sample[key] = torch.tensor(value)
            else:
                # Only convert the specified fields if they exist in the sample.
                for field in self.fields:
                    if field in sample:
                        value = sample[field]
                        if not isinstance(value, torch.Tensor) and isinstance(value, (int, float, bool, list, tuple, np.ndarray)):
                            sample[field] = torch.tensor(value)
            return sample
        
        elif isinstance(sample, (list, tuple)):
            converted = []
            for item in sample:
                if not isinstance(item, torch.Tensor) and isinstance(item, (int, float, bool, list, tuple, np.ndarray)):
                    converted.append(torch.tensor(item))
                else:
                    converted.append(item)
            return tuple(converted) if isinstance(sample, tuple) else converted
        
        elif isinstance(sample, np.ndarray):
            return torch.tensor(sample)
        
        elif isinstance(sample, torch.Tensor):
            return sample
        
        else:
            raise TypeError("Expected sample to be a dict, list, tuple, np.ndarray, or torch.Tensor.")
    
class HFStack:
    """
    Stack the content from multiple fields along a new dimension and store it under a new key.

    Parameters
    ----------
    fields : list of str or str
        The keys in the sample dict to be stacked.
    dim : int
        The dimension along which to stack the tensors.
    new_name : str
        The key under which to store the stacked tensor.
    """
    def __init__(self, fields, dim: int, new_name: str):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.dim = dim
        self.new_name = new_name

    def __call__(self, sample: dict) -> dict:
        if len(self.fields) == 1:
            if self.fields[0] in sample:
                sample[self.new_name] = torch.as_tensor(sample[self.fields[0]])
                if sample[self.new_name].dim()<=1: sample[self.new_name] = sample[self.new_name].unsqueeze(self.dim)
        else:
            values = []
            for field in self.fields:
                if field in sample:
                    val = sample[field]
                    # Convert to tensor if needed.
                    if not isinstance(val, torch.Tensor):
                        val = torch.tensor(val)
                    values.append(val)
                else:
                    raise KeyError(f"Field '{field}' not found in the sample.")
            # Stack the tensors along the specified dimension.
            sample[self.new_name] = torch.stack(values, dim=self.dim)
        return sample
    
def _to_tensor1d(x):
    t = torch.as_tensor(x)      # cheaper than torch.tensor
    if t.dim() == 0:            # make scalars 1-D
        t = t.unsqueeze(0)
    return t

class HFCat:
    """
    Concatenate the content from multiple fields along the given dimension and store it under a new key.

    Parameters
    ----------
    fields : list of str or str
        The keys in the sample dict to be concatenated.
    dim : int
        The dimension along which to concatenate the tensors.
    new_name : str
        The key under which to store the concatenated tensor.
    """
    def __init__(self, fields, dim: int, new_name: str, squeeze1D=True):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.dim = dim
        self.new_name = new_name
        self.squeeze1D = squeeze1D

    def __call__(self, sample: dict) -> dict:

        # Cat the tensors along the specified dimension.
        if len(self.fields) == 1:
            f = self.fields[0]
            if f in sample:
                if self.squeeze1D:
                    sample[self.new_name] = torch.as_tensor(sample[f])
                else:
                    sample[self.new_name] = _to_tensor1d(sample[f])
            return sample

        values = []
        for field in self.fields:
            if field not in sample:
                raise KeyError(f"Field '{field}' not found in the sample.")
            values.append(_to_tensor1d(sample[field]))
        sample[self.new_name] = torch.cat(values, dim=self.dim)
        return sample
    
class HFSwapAxes:
    """
    Swap two axes in the specified fields of a sample dictionary.

    Parameters
    ----------
    fields : str or list of str
        The key or keys in the sample dict whose content will have axes swapped.
    axis1 : int
        The first axis to swap.
    axis2 : int
        The second axis to swap.
    """
    def __init__(self, fields, axis1: int, axis2: int):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.axis1 = axis1
        self.axis2 = axis2

    def __call__(self, sample: dict) -> dict:
        for field in self.fields:
            if field in sample:
                data = sample[field]
                if not isinstance(data, torch.Tensor):
                    raise TypeError(f"Field '{field}' is not a torch.Tensor (got type {type(data)}).")
                sample[field] = torch.transpose(data, self.axis1, self.axis2)
        return sample

    
class HFSampleWindow:
    """
    Sample a random window from one or more target fields in an example dictionary.
    Fields not targeted are retained (or modified in place if desired).
    
    Parameters
    ----------
    window_size : int
        The length of the window to sample.
    target_field : str or list of str
        The key (or keys) in the input example dict containing the time-series data.
    output_field : None, str, or dict, optional
        If None (default), the target field is replaced with the window.
        If a string and only one target field is provided, that string is used as the output key.
        If a dict, it should map target field names to desired output field names.
    inplace : bool, optional
        If True (default), modifies the input dictionary in place to save memory.
        If False, a shallow copy is made.
    """
    def __init__(self, window_size: int, target_field, output_field=None, inplace=True):
        self.window_size = window_size
        # Normalize target_field to a list.
        if isinstance(target_field, str):
            self.target_fields = [target_field]
        else:
            self.target_fields = list(target_field)
        self.output_field = output_field
        self.inplace = inplace

    def __call__(self, example: dict) -> dict:
        # Use the input dict directly if inplace modification is allowed.
        out = example if self.inplace else example.copy()
        
        for field in self.target_fields:
            if field in example:
                series = example[field]
                # If the series is shorter than the desired window size, use the whole series.
                if len(series) < self.window_size:
                    window = series
                else:
                    # Compute the valid range for the starting index.
                    if isinstance(series, torch.Tensor):
                        # Use torch.Tensor.size(0) for tensor length and narrow to get a view.
                        max_start = series.size(0) - self.window_size + 1
                        start_idx = torch.randint(0, max_start, (1,)).item()
                        window = series.narrow(0, start_idx, self.window_size)
                    else:
                        # Assume series supports len() and slicing (e.g. list, tuple, or numpy array)
                        max_start = len(series) - self.window_size + 1
                        start_idx = torch.randint(0, max_start, (1,)).item()
                        window = series[start_idx: start_idx + self.window_size]
                        
                # Determine where to store the result.
                if self.output_field is not None:
                    if isinstance(self.output_field, dict):
                        out[self.output_field.get(field, field)] = window
                    else:
                        # When a single output field is provided and only one target field exists.
                        out[self.output_field] = window
                else:
                    out[field] = window
                    
        return out
class HFSplitWindow:
    """
    Split an already-windowed field (or fields) into two parts.

    For each target field, assume the value is a sequence (e.g. list, numpy array, or torch.Tensor)
    of length (seq_len + pred_len) (or longer). The split is done as follows:
    
      - Part 1: the first `seq_len` elements, i.e. window[0:seq_len]
      - Part 2: the remaining elements starting from index (label_len + pred_len),
                where label_len is optional. If label_len is None, it is set to
                (seq_len - pred_len), so that (label_len + pred_len) equals seq_len,
                and Part 2 becomes window[seq_len:].
                
    This transform is applied independently to each target field and can store the split
    tuple (part1, part2) either in place or under new keys specified by `output_field`.

    Parameters
    ----------
    seq_len : int
        The length of the first part of the window.
    pred_len : int
        The length used to determine the start of the second part.
    label_len : int or None, optional
        An optional offset for the split. If None (default), it is set to seq_len - pred_len.
    target_field : str or list of str
        The key (or keys) in the input dictionary that contain the windowed data.
    output_field : None, str, or dict, optional
        If None (default), the target field is replaced with the tuple (part1, part2).
        If a string (and only one target field is provided), that string is used as the output key.
        If a dict, it should map target field names to desired output field names.
    inplace : bool, optional
        If True (default), the input dictionary is modified in place.
    """
    
    def __init__(self, seq_len: int, pred_len: int, target_field, output_field=None, inplace=True, label_len: int = None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        # If label_len is not provided, default to seq_len - pred_len.
        self.label_len = seq_len - pred_len if label_len is None else label_len
        
        if isinstance(target_field, str):
            self.target_fields = [target_field]
        else:
            self.target_fields = list(target_field)
            
        self.output_field = output_field
        self.inplace = inplace

    def __call__(self, example: dict) -> dict:
        # Use the input dictionary directly if modifying in place.
        out = example if self.inplace else example.copy()
        
        for field in self.target_fields:
            if field not in example:
                continue  # Skip missing fields.
                
            window = example[field]
            
            # Split the window:
            # Part 1: the first seq_len elements.
            part1 = window[0:self.seq_len]
            # Part 2: elements starting from index (label_len + pred_len) to the end.
            part2 = window[-(self.label_len + self.pred_len):]
            
            # Store the result according to output_field.
            if self.output_field is not None:
                if isinstance(self.output_field, dict):
                    out[self.output_field.get(field, field)] = part1
                    out[self.output_field.get(field, field)+"_pred"] = part2
                else:
                    out[self.output_field] = part1
                    out[self.output_field+"_pred"] = part2
            else:
                out[field] = part1
                out[field+"_pred"] = part2
                
        return out

class HFRename:
    """
    Rename fields in a sample dictionary.

    Parameters
    ----------
    fields : dict
        A mapping from the old field names to the new field names.
    """
    def __init__(self, fields: dict):
        self.fields = fields

    def __call__(self, sample: dict) -> dict:
        for old_key, new_key in self.fields.items():
            if old_key in sample:
                sample[new_key] = sample.pop(old_key)
        return sample

class HFSelect:
    """
    Select and output specific fields from a sample dictionary.

    If a dictionary is passed for `fields`, the transform will return a dictionary
    where each key is the new field name (the corresponding dictionary value) and the
    value is taken from the sample using the original field name (the dictionary key).

    Parameters
    ----------
    fields : str, list of str, or dict
        - If a string or list is provided, these keys are extracted from the sample.
        - If a dict is provided, it maps old field names to new field names.
    output_type : str, optional (default: "dict")
        Determines the format of the output if `fields` is a string or list.
        If "tuple", returns a tuple of values; if "dict", returns a dictionary
        mapping field names to values. (Ignored if `fields` is a dict.)
    """
    def __init__(self, fields, output_type="dict"):
        if isinstance(fields, (dict, omegaconf.dictconfig.DictConfig)):
            self.fields_mapping = fields
            self.fields = None
        else:
            self.fields_mapping = None
            if isinstance(fields, str):
                self.fields = [fields]
            else:
                self.fields = list(fields)
        if output_type not in ("tuple", "dict"):
            raise ValueError("output_type must be 'tuple' or 'dict'.")
        self.output_type = output_type

    def __call__(self, sample: dict):
        if self.fields_mapping is not None:
            # Rename selected fields according to the provided mapping.
            return {new_field: sample[old_field] for old_field, new_field in self.fields_mapping.items()}
        else:
            if self.output_type == "tuple":
                return tuple(sample[field] for field in self.fields)
            else:  # output_type == "dict"
                return {field: sample[field] for field in self.fields}

class HFDrop:
    """
    Drop (delete) specified fields from a sample dictionary.

    Parameters
    ----------
    fields : str or list of str
        The key or keys in the sample dictionary that should be removed.
    """
    def __init__(self, fields):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)

    def __call__(self, sample: dict) -> dict:
        for field in self.fields:
            if field in sample:
                sample.pop(field)
        return sample

class HFResize:
    """
    Resize selected tensor fields to a specified length via interpolation.

    For each specified field (expected to be a 1D torch.Tensor):
      - If the tensor length differs from target_length, it is resized using linear interpolation.
      - Before interpolation, the tensor is converted to float; after interpolation, it is cast back to its original dtype.

    Parameters
    ----------
    fields : str or list of str
        The key(s) in the sample dictionary to be resized.
    target_length : int
        The desired length for the tensor.
    mode : str, optional
        The interpolation mode (default: 'linear').
    align_corners : bool, optional
        Whether to align corners in interpolation (default: True).
    """
    def __init__(self, fields, target_length, mode='linear', align_corners=True):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.target_length = target_length
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, sample: dict) -> dict:
        for field in self.fields:
            if field not in sample:
                continue  # Optionally, raise an error if the field is missing.
            x = sample[field]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Field '{field}' is not a torch.Tensor.")
            # Assume x is 1D (shape [L])
            L = x.shape[0]
            if L == self.target_length:
                continue  # No resizing needed.
            # Save original dtype and convert to float for interpolation.
            original_dtype = x.dtype
            xdim = x.dim()
            x = x.float()
            # Reshape to [1, 1, L] for F.interpolate.
            for i in range(3-xdim):
                x = x.unsqueeze(0)
            # Interpolate to the target length.
            x = F.interpolate(x, size=self.target_length, mode=self.mode, align_corners=self.align_corners)
            # Squeeze back to original dim.
            for i in range(3-xdim):
                x = x.squeeze(0)
            # Convert back to original dtype.
            sample[field] = x.to(original_dtype)
        return sample


class HFNormalize:
    """
    Normalize the specified fields in a sample dictionary using torch.nn.functional.normalize.

    For each specified field, if the value is a torch.Tensor, the tensor is normalized along the given dimension.
    
    Parameters
    ----------
    fields : str or list of str
        The key or keys in the sample dictionary whose content will be normalized.
    dim : int
        The dimension along which to normalize.
    p : float, optional
        The exponent value in the norm formulation (default is 2.0 for L2 normalization).
    eps : float, optional
        A small value to avoid division by zero (default is 1e-12).
    """
    def __init__(self, fields, dim, p=2.0, eps=1e-12):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.dim = dim
        self.p = p
        self.eps = eps

    def __call__(self, sample: dict) -> dict:
        for field in self.fields:
            if field in sample:
                data = sample[field]
                if not isinstance(data, torch.Tensor):
                    raise TypeError(f"Field '{field}' is not a torch.Tensor (got type {type(data)}).")
                # Normalize the tensor along the specified dimension.
                sample[field] = F.normalize(data, p=self.p, dim=self.dim, eps=self.eps)
        return sample

class HFStandardize:
    """
    Standardize the specified fields in a sample dictionary using z-score normalization.

    For each specified field (which must be a torch.Tensor), the transform computes the mean and
    standard deviation along the given dimension, then subtracts the mean and divides by the standard
    deviation (with a small epsilon added for numerical stability).

    Parameters
    ----------
    fields : str or list of str
        The key or keys in the sample dictionary whose content will be standardized.
    dim : int
        The dimension along which to compute the mean and standard deviation.
    eps : float, optional
        A small constant added to the standard deviation to avoid division by zero (default is 1e-12).
    """
    def __init__(self, fields, dim=-1, eps=1e-12):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.dim = dim
        self.eps = eps

    def __call__(self, sample: dict) -> dict:
        for field in self.fields:
            if field in sample:
                data = sample[field]
                if not isinstance(data, torch.Tensor):
                    raise TypeError(f"Field '{field}' is not a torch.Tensor (got type {type(data)}).")
                # Compute mean and standard deviation along the specified dimension.
                mean = data.mean(dim=self.dim, keepdim=True)
                std = data.std(dim=self.dim, keepdim=True)
                # Standardize: (x - mean) / (std + eps)
                sample[field] = (data - mean) / (std + self.eps)
        return sample
    
class HFStandardize2:
    """
    Standardize the specified fields in a sample dictionary using z-score normalization.

    This transform supports using precomputed mean and std values (e.g., from training data).
    If no mean or std is provided, they are computed from the input dynamically.

    Parameters
    ----------
    fields : str or list of str
        Keys in the sample dictionary to be standardized.
    mean : dict or None
        Optional dictionary mapping field names to precomputed mean tensors.
    std : dict or None
        Optional dictionary mapping field names to precomputed std tensors.
    dim : int
        Dimension along which to compute mean and std.
    eps : float
        Small constant to avoid division by zero.
    """

    def __init__(self, fields_x=[], fields_y=[], mean=None, std=None, dim=-1, eps=1e-8, mode='input'):
        if isinstance(fields_x, str):
            fields_x = [fields_x]
        if isinstance(fields_y, str):
            fields_y = [fields_y]

        self.fields_x = list(fields_x)
        self.fields_y = list(fields_y)
        self.mean = mean if mean is not None else {}
        self.std = std if std is not None else {}
        self.dim = dim
        self.eps = eps
        self.mode = mode

    def __call__(self, sample: dict) -> dict:
        if self.fields_x:
            for channel, field in enumerate(self.fields_x):
                data = sample["x"][channel]
                if not isinstance(data, torch.Tensor):
                    raise TypeError(f"Field '{field}' must be a torch.Tensor (got {type(data)})")
                # Use precomputed mean/std if available, otherwise compute dynamically
                if field in self.mean:
                    mean = self.mean[field]
                else:
                    mean = data.mean(dim=self.dim, keepdim=True)
                if field in self.std:
                    std = self.std[field]           
                else:
                    std = data.std(dim=self.dim, keepdim=True)
                # mean = self.mean.get(field, data.mean(dim=self.dim, keepdim=True))
                # std = self.std.get(field, data.std(dim=self.dim, keepdim=True))
                sample["x"][channel] = (data - mean) / (std + self.eps)
        if self.fields_y:
            y = sample["y"]
            if not isinstance(y, torch.Tensor):
                raise TypeError(f"'y' must be a torch.Tensor (got {type(y)})")

            # Ensure y is at least 1D: scalar -> [1]
            if y.dim() == 0:
                y = y.unsqueeze(0)

            for channel, field in enumerate(self.fields_y):
                data = y[channel]  # scalar or vector depending on y
                if field in self.mean and field in self.std:
                    mean = self.mean[field]
                    std = self.std[field]
                else:
                    raise ValueError(
                        f"Field '{field}' is missing precomputed mean or std."
                    )
                y[channel] = (data - mean) / (std + self.eps)

            sample["y"] = y

        return sample
    
    def __repr__(self):
        field_str = ", ".join(self.fields)
        return (f"{self.__class__.__name__}(fields=[{field_str}], "
                f"dim={self.dim}, eps={self.eps}, "
                f"mean={'provided' if self.mean else 'none'}, "
                f"std={'provided' if self.std else 'none'})")
    
class HFUnsqueeze:
    """
    Unsqueeze the specified fields in a sample dictionary along a given dimension.

    Parameters
    ----------
    fields : str, list of str, or None
        The key or keys in the sample dictionary whose content will be unsqueezed.
    dim : int, optional
        The dimension along which to unsqueeze the tensor. Default is -1.
    """
    def __init__(self, fields, dim: int = -1):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.dim = dim

    def __call__(self, sample: dict) -> dict:
        for field in self.fields:
            if field in sample:
                data = sample[field]
                if isinstance(data, torch.Tensor):
                    sample[field] = data.unsqueeze(self.dim)
                else:
                    raise TypeError(
                        f"Field '{field}' is not a torch.Tensor (got type {type(data)})."
                    )
        return sample


class HFAddGaussianNoise:
    """
    Add Gaussian noise to the specified fields in a sample dictionary.

    For each specified field, if the value is a torch.Tensor, Gaussian noise
    (with the specified mean and standard deviation) is added element-wise.

    Parameters
    ----------
    fields : str, list of str, or None
        The key or keys in the sample dict to which the Gaussian noise will be added.
        If None, noise is added to every field in the sample that is a torch.Tensor.
    mean : float, optional
        The mean of the Gaussian noise (default is 0.0).
    std : float, optional
        The standard deviation of the Gaussian noise (default is 0.1).
    """
    def __init__(self, fields=None, mean=0.0, std=0.1):
        if fields is None:
            self.fields = None
        elif isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = list(fields)
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        if not isinstance(sample, dict):
            raise TypeError("Expected sample to be a dict.")

        # If no specific fields are provided, add noise to every tensor field.
        if self.fields is None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    noise = torch.randn_like(value) * self.std + self.mean
                    sample[key] = value + noise
        else:
            for field in self.fields:
                if field in sample:
                    value = sample[field]
                    if isinstance(value, torch.Tensor):
                        noise = torch.randn_like(value) * self.std + self.mean
                        sample[field] = value + noise
                    else:
                        raise TypeError(
                            f"Field '{field}' is not a torch.Tensor (got type {type(value)})."
                        )
        return sample



class HFIdentity:
    """
    Convert the content of specified fields in a sample to torch.Tensors.
    
    The sample can be:
      - A dict: If fields is None, attempt to convert every convertible field.
                Otherwise, only convert the specified fields.
      - A list or tuple: Convert each convertible item to a torch.Tensor.
      - A single np.ndarray: Convert it to a torch.Tensor.
      - A single torch.Tensor: Returned as is.
    
    """
    def __call__(self, sample):        
        
        return sample
        

################################################################################################################


class HFRandomCropPad:
    def __init__(self, fields, magnitude: float = 0.1, p: float = 1.0):
        self.fields = [fields] if isinstance(fields, str) else fields
        self.magnitude = magnitude
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() > self.p or self.magnitude <= 0:
            return sample

        for field in self.fields:
            x = sample[field]
            L = x.shape[-1]
            if L < 2:
                continue
            rand = np.random.beta(self.magnitude, self.magnitude)
            lambd = max(rand, 1 - rand)
            win_len = int(round(L * lambd))
            if win_len >= L:
                continue
            start = np.random.randint(0, L - win_len)
            out = torch.zeros_like(x)
            out[..., start:start+win_len] = x[..., start:start+win_len]
            sample[field] = out
        return sample


def random_cum_curve_generator(seq_len: int, magnitude: float = 0.1, order: int = 4):
    """Generates a monotonically increasing curve (cumulative) used for time warping.

    We sample 3*(order-1)+1 random points from a normal distribution, then 
    build a spline and evaluate it at regular intervals from 0 to seq_len-1. 
    Finally, the curve is normalized to the range [0, seq_len-1].

    Args:
        seq_len (int): Length of the time dimension (L).
        magnitude (float): Controls how strongly to warp.
        order (int): Polynomial spline order (typically 3 or 4).

    Returns:
        np.ndarray: 1D array of shape [L], representing the warped time indices.
    """
    # 1) Sample random points for the spline
    random_points = np.random.normal(loc=1.0, scale=magnitude, size=3 * (order - 1) + 1)
    x_vals = np.linspace(-seq_len, 2 * seq_len - 1, len(random_points))

    # 2) Build the spline
    f = CubicSpline(x_vals, random_points, axis=-1)

    # 3) Evaluate it on [0, seq_len-1], then normalize to [0, seq_len-1]
    warped = f(np.arange(seq_len))
    warped -= warped[0]
    warped /= warped[-1]
    warped *= (seq_len - 1)
    return warped

    
class HFTimeWarp:
    """
    Applies time warping by resampling the time axis with a warped curve 
    using spline interpolation. Does NOT keep channel alignment

    Parameters
    ----------
    fields : str or list of str
        Keys of tensors to transform.
    sigma : float
        Standard deviation of the time warp curve.
    knot : int
        Number of control points in the warp spline.
    p : float
        Probability of applying the transform.
    """
    def __init__(self, fields, sigma: float = 0.2, knot: int = 4, p: float = 1.0):
        self.fields = [fields] if isinstance(fields, str) else fields
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() > self.p or self.sigma == 0:
            return sample

        for field in self.fields:
            x = sample[field]
            if isinstance(x, torch.Tensor):
                was_torch = True
                x_np = x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                was_torch = False
                x_np = x
            else:
                continue

            # Assume shape [C, L]
            if x_np.ndim != 2:
                continue
            C, L = x_np.shape

            orig_steps = np.arange(L)
            warp_steps = np.linspace(0, L - 1, num=self.knot + 2)
            warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, C))

            warped = np.zeros_like(x_np)
            for c in range(C):
                warper = CubicSpline(warp_steps, warp_steps * warps[:, c])
                time_warp = warper(orig_steps)
                scale = (L - 1) / time_warp[-1] if time_warp[-1] != 0 else 1.0
                warped[c, :] = np.interp(orig_steps, np.clip(scale * time_warp, 0, L - 1), x_np[c, :])

            if was_torch:
                warped = torch.tensor(warped, dtype=x.dtype, device=x.device)
            elif not was_torch:
                warped = warped.astype(x.dtype)
            sample[field] = warped

        return sample


class HFJitter:
    def __init__(self, fields, sigma: float = 0.03, p: float = 1.0):
        self.fields = [fields] if isinstance(fields, str) else fields
        self.sigma = sigma
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() > self.p or self.sigma <= 0:
            return sample
        for field in self.fields:
            x = sample[field]
            if isinstance(x, torch.Tensor):
                noise = torch.randn_like(x) * self.sigma
                sample[field] = x + noise
        return sample


class HFMagnitudeWarp:
    """
    Applies a smooth random warping to the magnitude of each dimension over time
    for the specified fields in a sample dictionary.

    Parameters
    ----------
    fields : str or list of str
        The keys of the tensors to transform.
    sigma : float
        Standard deviation for the warping factor.
    knot : int
        Number of spline knots (defines warp smoothness).
    p : float
        Probability of applying the transform.
    """
    def __init__(self, fields, sigma: float = 0.2, knot: int = 4, p: float = 1.0):
        self.fields = [fields] if isinstance(fields, str) else fields
        self.sigma = sigma
        self.knot = knot
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if torch.rand(1).item() > self.p or self.sigma == 0:
            return sample

        for field in self.fields:
            x = sample[field]
            if isinstance(x, torch.Tensor):
                was_torch = True
                x_np = x.detach().cpu().numpy()
            elif isinstance(x, np.ndarray):
                was_torch = False
                x_np = x
            else:
                continue

            # Assume shape [L, C] or [C, L]
            if x_np.ndim != 2:
                continue
            L, C = x_np.shape if x_np.shape[0] > x_np.shape[1] else (x_np.shape[1], x_np.shape[0])
            x_np = x_np if x_np.shape[0] == L else x_np.T

            orig_steps = np.arange(L)
            warp_steps = np.linspace(0, L - 1, num=self.knot + 2)
            warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot + 2, C))

            warped = np.zeros_like(x_np)
            for c in range(C):
                spline = CubicSpline(warp_steps, warps[:, c])
                warp_curve = spline(orig_steps)
                warped[:, c] = x_np[:, c] * warp_curve

            if was_torch:
                warped = torch.tensor(warped, dtype=x.dtype, device=x.device)
            elif not was_torch:
                warped = warped.astype(x.dtype)
            sample[field] = warped.T if x.shape != warped.shape else warped

        return sample




################################################################################################################

### Helper func ###
