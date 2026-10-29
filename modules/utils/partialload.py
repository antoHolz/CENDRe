import torch

def load_partial_state_dict(model, checkpoint_path):
    """
    Load a partial state_dict into a model, loading only the matching layers based on names,
    and return the mismatches over union for layers and parameters.

    Parameters:
    - model: The model into which to load the weights.
    - checkpoint_path: Path to the checkpoint containing the state_dict.

    Returns:
    - mismatch_over_union_layers: Ratio of mismatched layers over the union of layers.
    - mismatch_over_union_params: Ratio of mismatched parameters over the union of parameters.
    """
    # Load the saved state_dict
    saved_state_dict = torch.load(checkpoint_path)
    # Get the model's current state_dict
    model_state_dict = model.state_dict()

    # Get sets of parameter names
    model_param_names = set(model_state_dict.keys())
    saved_param_names = set(saved_state_dict.keys())

    # Union and intersection of parameter names
    union_names = model_param_names.union(saved_param_names)
    intersection_names = model_param_names.intersection(saved_param_names)

    # Initialize counts
    total_union_layers = len(union_names)
    total_union_params = 0
    mismatched_layers_count = 0
    mismatched_params_count = 0

    # Layers with size mismatches or missing in one of the state_dicts
    mismatched_layers = []

    # Calculate total parameters in union
    for name in union_names:
        if name in model_state_dict:
            total_union_params += model_state_dict[name].numel()
        elif name in saved_state_dict:
            total_union_params += saved_state_dict[name].numel()

    # Identify mismatched layers and parameters
    for name in union_names:
        if name in model_state_dict and name in saved_state_dict:
            model_param = model_state_dict[name]
            saved_param = saved_state_dict[name]
            if model_param.size() == saved_param.size():
                # Sizes match, copy the parameter
                model_param.copy_(saved_param)
            else:
                # Size mismatch
                mismatched_layers.append(name)
                mismatched_layers_count += 1
                mismatched_params_count += model_param.numel()
        else:
            # Layer missing in one of the state_dicts
            mismatched_layers.append(name)
            mismatched_layers_count += 1
            if name in model_state_dict:
                mismatched_params_count += model_state_dict[name].numel()
            elif name in saved_state_dict:
                mismatched_params_count += saved_state_dict[name].numel()

    # Optionally, log mismatched layers
    if mismatched_layers:
        for layer_name in mismatched_layers:
            print(f" - {layer_name}")

    # Calculate mismatches over union
    mismatch_over_union_layers = mismatched_layers_count / total_union_layers if total_union_layers > 0 else 0
    mismatch_over_union_params = mismatched_params_count / total_union_params if total_union_params > 0 else 0

    return mismatch_over_union_layers, mismatch_over_union_params
