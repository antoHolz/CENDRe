import torch

def compute_training_metrics(pl_module, layerwise=True, log_histograms=True):
    """
    Computes and logs various training metrics for monitoring model training.

    Metrics computed and logged:
    - Gradient and Weight Mean
    - Gradient and Weight Standard Deviation (std)
    - Gradient and Weight Norms (L2 Norm)
    - Gradient and Weight Sparsity
    - Gradient and Weight Entropy (computed from std)
    - Gradient and Weight Histograms (if log_histograms is True)

    Args:
        pl_module (pl.LightningModule): The LightningModule being trained.
        layerwise (bool): If True, compute and log metrics per layer as well.
        log_histograms (bool): If True, compute and log histograms.
    """
    # Initialize containers for model-wise metrics
    grad_values = []
    param_values = []
    grad_norms = []
    param_norms = []

    # For layer-wise metrics
    if layerwise:
        grad_metrics_per_layer = {}
        param_metrics_per_layer = {}

    for name, p in pl_module.named_parameters():
        # --------------------------
        # Metrics for Weights (Parameters)
        # --------------------------
        param = p.data.detach()
        param_flat = param.view(-1)
        param_values.append(param_flat)

        param_elements = param_flat.numel()
        zero_param_elements = (param_flat == 0).sum()

        param_mean = param_flat.mean()
        param_std = param_flat.std()
        param_sparsity = zero_param_elements.float() / param_elements if param_elements > 0 else torch.tensor(0.0, device=param.device)
        param_norm = torch.norm(param, p=2)  # L2 Norm
        param_norms.append(param_norm)

        # Compute entropy for parameters using std deviation
        param_variance = param_std ** 2
        param_entropy = 0.5 * (torch.log(2 * torch.pi * param_variance + 1e-12) + 1)

        if layerwise:
            param_metrics_per_layer[name] = {
                'mean': param_mean,
                'std': param_std,
                'sparsity': param_sparsity,
                'entropy': param_entropy,
                'norm': param_norm
            }

            # Log parameter histograms per layer if logging is enabled
            if log_histograms and pl_module.logger is not None and hasattr(pl_module.logger.experiment, 'add_histogram'):
                pl_module.logger.experiment.add_histogram(f'parameters/{name}', param, pl_module.global_step)

        # --------------------------
        # Metrics for Gradients
        # --------------------------
        if p.grad is not None:
            grad = p.grad.detach()
            grad_flat = grad.view(-1)
            grad_values.append(grad_flat)

            grad_elements = grad_flat.numel()
            zero_grad_elements = (grad_flat == 0).sum()

            grad_mean = grad_flat.mean()
            grad_std = grad_flat.std()
            grad_sparsity = zero_grad_elements.float() / grad_elements if grad_elements > 0 else torch.tensor(0.0, device=grad.device)
            grad_norm = torch.norm(grad, p=2)  # L2 Norm
            grad_norms.append(grad_norm)

            # Compute entropy for gradients using std deviation
            grad_variance = grad_std ** 2
            grad_entropy = 0.5 * (torch.log(2 * torch.pi * grad_variance + 1e-12) + 1)

            if layerwise:
                grad_metrics_per_layer[name] = {
                    'mean': grad_mean,
                    'std': grad_std,
                    'sparsity': grad_sparsity,
                    'entropy': grad_entropy,
                    'norm': grad_norm
                }

                # Log gradient histograms per layer if logging is enabled
                if log_histograms and pl_module.logger is not None and hasattr(pl_module.logger.experiment, 'add_histogram'):
                    pl_module.logger.experiment.add_histogram(f'gradients/{name}', grad, pl_module.global_step)
        else:
            if layerwise:
                grad_metrics_per_layer[name] = {
                    'mean': None,
                    'std': None,
                    'sparsity': None,
                    'entropy': None,
                    'norm': None
                }

    # --------------------------
    # Model-wise Metrics for Parameters
    # --------------------------
    all_params = torch.cat(param_values)
    total_param_elements = all_params.numel()
    total_zero_param_elements = (all_params == 0).sum()
    param_mean = all_params.mean()
    param_std = all_params.std()
    param_sparsity = total_zero_param_elements.float() / total_param_elements if total_param_elements > 0 else torch.tensor(0.0, device=all_params.device)
    total_param_norm = torch.norm(torch.stack(param_norms), p=2)  # Aggregate L2 Norm

    # Compute entropy for parameters using std deviation
    param_variance = param_std ** 2
    param_entropy = 0.5 * (torch.log(2 * torch.pi * param_variance + 1e-12) + 1)

    # Log model-wise parameter histograms if logging is enabled
    if log_histograms and pl_module.logger is not None and hasattr(pl_module.logger.experiment, 'add_histogram'):
        pl_module.logger.experiment.add_histogram('parameters', all_params, pl_module.global_step)

    # Log model-wise metrics for parameters
    pl_module.log('param_mean', param_mean)
    pl_module.log('param_std', param_std)
    pl_module.log('param_sparsity', param_sparsity)
    pl_module.log('param_entropy', param_entropy)
    pl_module.log('param_norm', total_param_norm)

    # --------------------------
    # Model-wise Metrics for Gradients
    # --------------------------
    if grad_values:
        all_grads = torch.cat(grad_values)
        total_grad_elements = all_grads.numel()
        total_zero_grad_elements = (all_grads == 0).sum()
        grad_mean = all_grads.mean()
        grad_std = all_grads.std()
        grad_sparsity = total_zero_grad_elements.float() / total_grad_elements if total_grad_elements > 0 else torch.tensor(0.0, device=all_grads.device)
        total_grad_norm = torch.norm(torch.stack(grad_norms), p=2)  # Aggregate L2 Norm

        # Compute entropy for gradients using std deviation
        grad_variance = grad_std ** 2
        grad_entropy = 0.5 * (torch.log(2 * torch.pi * grad_variance + 1e-12) + 1)

        # Log model-wise gradient histograms if logging is enabled
        if log_histograms and pl_module.logger is not None and hasattr(pl_module.logger.experiment, 'add_histogram'):
            pl_module.logger.experiment.add_histogram('gradients', all_grads, pl_module.global_step)

        # Log model-wise metrics for gradients
        pl_module.log('grad_mean', grad_mean)
        pl_module.log('grad_std', grad_std)
        pl_module.log('grad_sparsity', grad_sparsity)
        pl_module.log('grad_entropy', grad_entropy)
        pl_module.log('grad_norm', total_grad_norm)
    else:
        # No gradients available
        device = pl_module.device if hasattr(pl_module, 'device') else 'cpu'
        zero_tensor = torch.tensor(0.0, device=device)
        pl_module.log('grad_mean', zero_tensor)
        pl_module.log('grad_std', zero_tensor)
        pl_module.log('grad_sparsity', zero_tensor)
        pl_module.log('grad_entropy', zero_tensor)
        pl_module.log('grad_norm', zero_tensor)

    # --------------------------
    # Layer-wise Metrics Logging
    # --------------------------
    if layerwise:
        for name in param_metrics_per_layer:
            # Parameters (Weights)
            pl_module.log(f'param_mean/{name}', param_metrics_per_layer[name]['mean'])
            pl_module.log(f'param_std/{name}', param_metrics_per_layer[name]['std'])
            pl_module.log(f'param_sparsity/{name}', param_metrics_per_layer[name]['sparsity'])
            pl_module.log(f'param_entropy/{name}', param_metrics_per_layer[name]['entropy'])
            pl_module.log(f'param_norm/{name}', param_metrics_per_layer[name]['norm'])

            # Gradients
            grad_metrics = grad_metrics_per_layer[name]
            if grad_metrics['mean'] is not None:
                pl_module.log(f'grad_mean/{name}', grad_metrics['mean'])
                pl_module.log(f'grad_std/{name}', grad_metrics['std'])
                pl_module.log(f'grad_sparsity/{name}', grad_metrics['sparsity'])
                pl_module.log(f'grad_entropy/{name}', grad_metrics['entropy'])
                pl_module.log(f'grad_norm/{name}', grad_metrics['norm'])
            else:
                # Log zeros if gradient is None
                zero_tensor = torch.tensor(0.0, device=param.device)
                pl_module.log(f'grad_mean/{name}', zero_tensor)
                pl_module.log(f'grad_std/{name}', zero_tensor)
                pl_module.log(f'grad_sparsity/{name}', zero_tensor)
                pl_module.log(f'grad_entropy/{name}', zero_tensor)
                pl_module.log(f'grad_norm/{name}', zero_tensor)
