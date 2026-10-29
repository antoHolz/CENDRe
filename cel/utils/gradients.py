import torch

def get_gradients(g_outputs,inputs):
    """     dgdx=torch.autograd.functional.jacobian(lambda t: g(model(t)), inputs)
    dgdx= torch.diagonal(dgdx, offset=0, dim1=0, dim2=1)
    dgdx= dgdx.movedim(-1, 0) """
    dgdx=torch.autograd.grad(g_outputs, inputs,allow_unused=True, 
                    grad_outputs=torch.ones_like(g_outputs),
                    retain_graph=True
                    #is_grads_batched=True
                    )
    return dgdx[0]

_BATCHED_GRAD_THRESHOLD = 5  # use batched backward when n_classes > this

def get_classGradients(outputs, inputs, out_format='asdict'):
    """Compute per-class gradients of outputs w.r.t. inputs.
    For many classes (> _BATCHED_GRAD_THRESHOLD), uses a single batched
    autograd.grad call; otherwise loops sequentially (lower overhead)."""
    n_classes = outputs.shape[1]

    if out_format == 'astensor' and n_classes > _BATCHED_GRAD_THRESHOLD:
        B = outputs.shape[0]
        basis = torch.eye(n_classes, device=outputs.device, dtype=outputs.dtype
                         ).unsqueeze(1).expand(n_classes, B, n_classes)
        return torch.autograd.grad(
            outputs, inputs,
            grad_outputs=basis,
            retain_graph=True,
            is_grads_batched=True,
        )[0]  # (n_classes, B, ...)

    dgdx = {}
    for i in range(n_classes):
        dgdx[i] = torch.autograd.grad(
            outputs[:, i], inputs, allow_unused=True,
            grad_outputs=torch.ones_like(outputs[:, i]),
            retain_graph=True,
        )[0]
    if out_format == 'astensor':
        return torch.stack(list(dgdx.values()), dim=0)
    return dgdx