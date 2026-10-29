from sklearn.cluster import MiniBatchKMeans, KMeans

from typing import Any, Dict, Optional, Union
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np


def load_kmeans(centroids, n_concepts: Optional[int] = None):
    if not n_concepts:
        n_concepts = centroids.shape[0]
    kmeans = KMeans(n_clusters=n_concepts, init=centroids, n_init=1, random_state=0)
    kmeans.fit(centroids)
    return kmeans

def predict_concept(lads, centroids):
    ### Given a set of centroids (C,A), determine the closest centroid for the inputs in a tensor of shape (N,A) ###
    distances = torch.cdist(lads, centroids)  # (N, C)
    labels = torch.argmin(distances, dim=-1)  # (N,)
    return labels, distances


def concept_similarity(lads, centroids, mode="cosinesim"):
    ### Given a set of centroids (C,A), determine the closest centroid for the inputs in a tensor of shape (N,A) ###
    # modes: cosinesim or dot
    C = centroids.shape[0]
    N = lads.shape[0]
    lads = lads.unsqueeze(1).repeat(1, C, 1)
    centroids = centroids.unsqueeze(0).repeat(N, 1, 1)
    if mode == "cosinesim":
        sim = torch.cosine_similarity(lads, centroids, dim = 2)
    elif mode == "dot":
        lads = F.normalize(lads, p=2.0, dim=-1)
        centroids = F.normalize(centroids, p=2.0, dim=-1)
        sim = torch.mul(lads, centroids).sum(dim=-1) 
    labels = torch.argmax(sim.squeeze(), dim=-1)
    return labels, sim.squeeze()



def compute_second_derivative_elbow(
    eval_points: torch.Tensor, y_values: torch.Tensor):
    """
    Find the elbow point using a finite difference approximation of the second derivative
    with non-uniform spacing (at halfway points).

    Args:
        eval_points (torch.Tensor): 1D tensor of x values (e.g., parameter settings)
        y_values (torch.Tensor): 1D tensor of corresponding metric values

    Returns:
        int: Index of the estimated elbow point in eval_points/y_values
    """

    assert eval_points.ndim == 1 and y_values.ndim == 1, "Inputs must be 1D tensors"
    assert len(eval_points) == len(y_values), "eval_points and y_values must have the same length"

    n = len(eval_points)
    if n < 3:
        raise ValueError("Need at least 3 points to compute second derivatives")

    second_derivs = []

    for i in range(1, n - 1):
            x_prev, x_curr, x_next = eval_points[i - 1], eval_points[i], eval_points[i + 1]
            y_prev, y_curr, y_next = y_values[i - 1], y_values[i], y_values[i + 1]

            h_prev = x_curr - x_prev
            h_next = x_next - x_curr
            h_total = x_next - x_prev

            if h_prev <= 0 or h_next <= 0:
                second_derivs.append(torch.tensor(0.0))
                continue

            term1 = y_next * (x_curr-x_prev)
            term2 = y_prev * (x_next-x_curr)
            term3 = y_curr * (x_next-x_prev)

            second_deriv =  2*(term1 + term2 - term3) / (h_prev*h_next*h_total)
            second_derivs.append(second_deriv)

    second_derivs = torch.stack(second_derivs)
    elbow_index = torch.argmax(second_derivs).item() + 1  # +1 to correct for offset

    return elbow_index

##########################
# Hyperparameter search  #
##########################

def compute_elbow_lmethod(eval_points, y_values):
    """
    Compute the elbow index using the L-method with robust handling of non-uniform eval_points.

    Args:
        eval_points (torch.Tensor): 1D tensor of x-values (e.g., parameter values)
        y_values (torch.Tensor): 1D tensor of y-values (e.g., loss or error metrics)

    Returns:
        int: Estimated elbow index (in eval_points and y_values)
    """
    if isinstance(eval_points, np.ndarray): eval_points = torch.as_tensor(eval_points, dtype=torch.float)
    if isinstance(y_values, np.ndarray): y_values = torch.as_tensor(y_values,  dtype=torch.float)

    assert eval_points.ndim == 1 and y_values.ndim == 1
    assert eval_points.shape[0] == y_values.shape[0]
    n = eval_points.shape[0]
    if n < 5:
        raise ValueError("At least 5 points are needed for the L-method")

    def linear_fit_mse(x, y):
        # Fit y = ax + b, return MSE
        X = torch.stack([x, torch.ones_like(x)], dim=1)  # [x, 1]
        coeffs = torch.linalg.lstsq( X, y.unsqueeze(1)).solution  # Least squares
        y_pred = (X @ coeffs).squeeze()
        return torch.mean((y - y_pred) ** 2)

    total_errors = []

    for i in range(2, n-2):  # avoid too-short segments
        x1, y1 = eval_points[:i], y_values[:i]
        x2, y2 = eval_points[i:], y_values[i:]

        mse1 = linear_fit_mse(x1, y1)
        mse2 = linear_fit_mse(x2, y2)

        total_error_scaled = mse1*(eval_points[i]-x1.min()) + mse2*(x2.max()-eval_points[i]) #Since range between 0 and 1
        total_errors.append(total_error_scaled)

    total_errors = torch.tensor(total_errors)
    elbow_index = torch.argmin(total_errors).item() + 2  # offset by loop start index

    return elbow_index



def distance_combined(X, Y, alpha=0.1, threshold=0.1):
    """
    X: [N, D] base features (tensors) -> centroid position
    Y: [N, D'] auxiliary features (e.g. metadata) ->IS
    Returns: [N, N] pairwise custom distances
    """
    # Compute pairwise Euclidean distances
    base_dist = cdist(X, X, metric='euclidean')  # [N, N]
    aux_dist = cdist(Y, Y, metric='euclidean')   # [N, N]

    # Apply your formula
    combined = base_dist/base_dist.max() + np.where(aux_dist > threshold, alpha * aux_dist, -alpha * aux_dist)/aux_dist.max()
    return combined

def explained_variation(X, labels): #TODO double check this expression
    total_var = np.var(X, axis=0).sum()
    within_var = 0
    for i in range(labels.max()+1):
        points = X[labels == i]
        if len(points) > 0:
            centroid = points.mean(axis=0)
            within_var += ((points - centroid) ** 2).sum()
    return 1 - (within_var / (X.shape[0] * total_var))