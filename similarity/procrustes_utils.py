import numpy as np
from numpy.linalg import svd, norm

def procrustes_frame_dist(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute Procrustes distance between frames X and Y.
    Returns per-joint distances and mean distance.
    """
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)
    U, _, Vt = svd(X0.T @ Y0)
    R = U @ Vt
    scale = (Y0 * (X0.dot(R))).sum() / (norm(X0)**2 + 1e-8)
    Yp = scale * X0.dot(R)
    dists = norm(Y0 - Yp, axis=1)
    return dists, dists.mean()


def compute_procrustes_transform(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix aligning Y to X.
    """
    X0 = X - X.mean(axis=0)
    Y0 = Y - Y.mean(axis=0)
    U, _, Vt = svd(X0.T @ Y0)
    return U @ Vt