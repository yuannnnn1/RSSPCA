"""Principal components for standardized image matrices (one row per image)."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import svd
from scipy.sparse.linalg import spsolve


def grid_laplacian(height: int, width: int) -> sparse.csc_matrix:
    """Build D - A for four-neighbor pixels in row-major order."""
    pixels = np.arange(height * width).reshape(height, width)
    start = np.concatenate((pixels[:, :-1].ravel(), pixels[:-1].ravel()))
    end = np.concatenate((pixels[:, 1:].ravel(), pixels[1:].ravel()))
    rows = np.concatenate((start, end))
    cols = np.concatenate((end, start))
    adjacency = sparse.csc_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(pixels.size, pixels.size)
    )
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(degrees, format="csc") - adjacency


def pca_components(samples: np.ndarray, count: int) -> np.ndarray:
    """Return the leading right singular vectors as columns."""
    if not 1 <= count <= min(samples.shape):
        raise ValueError("Component count exceeds the matrix dimensions.")
    _, _, right = svd(samples, full_matrices=False, check_finite=False)
    return right[:count].T.copy()


@dataclass
class RSSPCAResult:
    components: np.ndarray
    iterations: list[int]
    converged: list[bool]


def fit_rsspca(
    samples: np.ndarray,
    laplacian: sparse.csc_matrix,
    initial: np.ndarray,
    eta1: float,
    eta2: float,
) -> RSSPCAResult:
    """Solve Wang et al. (2025), equations (32)–(33), with residual deflation."""
    if not np.isfinite([eta1, eta2]).all() or eta1 <= 0 or eta2 < 0:
        raise ValueError("RSSPCA requires finite eta1 > 0 and eta2 >= 0.")

    residual = samples.copy()
    components = np.empty_like(initial)
    iterations = []
    converged = []
    smoothness = eta2 * laplacian

    for k, direction in enumerate(initial.T):
        for iteration in range(1, 101):
            # Direct contractions avoid spurious floating-point flags in Accelerate.
            signs = np.sign(np.einsum("ij,j->i", residual, direction))
            linear_term = np.einsum("i,ij->j", signs, residual)
            weights = np.abs(direction)
            # For S = sqrt(U), solve the equivalent symmetric system for z.
            # The unnormalized update Uq is then S z, also when U has zeros.
            root = np.sqrt(weights)
            scale = sparse.diags(root, format="csc")
            matrix = sparse.diags(weights + eta1, format="csc")
            matrix += scale @ smoothness @ scale
            updated = root * spsolve(matrix, root * linear_term)
            norm = np.linalg.norm(updated)
            if not np.isfinite(norm) or norm <= np.finfo(float).eps:
                raise ValueError(f"Degenerate RSSPCA update for component {k + 1}.")

            updated /= norm
            # A component and its negative give identical projected distances.
            if np.einsum("i,i->", updated, direction) < 0:
                updated *= -1
            change = np.linalg.norm(updated - direction)
            direction = updated
            if change <= 1e-4:
                break

        components[:, k] = direction
        iterations.append(iteration)
        converged.append(bool(change <= 1e-4))
        residual -= np.outer(np.einsum("ij,j->i", residual, direction), direction)

    return RSSPCAResult(components, iterations, converged)
