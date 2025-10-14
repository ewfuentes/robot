"""
Spectral decomposition of graph Laplacian.

Computes eigenvectors and eigenvalues for image segmentation.
Uses the generalized eigenvalue formulation for numerical stability.
"""

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


class SpectralDecomposer:
    """
    Computes eigendecomposition of normalized graph Laplacian.

    Uses the generalized eigenvalue problem formulation:
    (D - W) v = λ D v

    This is equivalent to the normalized Laplacian but more numerically stable
    than explicitly computing D^(-1/2) (D - W) D^(-1/2).
    """

    def __init__(self, num_eigenvectors: int = 15):
        """
        Args:
            num_eigenvectors: Number of eigenvectors to compute (excluding constant)
        """
        self.num_eigenvectors = num_eigenvectors

    def decompose(
        self,
        W: torch.Tensor,
        return_all: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition of the normalized Laplacian.

        Solves the generalized eigenvalue problem: (D - W) v = λ D v

        Args:
            W: [N, N] affinity matrix
            return_all: If True, return all N eigenvectors/values

        Returns:
            eigenvalues: [K] or [N] eigenvalues (sorted ascending)
            eigenvectors: [N, K] or [N, N] eigenvectors (columns)
        """
        N = W.shape[0]
        device = W.device

        # Zero out diagonal to remove self-loops
        W = W.clone()
        W.fill_diagonal_(0.0)

        # Normalize W by its maximum (as in the reference implementation)
        W_max = W.max()
        if W_max > 0:
            W = W / W_max

        # Move to numpy for scipy
        W_np = W.cpu().numpy()

        # Compute degree matrix D (row sums)
        D_diag = W_np.sum(axis=1)
        # Prevent division by zero
        D_diag[D_diag < 1e-12] = 1.0
        D = sp.diags(D_diag)

        # Compute Laplacian matrix (D - W)
        L = D - W_np

        # Solve generalized eigenvalue problem: L v = λ D v
        if return_all or self.num_eigenvectors >= N - 1:
            # Compute all eigenvectors using dense solver
            L_dense = L if isinstance(L, np.ndarray) else L.toarray()
            D_dense = D.toarray()
            eigenvalues, eigenvectors = sp.linalg.eigh(L_dense, D_dense)
        else:
            # Compute smallest k+1 eigenvectors (including constant)
            k = min(self.num_eigenvectors + 1, N - 1)
            try:
                # Try with sigma=0 (shift-invert mode for smallest eigenvalues)
                eigenvalues, eigenvectors = eigsh(
                    L,
                    k=k,
                    M=D,
                    sigma=0,
                    which='LM',
                    tol=1e-6
                )
            except:
                # Fallback: compute smallest eigenvalues directly
                eigenvalues, eigenvectors = eigsh(
                    L,
                    k=k,
                    M=D,
                    which='SM',
                    tol=1e-6
                )

        # Convert back to torch
        eigenvalues = torch.from_numpy(eigenvalues).float().to(device)
        eigenvectors = torch.from_numpy(eigenvectors).float().to(device)

        # Skip the constant eigenvector (eigenvalue ≈ 0)
        if eigenvalues[0] < 1e-6:
            eigenvalues = eigenvalues[1:]
            eigenvectors = eigenvectors[:, 1:]

        # Fix sign ambiguity: flip eigenvectors so majority is positive
        for k in range(eigenvectors.shape[1]):
            if torch.mean((eigenvectors[:, k] > 0).float()).item() < 0.5:
                eigenvectors[:, k] = -eigenvectors[:, k]

        return eigenvalues, eigenvectors

    def __call__(
        self,
        W: torch.Tensor,
        return_all: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for decompose().
        """
        return self.decompose(W, return_all=return_all)
