"""
Spectral decomposition of graph Laplacian.

Computes eigenvectors and eigenvalues for image segmentation.
"""

import torch
import numpy as np
from scipy.sparse.linalg import eigsh


class SpectralDecomposer:
    """
    Computes eigendecomposition of normalized graph Laplacian.
    """

    def __init__(self, num_eigenvectors: int = 15):
        """
        Args:
            num_eigenvectors: Number of eigenvectors to compute (excluding constant)
        """
        self.num_eigenvectors = num_eigenvectors

    def compute_laplacian(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized Laplacian matrix.

        L = D^(-1/2) (D - W) D^(-1/2)

        Args:
            W: [N, N] affinity matrix

        Returns:
            L: [N, N] normalized Laplacian matrix
        """
        # Compute degree matrix
        D = torch.diag(W.sum(dim=1))

        # Compute D^(-1/2)
        D_values = D.diag()
        D_inv_sqrt = torch.zeros_like(D)
        # Avoid division by zero
        nonzero_mask = D_values > 1e-12
        D_inv_sqrt[nonzero_mask, nonzero_mask] = 1.0 / torch.sqrt(D_values[nonzero_mask])

        # Compute normalized Laplacian: L = D^(-1/2) (D - W) D^(-1/2)
        L = D_inv_sqrt @ (D - W) @ D_inv_sqrt

        return L

    def decompose(
        self,
        W: torch.Tensor,
        return_all: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition of the Laplacian.

        Args:
            W: [N, N] affinity matrix
            return_all: If True, return all N eigenvectors/values

        Returns:
            eigenvalues: [K] or [N] eigenvalues (sorted ascending)
            eigenvectors: [N, K] or [N, N] eigenvectors (columns)
        """
        N = W.shape[0]
        device = W.device

        # Compute Laplacian
        L = self.compute_laplacian(W)

        # Move to CPU for eigendecomposition (scipy is faster for this)
        L_np = L.cpu().numpy()

        # Compute eigendecomposition
        if return_all or self.num_eigenvectors >= N - 1:
            # Compute all eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L_np)
        else:
            # Compute smallest k+1 eigenvectors (including constant)
            k = min(self.num_eigenvectors + 1, N - 1)
            eigenvalues, eigenvectors = eigsh(
                L_np,
                k=k,
                which='SM',  # Smallest magnitude
                tol=1e-6
            )

        # Convert back to torch
        eigenvalues = torch.from_numpy(eigenvalues).float().to(device)
        eigenvectors = torch.from_numpy(eigenvectors).float().to(device)

        # Skip the constant eigenvector (eigenvalue ≈ 0)
        # The first eigenvalue should be very close to 0
        if eigenvalues[0] < 1e-6:
            eigenvalues = eigenvalues[1:]
            eigenvectors = eigenvectors[:, 1:]

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
