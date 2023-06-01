from __future__ import annotations
from abc import abstractmethod
import math
import torch
from torch import Tensor
import torch.nn as nn
from torch_mdn.layer import MatrixDecompositionType
import torch_mdn.utils as utils


class _GaussianNLLLoss(nn.Module):
    """An abstract base class for developing the negative log-likelihood loss function."""

    def __init__(self, ndim: int, nmodes: int) -> None:
        """
        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__()
        # Sanity checks.
        if not isinstance(ndim, int) or ndim < 1:
            raise Exception("`ndim` must be a postive integer.")
        if not isinstance(nmodes, int) or nmodes < 1:
            raise Exception("`nmodes` must be a positive integer.")

        self.ndim = ndim
        self.nmodes = nmodes
        self.half_log_twopi = self.ndim * 0.5 * torch.log(torch.tensor(2 * math.pi))

    def extra_repr(self) -> str:
        """
        Returns detailed information about the loss function.

        Returns
        -------
        details : str
            Detailed information about the loss function.
        """
        return (
            f"ndim={self.ndim}, nmodes={self.nmodes}, "
            + f"target_size=[{self.ndim}, 1]"
        )


class _MatrixDecompositionNLLLoss(_GaussianNLLLoss):
    """An abstract base class for developing the matrix decomposition related operations for the
    negative log-likelihood loss function."""

    def __init__(self, ndim: int, nmodes: int) -> None:
        """
        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)

    @abstractmethod
    def compute_ln_det_sigma(self, cpmat_params: Tensor) -> Tensor:
        """
        Computes the ln(det(Sigma)) for the loss function, where sigma is the predicted
        covariance/precision matrices from the neural network. Note that Sigma is also decomposed.

        Parameters
        ----------
        cpmat_params : torch.Tensor
            The free parameters that are used to build the covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of ln(det(Sigma)).
        """
        raise NotImplementedError("This is an abstract class.")

    @abstractmethod
    def compute_quad_sigma(self, residual: Tensor, cpmat_params: Tensor) -> Tensor:
        """
        Computes the quadratic (x^T) * Sigma * (x) for the loss function, where Sigma is the
        predicted covariance/precision matrices from the neural network. Note that Sigma is also
        decomposed.

        Parameters
        ----------
        residual : torch.Tensor
            The residual/error that is computed using the target from the data.

        cpmat_params : torch.Tensor
            The free parameters that are used to build the covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of (x^T) * Sigma * (x).
        """
        raise NotImplementedError("This is an abstract class.")

    @abstractmethod
    def decomposed_name(self) -> str:
        """
        Returns the name of the decomposition method.

        Returns
        -------
        name : str
            The name of the decomposition method.
        """
        raise NotImplementedError("This is an abstract class.")


class _CholeskyDecompositionNLLLoss(_MatrixDecompositionNLLLoss):
    """An implementation of the negative log-likelihood loss function using the Cholesky Matrix
    Decomposition."""

    def __init__(self, ndim: int, nmodes: int = 1) -> None:
        """
        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)
        self.__u_diag_indices = torch.tensor(
            utils.diag_indices_tri(ndim=self.ndim, is_lower=False), dtype=torch.int64
        )

    def compute_ln_det_sigma(self, cpmat_params: Tensor) -> Tensor:
        """
        Computes the $ln(det(\Sigma))$ for the loss function, where Sigma is the predicted
        covariance/precision matrices from the output layer. Note that $\Sigma$ is also
        decomposed.

        Parameters
        ----------
        cpmat_params : torch.Tensor
            The free parameters that are used to build the covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of ln(det(\Sigma)).
        """
        # Extract the diagonal indices for the U matrices.
        diag_u = cpmat_params.index_select(2, self.__u_diag_indices)

        # Compute the sum of log(diag_u). This is equivalent to Tr(log(U)).
        return torch.log(diag_u).sum(dim=2, keepdim=True)

    def compute_quad_sigma(self, residual: Tensor, cpmat_params: Tensor) -> Tensor:
        """
        Computes the quadratic (x^T) * Sigma * (x) for the loss function, where Sigma is the
        predicted covariance/precision matrices from the neural network. Note that Sigma is also
        decomposed.

        Parameters
        ----------
        residual : torch.Tensor
            The residual/error that is computed using the target from the data.

        cpmat_params : torch.Tensor
            The free parameters that are used to build the covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of (x^T) * Sigma * (x).
        """
        # Do shape sanity check: residual.size[1:] == (nmodes, ndim, 1)
        if tuple(residual.size()[1:]) != (self.nmodes, self.ndim, 1):
            raise ValueError(
                f"residual.shape[1:] should be {(self.nmodes, self.ndim, 1)}, but got {tuple(residual.size()[1:])}."
            )

        # Create U matrix of dim -> [batch, nmodes, ndim, ndim]
        u_mat = utils.to_triangular_matrix(
            ndim=self.ndim, params=cpmat_params, is_lower=False
        )

        # Compatible sanity check before computing norm.
        if u_mat.size()[:2] != residual.size()[:2]:
            raise ValueError(
                f"u_mat.shape[:2] {tuple(residual.size()[:2])} != residual.shape[:2] {tuple(residual.size()[:2])}."
            )
        if u_mat.size()[-1] != self.ndim:
            raise ValueError(
                f"u_mat.shape[-1] should be {self.ndim}, but got {int(u_mat.size()[-1])}."
            )

        # Compute ||U * (x - mu)||^2_2
        ur = torch.matmul(u_mat, residual)
        ur = torch.square(ur)
        ur = ur.sum(dim=2, keepdim=False)

        return ur

    def decomposed_name(self) -> str:
        """
        Returns the name of the decomposition method.

        Returns
        -------
        name : str
            The name of the decomposition method.
        """
        return "Cholesky"


class _GaussianDecompositionNLLLoss(_GaussianNLLLoss):
    """A base class for Gaussian NLL classes that use matrix decomposition methods."""

    def __init__(
        self, matrix_decomp_type: MatrixDecompositionType, ndim: int, nmodes: int
    ) -> None:
        """
        matrix_decomp_type : MatrixDecompositionType
            The matrix decomposition type.

        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)
        if matrix_decomp_type == MatrixDecompositionType.full_UU:
            self.decomp_loss = _CholeskyDecompositionNLLLoss(
                ndim=self.ndim, nmodes=self.nmodes
            )
        else:
            raise NotImplementedError(
                f"Matrix decomposition {matrix_decomp_type} is not implemented."
            )

    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian matrix decomposition NLL loss function.

        Returns
        -------
        details : str
            Detailed information about the Gaussian matrix decomposition NLL loss function.
        """
        return (
            super().extra_repr()
            + f"matrix_decomposition={self.decomp_loss.decomposed_name()}"
        )


class GaussianCovarianceNLLoss(_GaussianDecompositionNLLLoss):
    """
    The negative log-likelihood loss function for `torch_mdn.layer.GaussianCovarianceLayer`.
    """

    def __init__(
        self, matrix_decomp_type: MatrixDecompositionType, ndim: int, nmodes: int
    ) -> None:
        """
        matrix_decomp_type : MatrixDecompositionType
            The matrix decomposition type.

        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(
            matrix_decomp_type=matrix_decomp_type, ndim=ndim, nmodes=nmodes
        )

    def forward(self, cpmat_params: Tensor, target: Tensor) -> Tensor:
        """
        Computes the negative log-likelihood loss for a zero-mean Gaussian PDF. This loss function
        is used for learning the covariance of a Gaussian function only.

        Parameters
        ----------
        cpmat_params : torch.Tensor
            The covariance/precision matrix free parameters from the covariance/precision layer.

        target : torch.Tensor
            The residual or error. This error is assumed to be sampled from the covariance \Sigma.

        Returns
        -------
        res : Tensor
            The mean negative log-likelihood of the batch.
        """
        # Reshape the target/residual.
        target = target.view((-1, 1, self.ndim, 1))

        # Compute the natural log of the determinant: ln(det(Sigma)^-1/2).
        ln_det_sigma = self.decomp_loss.compute_ln_det_sigma(cpmat_params=cpmat_params)

        # Compute the quadratic: e^T * Sigma^-1 * e.
        quad_sigma = self.decomp_loss.compute_quad_sigma(
            residual=target, cpmat_params=cpmat_params
        )

        res = -self.half_log_twopi + ln_det_sigma - (0.5 * quad_sigma)
        return -1.0 * torch.mean(res)


class GaussianNLLoss(_GaussianDecompositionNLLLoss):
    """
    The negative log-likelihood loss function for `torch_mdn.layer.GaussianLayer`.
    """

    def __init__(
        self, matrix_decomp_type: MatrixDecompositionType, ndim: int, nmodes: int
    ) -> GaussianNLLoss:
        """
        Creates an instance of the negative log-likelihood loss function for a Gaussian layer.

        matrix_decomp_type : MatrixDecompositionType
            The matrix decomposition type.

        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(
            matrix_decomp_type=matrix_decomp_type, ndim=ndim, nmodes=nmodes
        )

    def forward(
        self, mu_params: Tensor, cpmat_params: Tensor, target: Tensor
    ) -> Tensor:
        """
        Computes the negative log-likelihood loss for a Gaussian PDF. This loss function is used
        for learning the mean and covariance of a Gaussian PDF.

        Parameters
        ----------
        mu_params : torch.Tensor
            The mean free parameters from the Gaussian layer.

        cpmat_params : torch.Tensor
            The covariance/precision matrix free parameters from the Gaussian layer.

        target : torch.Tensor
            The target or error.

        Returns
        -------
        res : Tensor
            The mean negative log-likelihood of the batch.
        """
        # Compute the residual.
        target = target.view((-1, 1, self.ndim, 1))
        mu_params = mu_params.view((-1, self.nmodes, self.ndim, 1))
        residual = target - mu_params

        # Compute the natural log of the determinant: ln(det(Sigma)^-1/2).
        ln_det_sigma = self.decomp_loss.compute_ln_det_sigma(cpmat_params=cpmat_params)

        # Compute the quadratic: e^T * Sigma^-1 * e.
        quad_sigma = self.decomp_loss.compute_quad_sigma(
            residual=residual, cpmat_params=cpmat_params
        )

        res = -self.half_log_twopi + ln_det_sigma - (0.5 * quad_sigma)
        return -1.0 * torch.mean(res)


class GaussianMixtureNLLoss(_GaussianDecompositionNLLLoss):
    """
    The negative log-likelihood loss function for `torch_mdn.layer.GaussianMixtureLayer`.
    """

    def __init__(
        self, matrix_decomp_type: MatrixDecompositionType, ndim: int, nmodes: int
    ) -> GaussianMixtureNLLoss:
        """
        Creates an instance of the negative log-likelihood loss function for a Gaussian mixture
        layer.

        matrix_decomp_type : MatrixDecompositionType
            The matrix decomposition type.

        ndim : int
            The number of dimensions of the Gaussian distribution.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(
            matrix_decomp_type=matrix_decomp_type, ndim=ndim, nmodes=nmodes
        )

    def forward(
        self,
        coeff_params: Tensor,
        mu_params: Tensor,
        cpmat_params: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the negative log-likelihood loss for a mixture of Gaussians PDF. This loss
        function is used for learning the mixture coefficients, mean, and covariance of a mixture
        of Gaussians PDF.

        Parameters
        ----------
        coeff_params : torch.Tensor
            The free parameters for mixture coefficients from the Gaussian layer.

        mu_params : torch.Tensor
            The mean free parameters from the Gaussian layer.

        cpmat_params : torch.Tensor
            The covariance/precision matrix free parameters from the Gaussian layer.

        target : torch.Tensor
            The target or error.

        Returns
        -------
        res : Tensor
            The mean negative log-likelihood of the batch.
        """
        # Compute the residual.
        target = target.view((-1, 1, self.ndim, 1))
        mu_params = mu_params.view((-1, self.nmodes, self.ndim, 1))
        residual = target - mu_params

        # Compute the natural log of the mixture coefficients: ln(pi_{1...n})
        ln_coeffs = torch.log(coeff_params)

        # Compute the natural log of the determinant: ln(det(Sigma)^-1/2).
        ln_det_sigma = self.decomp_loss.compute_ln_det_sigma(cpmat_params=cpmat_params)

        # Compute the quadratic: e^T * Sigma^-1 * e.
        quad_sigma = self.decomp_loss.compute_quad_sigma(
            residual=residual, cpmat_params=cpmat_params
        )

        # Perform the log-sum-exp trick.
        x = ln_coeffs - self.half_log_twopi + ln_det_sigma - (0.5 * quad_sigma)
        lse = torch.logsumexp(x, 2)
        return -1.0 * torch.mean(lse)
