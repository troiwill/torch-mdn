from abc import ABC, abstractmethod
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch_mdn.layer as layer
import torch_mdn.utils as utils


class _GaussianNLLLoss(nn.Module):

    def __init__(self, ndim: int, nmodes: int) -> None:
        super().__init__()
        # Sanity checks.
        if not isinstance(ndim, int) or ndim < 1:
            raise Exception("`ndim` must be a postive integer.")
        if not isinstance(nmodes, int) or nmodes < 1:
            raise Exception("`nmodes` must be a positive integer.")

        self.ndim = ndim
        self.nmodes = nmodes
        self.half_log_twopi = self.ndim * 0.5 \
            * torch.log(torch.tensor(2 * math.pi))
    #end def

    def extra_repr(self) -> str:
        return f"ndim={self.ndim}, nmodes={self.nmodes}, " \
            + f"target_size={self.tgt_size}"
#end def


class _MatrixDecompositionNLLLoss(_GaussianNLLLoss):

    def __init__(self, ndim: int, nmodes: int) -> None:
        super().__init__(ndim, nmodes)

    @abstractmethod
    def compute_ln_det_sigma(self, cpmat_params: Tensor) -> Tensor:
        """
        Computes the ln(det(sigma)) for the loss function, where sigma is the 
        predicted covariance/precision matrices from the neural network. Note 
        that Sigma is also decomposed.

        Parameters
        ----------
        cpmat_params : torch.Tensor
            The free parameters that are used to build the 
            covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of ln(det(Sigma)).
        """
        raise NotImplementedError("This is an abstract class.")

    @abstractmethod
    def compute_quad_sigma(self, residual: Tensor, cpmat_params: Tensor) \
        -> Tensor:
        """
        Computes the quadratic (x^T) * Sigma * (x) for the loss function,
        where Sigma is the predicted covariance/precision matrices from the 
        neural network. Note that Sigma is also decomposed.

        Parameters
        ----------
        residual : torch.Tensor
            The residual/error that is computed using the target from the data.

        cpmat_params : torch.Tensor
            The free parameters that are used to build the 
            covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of (x^T) * Sigma * (x).
        """
        raise NotImplementedError("This is an abstract class.")

    @abstractmethod
    def decomposed_name(self) -> str:
        raise NotImplementedError("This is an abstract class.")
#end class


class _CholeskyDecompositionNLLLoss(_MatrixDecompositionNLLLoss):

    def __init__(self, ndim: int, nmodes: int = 1) -> None:
        super().__init__(ndim, nmodes)
        self.__u_diag_indices = torch.tensor(utils.diag_indices_tri(
            ndim = self.ndim, is_lower = False), dtype = torch.int64)
    #end def

    def compute_ln_det_sigma(self, cpmat_params: Tensor) -> Tensor:
        """
        Computes the $ln(det(\Sigma))$ for the loss function, where sigma is the 
        predicted covariance/precision matrices from the output layer. Note 
        that $\Sigma$ is also decomposed.

        Parameters
        ----------
        cpmat_params : torch.Tensor
            The free parameters that are used to build the 
            covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of ln(det(\Sigma)).
        """
        # Extract the diagonal indices for the U matrices.
        diag_u = cpmat_params.index_select(2, self.__u_diag_indices)

        # Compute the sum of log(diag_u). This is equivalent to Tr(log(U)).
        return torch.log(diag_u).sum(dim = 2, keepdim = True)
    #end def

    def compute_quad_sigma(self, residual: Tensor, cpmat_params: Tensor) \
        -> Tensor:
        """
        Computes the quadratic (x^T) * Sigma * (x) for the loss function,
        where Sigma is the predicted covariance/precision matrices from the 
        neural network. Note that Sigma is also decomposed.

        Parameters
        ----------
        residual : torch.Tensor
            The residual/error that is computed using the target from the data.

        cpmat_params : torch.Tensor
            The free parameters that are used to build the 
            covariance/precision matrix.

        Returns
        -------
        res : Tensor
            The result of (x^T) * Sigma * (x).
        """
        # Do shape sanity check: residual.size[1:] == (nmodes, ndim, 1)
        assert tuple(residual.size()[1:]) == (self.nmodes, self.ndim, 1)

        # Create U matrix of dim -> [batch, nmodes, ndim, ndim]
        u_mat = utils.to_triangular_matrix(ndim=self.ndim,
            params=cpmat_params, is_lower=False)

        # Compatible sanity check before computing norm.
        assert u_mat.size()[:2] == residual.size()[:2]
        assert u_mat.size()[-1] == self.ndim

        # Compute ||U * (x - mu)||^2_2
        ur = torch.matmul(u_mat, residual)
        ur = torch.square(ur)
        ur = ur.sum(dim = 2, keepdim = False)

        return ur
    #end def

    def decomposed_name(self) -> str:
        return "Cholesky"
#end class


class _GaussianDecompositionNLLLoss(_GaussianNLLLoss):

    def __init__(self, ndim: int, nmodes: int, decomp_loss_type: int) -> None:
        super().__init__(ndim, nmodes)
        if decomp_loss_type == layer.GMD_FULL_UU:
            self.decomp_loss = _CholeskyDecompositionNLLLoss(
                ndim=self.ndim, nmodes=self.nmodes)
        else:
            raise Exception("Unknown or unhandled decomposition type.")
    #end def

    def extra_repr(self) -> str:
        return super().extra_repr() \
            + f"matrix_decomp={self.decomp_loss.decomposed_name()}"
#end class


class GaussianCovarianceNLLoss(_GaussianDecompositionNLLLoss):

    def __init__(self, ndim: int, nmodes: int, decomp_loss_type: int) -> None:
        super().__init__(ndim, nmodes, decomp_loss_type)

    def forward(self, cpmat_params: Tensor, target: Tensor) -> Tensor:
        """
        Computes the negative log-likelihood loss for a zero-mean Gaussian 
        PDF. This loss function is used for learning the covariance of a 
        Gaussian function only.

        Parameters
        ----------
        cpmat_params : torch.Tensor
            The covariance/precision matrix free parameters from the 
            covariance/precision layer.
        target : torch.Tensor
            The residual or error. This error is assumed to be sampled from 
            the covariance \Sigma.

        Returns
        -------
        res : Tensor
            The mean negative log-likelihood of the batch.
        """
        # Reshape the target/residual.
        target = target.view((-1,) + self.tgt_size)

        # Compute the natural log of the determinant: ln(det(Sigma)^-1/2).
        ln_det_sigma = self.decomp_loss.compute_ln_det_sigma(
            cpmat_params=cpmat_params)

        # Compute the quadratic: e^T * Sigma^-1 * e.
        quad_sigma = self.decomp_loss.compute_quad_sigma(residual=target,
            cpmat_params=cpmat_params)

        res = -self.half_log_twopi + ln_det_sigma - (0.5 * quad_sigma)
        return -1.0 * torch.mean(res)
    #end def
#end class


class GaussianNLLoss(_GaussianDecompositionNLLLoss):

    def __init__(self, ndim: int, nmodes: int, decomp_loss_type: int) -> None:
        super().__init__(ndim, nmodes, decomp_loss_type)

    def forward(self, mu_params: Tensor, cpmat_params: Tensor,
        target: Tensor) -> Tensor:
        """
        Computes the negative log-likelihood loss for a Gaussian PDF.
        This loss function is used for learning the mean and covariance of a 
        Gaussian PDF.

        Parameters
        ----------
        mu_params : torch.Tensor
            The mean free parameters from the Gaussian layer.
        cpmat_params : torch.Tensor
            The covariance/precision matrix free parameters from the 
            Gaussian layer.
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
        ln_det_sigma = self.decomp_loss.compute_ln_det_sigma(
            cpmat_params=cpmat_params)

        # Compute the quadratic: e^T * Sigma^-1 * e.
        quad_sigma = self.decomp_loss.compute_quad_sigma(residual=residual,
            cpmat_params=cpmat_params)

        res = -self.half_log_twopi + ln_det_sigma - (0.5 * quad_sigma)
        return -1.0 * torch.mean(res)
    #end def
#end class


class GaussianMixtureNLLoss(_GaussianDecompositionNLLLoss):

    def __init__(self, ndim: int, nmodes: int, decomp_loss_type: int) -> None:
        super().__init__(ndim, nmodes, decomp_loss_type)

    def forward(self, coeff_params: Tensor, mu_params: Tensor,
        cpmat_params: Tensor, target: Tensor) -> Tensor:
        """
        Computes the negative log-likelihood loss for a mixture of Gaussians 
        PDF. This loss function is used for learning the mixture coefficients, 
        mean, and covariance of a mixture of Gaussians PDF.

        Parameters
        ----------
        coeff_params : torch.Tensor
            The free parameters for mixture coefficients from the Gaussian 
            layer.
        mu_params : torch.Tensor
            The mean free parameters from the Gaussian layer.
        cpmat_params : torch.Tensor
            The covariance/precision matrix free parameters from the 
            Gaussian layer.
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
        ln_det_sigma = self.decomp_loss.compute_ln_det_sigma(
            cpmat_params=cpmat_params)

        # Compute the quadratic: e^T * Sigma^-1 * e.
        quad_sigma = self.decomp_loss.compute_quad_sigma(residual=residual,
            cpmat_params=cpmat_params)

        # Perform the log-sum-exp trick.
        x = ln_coeffs - self.half_log_twopi + ln_det_sigma - (0.5 * quad_sigma)
        lse = torch.logsumexp(x, 2)
        return -1.0 * torch.mean(lse)
    #end def
#end class
