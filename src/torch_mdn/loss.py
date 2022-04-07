import math
import torch
from torch import Tensor
from torch.nn import Module
from torch_mdn.utils import to_triangular_matrix, diag_indices_tri


class GMLoss(Module):
    r"""
        Creates a criterion that computes the negative log-likelihood (NLL) 
        error for a Mixture Density Network with Gaussian kernels.
    """

    def __init__(self, ndim: int, nmodes: int, mat_is_covar: bool) -> None:
        super(GMLoss, self).__init__()
        if not isinstance(ndim, int) or ndim < 1:
            raise Exception("`ndim` must be a postive integer.")
        
        if not isinstance(nmodes, int) or nmodes < 0:
            raise Exception("`nmodes` must be a non-negative integer.")

        if not isinstance(mat_is_covar, bool):
            raise Exception("`mat_is_covar` must be a boolean variable.")

        self.__ndim = ndim
        self.__nmodes = nmodes
        self.__mat_is_covar = mat_is_covar
        self.__half_log_twopi = self.__ndim * 0.5 \
            * torch.log(torch.tensor(2 * math.pi))

        self.__tgt_size = (1, self.__ndim)
        self.__means_size = (self.__nmodes, self.__ndim)

        self.__u_diag_indices = torch.tensor(diag_indices_tri(ndim, True),
            dtype = torch.int64)

        if self.__mat_is_covar is True:
            raise NotImplementedError(
                "GMLoss cannot handle covariance matrices for now.")
    #end def

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Computes the negative log-likelihood error given a prediction from
        a Mixture Density Network and the target vector.
        """
        # Sanity checks.
        # Separate the parameters for the mixture coefficients, means, and 
        # covariance/precision matrices.
        mix_params, mean_params, cpmat_params = pred

        # Compute the exponential 'x' for the logsumexp function.
        target = target.view((-1,) + self.__tgt_size)
        mean_params = mean_params.view((-1,) + self.__means_size)
        residual = target - mean_params

        ln_mixpi = torch.log(mix_params)

        ln_det_sig = self.compute_ln_det_sigma(cpmat_params)

        residual = residual.view((-1, self.__nmodes, self.__ndim, 1))
        quad_sig = self.compute_quad_sigma(residual, cpmat_params)

        x = ln_mixpi - self.__half_log_twopi + ln_det_sig - (0.5 * quad_sig)

        # Perform the log-sum-exp trick if requested.
        lse = torch.logsumexp(x, 2)
        return -1.0 * torch.mean(lse)
    #end def

    def compute_ln_det_sigma(self, cpmat_params: Tensor) -> Tensor:
        """
        Computes the ln(det(sigma)) for the loss function, where sigma is the 
        predicted covariance/precision matrices from the Mixture Desnity
        Network.
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
        where sigma is the predicted covariance/precision matrices from the 
        Mixture Density Network and x is the residual.
        """
        # TODO: Sanity check.
        # Create U matrix of dim -> [batch, ndim, ndim, nmodes]
        u_mat = to_triangular_matrix(ndim = self.__ndim, params = cpmat_params)

        # Compatible sanity check.
        assert u_mat.size()[:2] == residual.size()[:2]
        assert u_mat.size()[-1] == residual.size()[-1]

        # Compute ||U(x - mu)||^2_2
        ur = torch.matmul(u_mat, residual)
        ur = torch.square(ur)
        ur = ur.sum(dim = 2, keepdim = False)

        return ur
    #end def
#end class
