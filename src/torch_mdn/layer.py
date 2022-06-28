import numpy as np
from typing import Tuple
import torch
import torch_mdn.utils
from torch import Tensor
import torch.nn
import torch.nn.functional as F


# Types of valid covariance matrices.
GM_COVAR_DIAG: int = 0
GM_COVAR_FULL_LDL: int = 1
GM_COVAR_FULL_UU: int = 2

GM_VALID_COVAR_TYPES = dict([
    ("DIAG", GM_COVAR_DIAG),
    ("LDL", GM_COVAR_FULL_LDL),
    ("UU", GM_COVAR_FULL_UU)
])

class GaussianMixtureLayer(torch.nn.Module):

    def __init__(self, in_features: int, ndim: int, nmodes: int,
        cpm_decomp: int, dtype=torch.float32) -> None:
        super(GaussianMixtureLayer, self).__init__()

        self.ndim = ndim
        self.nmodes = nmodes
        self.cpm_decomp_int = cpm_decomp
        if self.cpm_decomp_int != GM_COVAR_FULL_UU:
            raise Exception(f"Invalid CPM type {self.cpm_decomp_int} provided.")

        self.pmix_dist_shape = (self.nmodes, 1)
        self.pmu_dist_shape = (self.nmodes, self.ndim)

        num_mat_params: int
        if self.cpm_decomp_int == GM_COVAR_FULL_UU:
            num_mat_params = torch_mdn.utils.num_tri_matrix_params_per_mode(
                self.ndim, False)
            self.cpm_dist_shape = (self.nmodes, num_mat_params)
        else:
            raise Exception("CPM type not implemented.")

        # Create the free parameter linear layers.
        self.mixpi_freeparams = torch.nn.Linear(in_features=in_features,
            out_features=np.prod(self.pmix_dist_shape), dtype=dtype)

        self.mu_freeparams = torch.nn.Linear(in_features=in_features,
            out_features=np.prod(self.pmu_dist_shape), dtype=dtype)

        self.cpm_freeparams = torch.nn.Linear(in_features=in_features,
            out_features=np.prod(self.cpm_dist_shape), dtype=dtype)
    #end def

    def forward(self, x: Tensor, compute_mat: bool = False) \
        -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the free parameters for the mixture coefficients, mu, and
        covariance/precision matrices (CPM). Then reshapes the linear output of
        the MixPi, Mu, and CPM linear layers. Finally, applies activation 
        functions to each of the reshaped outputs.
        """
        mixcoeff = self.mixpi_freeparams(x)
        mu = self.mu_freeparams(x)
        cpm = self.cpm_freeparams(x)

        mixcoeff, mu, cpm = self.reshape_linear_output(mixcoeff, mu, cpm)
        mixcoeff, mu, cpm = self.apply_activation(mixcoeff, mu, cpm)

        if compute_mat:
            cpm = torch_mdn.utils.to_triangular_matrix(self.ndim, cpm, False)
            cpm = torch_mdn.utils.torch_matmul_4d(cpm.transpose(-2,-1), cpm)

        return mixcoeff, mu, cpm
    #end def

    def apply_activation(self, mixcoeff: Tensor, mu: Tensor, cpm: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies an activation function to the outpus from MixPi, Mu, and CPM
        linear layers. Before calling this function, please reshape each
        tensor. Returns the result of applying the activation functions.
        """
        mixcoeff = mixcoeff - torch.max(mixcoeff, dim = 1, keepdim = True)[0]
        mixcoeff = F.softmax(mixcoeff, dim = 1)

        if self.cpm_decomp_int == GM_COVAR_FULL_UU:
            diag_indices = torch_mdn.utils.diag_indices_tri(ndim = self.ndim,
                is_lower = False)
            cpm[:,:,diag_indices] = F.elu(cpm[:,:,diag_indices], alpha = 1.0) \
                + 1 + torch_mdn.utils.epsilon()
        else:
            raise Exception("CPM type not implemented.")

        return mixcoeff, mu, cpm
    #end def

    def predict(self, x: bool) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Infers a probability distribution p(y | x) given the input `x`.
        """
        return self.forward(x=x, compute_mat=True)
    #end def

    def reshape_linear_output(self, mixcoeff: Tensor, mu: Tensor,
        cpm: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Given outputs from the MixPi, Mu, and CPMat linear layers,
        this function reshapes each output to an appropriate shape. Returns 
        the linear outputs in their new shapes.
        """
        mixcoeff = mixcoeff.reshape((-1,) + self.pmix_dist_shape)
        mu = mu.reshape((-1,) + self.pmu_dist_shape)

        if self.cpm_decomp_int == GM_COVAR_FULL_UU:
            cpm = cpm.reshape((-1,) + self.cpm_dist_shape)
        else:
            raise Exception("CPM type not implemented.")

        return mixcoeff, mu, cpm
    #end def

    def extra_repr(self) -> str:
        return f"nmodes={self.nmodes}, ndims={self.ndim}, " \
            + "matrix_type=INFO [HARDCODED], " \
            + "matrix_decomposition=CHOLESKY [HARDCODED]"
    #end def
#end class
