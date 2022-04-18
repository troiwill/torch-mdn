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

    def __init__(self, ndim: int, nmodes: int, cpm_decomp: int) -> None:
        super(GaussianMixtureLayer, self).__init__()
        self.ndim = ndim
        self.nmodes = nmodes
        self.cpm_decomp_int = cpm_decomp
        if self.cpm_decomp_int != GM_COVAR_FULL_UU:
            raise Exception(f"Invalid CPM type {self.cpm_decomp_int} provided.")

        self.pmix_dist_shape = (self.nmodes, 1)
        self.pmu_dist_shape = (self.nmodes, self.ndim)

        if self.cpm_decomp_int == GM_COVAR_FULL_UU:
            num_mat_params = torch_mdn.utils.num_tri_matrix_params_per_mode(
                self.ndim, False)
            self.cpm_dist_shape = (self.nmodes, num_mat_params)
    #end def

    def forward(self, mixcoeff: Tensor, mu: Tensor, cpm: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor]:
        """
        Reshapes the linear output of the MixPi, Mu, and CPM linear layers.
        Then applies activation functions to each of the reshaped outputs.
        """
        mixcoeff, mu, cpm = self.reshape_linear_output(mixcoeff, mu, cpm)
        mixcoeff, mu, cpm = self.apply_activation(mixcoeff, mu, cpm)

        if not self.training:
            cpm = torch_mdn.utils.to_triangular_matrix(self.ndim, cpm, False)
            cpm = torch.einsum('abcd, abde -> abce', cpm, cpm)

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

        return mixcoeff, mu, cpm
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

        return mixcoeff, mu, cpm
    #end def

    def extra_repr(self) -> str:
        return f"nmodes={self.nmodes}, ndims={self.ndim}, " \
            + "matrix_type=INFO, matrix_decomposition=CHOLESKY"
    #end def
#end class
