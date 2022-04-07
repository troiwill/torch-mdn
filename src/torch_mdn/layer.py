from typing import Tuple
from torch_mdn.utils import epsilon
import torch
from torch import Tensor
from torch.nn import functional as F

from utils import diag_indices_tri, num_tri_matrix_params_per_mode


# Types of valid covariance matrices.
GM_COVAR_DIAG: int = 0
GM_COVAR_FULL_LDL: int = 1
GM_COVAR_FULL_UU: int = 2

GM_VALID_COVAR_TYPES = dict([
    ("DIAG", GM_COVAR_DIAG),
    ("LDL", GM_COVAR_FULL_LDL),
    ("UU", GM_COVAR_FULL_UU)
])

class GaussianMixtureLayer:

    def __init__(self, ndim: int, nmodes: int, cpm_type: int) -> None:
        self.ndim = ndim
        self.nmodes = nmodes
        self.cpm_type_int = cpm_type \
            if cpm_type in list(GM_VALID_COVAR_TYPES.keys()) \
                else GM_COVAR_FULL_UU
        self.is_training: bool = True
        self.set_train_mode()

        self.pmix_dist_shape = (self.nmodes, self.ndim)
        self.pmu_dist_shape = (self.nmodes, self.ndim)

        if self.cpm_type_int == GM_COVAR_FULL_UU:
            num_mat_params = num_tri_matrix_params_per_mode(self.ndim, False)
            self.cpm_dist_shape = (self.nmodes, num_mat_params)
        else:
            raise Exception("CPM type not implemented.")
    #end def

    def __call__(self, mix_out: Tensor, mu_out: Tensor, cpm_out: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor]:
        """
        Reshapes the linear output of the MixPi, Mu, and CPM linear layers.
        Then applies activation functions to each of the reshaped outputs.
        """
        out = self.reshape_linear_output(mix_out, mu_out, cpm_out)
        return self.apply_activation(*out)
    #end def

    def apply_activation(self, mix_out: Tensor, mu_out: Tensor,
        cpm_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies an activation function to the outpus from MixPi, Mu, and CPM
        linear layers. Before calling this function, please reshape each
        tensor. Returns the result of applying the activation functions.
        """
        mix_out = mix_out - torch.max(mix_out, dim = 1, keepdim = True)[0]
        mix_out = F.softmax(mix_out, dim = 1)

        if self.cpm_type_int == GM_COVAR_FULL_UU:
            diag_indices = diag_indices_tri(ndim = self.ndim,
                is_lower = False)
            cpm_out[:,:,diag_indices] = F.elu(cpm_out[:,:,diag_indices],
                alpha = 1.0) + 1 + epsilon()

        return mix_out, mu_out, cpm_out
    #end def

    def reshape_linear_output(self, mix_out: Tensor, mu_out: Tensor,
        cpm_out: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Given outputs from the MixPi, Mu, and CPMat linear layers,
        this function reshapes each output to an appropriate shape. Returns 
        the linear outputs in their new shapes.
        """
        mix_out = mix_out.reshape((-1,) + self.pmix_dist_shape)
        mu_out = mu_out.reshape((-1,) + self.pmu_dist_shape)

        if self.cpm_type_int == GM_COVAR_FULL_UU:
            cpm_out = cpm_out.reshape((-1,) + self.cpm_dist_shape)

        return mix_out, mu_out, cpm_out
    #end def

    def set_test_mode(self) -> None:
        self.is_training = False

    def set_train_mode(self) -> None:
        self.is_training = True

    def extra_repr(self) -> str:
        return ""
#end class
