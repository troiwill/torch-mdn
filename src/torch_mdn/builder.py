import torch
import torch_mdn.layer
import torch_mdn.utils
from typing import Dict


class GMParamBuilder:
    r"""
    A class that computes the arguments for Linear layers, the Gaussian
    Mixture Layer, and the loss layer.
    """

    def __init__(self, in_features: int, ndim: int, nmodes: int,
        covar_type: int, device: torch.device = None) -> None:
        assert isinstance(in_features, int) and in_features >= 1, \
            "`in_features` must be a positive integer."
        assert isinstance(nmodes, int) and nmodes >= 1, \
            "`nmodes` must be a positive integer."
        assert isinstance(ndim, int) and ndim >= 1, \
            "`ndim` must be a positive integer."
        assert isinstance(covar_type, int), \
            "`covar_type` must be an integer."

        self.in_features = in_features
        self.nmodes = nmodes
        self.ndim = ndim
        self.covar_type = covar_type
        self.device = device
        self.dtype = torch.float32

        self.gml_params: Dict
        self.mix_layer_params: Dict
        self.mean_layer_params: Dict
        self.cpm_layer_params: Dict
    #end def

    def build(self) -> None:
        """
        Creates a set of parameters used to build the linear, GMM, and loss
        layers.
        """
        self.gml_params = dict([
            ("ndim", self.ndim),
            ("nmodes", self.nmodes),
            ("cpm_decomp", self.covar_type)
        ])

        self.mix_layer_params = dict([
            ("in_features", self.in_features),
            ("out_features", self.nmodes),
            ("bias", True),
            ("device", self.device),
            ("dtype", self.dtype)
        ])

        self.mean_layer_params = dict([
            ("in_features", self.in_features),
            ("out_features", self.ndim * self.nmodes),
            ("bias", True),
            ("device", self.device),
            ("dtype", self.dtype)
        ])

        num_cpm_params: int
        if self.covar_type == torch_mdn.layer.GM_COVAR_FULL_UU:
            num_cpm_params = torch_mdn.utils.num_tri_matrix_params_per_mode(
                self.ndim, False)
        else:
            raise Exception("CPM type is not implemented.")

        self.cpm_layer_params = dict([
            ("in_features", self.in_features),
            ("out_features", num_cpm_params * self.nmodes),
            ("bias", True),
            ("device", self.device),
            ("dtype", self.dtype)
        ])
    #end def
#end class
