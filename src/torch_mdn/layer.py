from abc import ABC, abstractmethod
from collections import OrderedDict
import math
import torch
import torch_mdn.utils as utils
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, OrderedDict, Tuple, Union


# Types of valid covariance matrices.
GMD_DIAG: int = 0
GMD_FULL_LDL: int = 1
GMD_FULL_UU: int = 2
GM_VALID_COVAR_TYPES = dict([
    ("DIAG", GMD_DIAG),
    ("LDL", GMD_FULL_LDL),
    ("UU", GMD_FULL_UU)
])

# Types of valid output covariance/precision matrices.
GTL_MAT_OUT_DECOMPED: int = 0
GTL_MAT_OUT_PRECISION: int = 1
GTL_MAT_OUT_COVARIANCE: int = 2
GTL_VALID_MAT_OUT_TYPES = dict([
    ("DECOM", GTL_MAT_OUT_DECOMPED),
    ("PREC", GTL_MAT_OUT_PRECISION),
    ("COVAR", GTL_MAT_OUT_COVARIANCE)
])
GTL_VALID_MAT_OUT_NAMES = dict([
    (GTL_MAT_OUT_DECOMPED, "Decomposed Matrix"),
    (GTL_MAT_OUT_PRECISION, "Precision Matrix"),
    (GTL_MAT_OUT_COVARIANCE, "Covariance Matrix")
])


class _GaussianLayerComponent(nn.Module, ABC):

    def __init__(self, ndim: int, nmodes: int) -> None:
        super().__init__()

        # Sanity checks.
        if not isinstance(ndim, int) or ndim < 1:
            raise Exception('ndim must be a positive integer.')
        if not isinstance(nmodes, int) or nmodes < 1:
            raise Exception('nmodes must be a positive integer.')

        self.ndim = ndim
        self.nmodes = nmodes
    #end def

    @abstractmethod
    def apply_activation(self, x: Tensor) -> Tensor:
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build 
        some component.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This 
            argument is optional and is mainly useful for debugging.

        Result
        ------
        res : tuple[int]
            The expected input shape to this layer.
        """
        raise NotImplementedError("This is an abstract method.")
    #end def

    def forward(self, x: Tensor) -> Union[Tensor, Iterable[Tensor]]:
        # Reshape the input tensor.
        x = self.reshape_input(x)

        # Then apply the activation function.
        x = self.apply_activation(x)

        return x
    #end def

    def predict(self, x: Tensor) -> Tensor:
        """
        Performs inference given the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        Returns
        -------
        res : torch.Tensor
            The prediction of this component given the input tensor.
        """
        return self.forward(x)
    #end def

    @abstractmethod
    def reshape_input(self, x: Tensor) -> Tensor:
        raise NotImplementedError("This is an abstract method.")
#end class


class _GaussianMeanLayer(_GaussianLayerComponent):
    
    def __init__(self, ndim: int, nmodes: int) -> None:
        super().__init__(ndim, nmodes)
        self.mu_dist_shape = (self.nmodes, self.ndim)

    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Returns the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The reshaped, sub-tensor output from the neural network that
            corresponds to the mean of a Gaussian.

        Returns
        -------
        res : torch.Tensor
            The unaltered input to the method.
        """
        return x
    #end def

    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build 
        the mean. This function can be used to compute the number
        of output features from the previous linear layer as 
        `math.prod(mu_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This 
            argument is optional and is mainly useful for debugging.

        Result
        ------
        res : tuple[int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self.mu_dist_shape))
    #end def

    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for downstream operations.

        Parameters
        ----------
        x : torch.Tensor
            The sub-tensor output from the neural network that corresponds 
            to the mean of a Gaussian.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor with shape (batch_size, nmodes, ndim)
        """
        return x.reshape((-1,) + self.mu_dist_shape)
    #end def
#end class


class _GaussianMixCoeffLayer(_GaussianLayerComponent):
    
    def __init__(self, ndim: int, nmodes: int) -> None:
        super().__init__(ndim, nmodes)
        self.mix_dist_shape = (self.nmodes, 1)
    #end def

    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Returns the mixture weights derived from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The reshaped, sub-tensor output from the neural network that
            corresponds to the mixture weights of a Gaussian Mixture model.

        Returns
        -------
        res : torch.Tensor
            The mixture weights.
        """
        x = x - torch.max(x, dim = 1, keepdim = True)[0]
        x = F.softmax(x, dim = 1)

        return x
    #end def

    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build 
        the mixture weights. This function can be used to compute the number 
        of output features from the previous linear layer as 
        `math.prod(mix_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This 
            argument is optional and is mainly useful for debugging.

        Result
        ------
        res : tuple[int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self.mix_dist_shape))
    #end def

    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for downstream operations.

        Parameters
        ----------
        x : torch.Tensor
            The sub-tensor output from the neural network that corresponds to 
            the mixture weights of a Gaussian Mixture model.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor with shape (batch_size, nmodes, 1)
        """
        return x.reshape((-1,) + self.mix_dist_shape)
    #end def
#end class


class _CompositeLayerBase(nn.Module):

    def __init__(self, components: OrderedDict[str, _GaussianLayerComponent]) \
        -> None:
        super().__init__()
        # Sanity checks.
        if not isinstance(components, OrderedDict):
            raise Exception('components must be an ordered dictionary.')

        for c in components:
            if not isinstance(c, _GaussianLayerComponent):
                raise Exception('Invalid type found. Expected ' \
                    + f'_GaussianLayerComponent. Found {type(c)}.')
        
        # An ordered dictionary of Gaussian components.
        self.components = components

        self.input_slices = list()
        for _, layer in self.components.items():
            width = layer.expected_input_shape()[1]
            if len(self.input_slices) == 0:
                start = 0
            else:
                start = self.input_slices[-1][1]
            self.input_slices.append((start, start + width))
        #end for
    #end def
    
    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """
        Computes the forward for each component object.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        res : tuple of torch.Tensor
            The forward outputs of each component in a tuple.
        """
        return tuple([ comp.forward(x) for comp in self.components.values() ])

    def predict(self, x: Tensor) -> Tuple[Tensor]:
        """
        Computes the prediction for each component object.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        res : tuple of torch.Tensor
            The prediction outputs of each component in a tuple.
        """
        return tuple([ comp.predict(x) for comp in self.components.values() ])

    def split_input(self, x: Tensor) -> Iterable[Tensor]:
        return tuple([ x[:, in_slice[0] : in_slice[1]] \
            for in_slice in self.input_slices])
#end class


class GaussianCovarianceLayer(_GaussianLayerComponent):
    
    def __init__(self, ndim: int, nmodes: int, cpm_decomp: int, 
        pred_matrix_type: int) -> None:
        super().__init__(ndim, nmodes)
        # Determine the covariance/precision matrix type.
        self.__cpm_decomp_int: int = cpm_decomp
        if self.__cpm_decomp_int != GMD_FULL_UU:
            raise Exception("Invalid covariance/precision matrix type. " \
                + f"Found {self.__cpm_decomp_int}. Valid options are " \
                + f"{GM_VALID_COVAR_TYPES.values()}.")

        num_mat_params: int
        if self.__cpm_decomp_int == GMD_FULL_UU:
            num_mat_params = utils.num_tri_matrix_params_per_mode(
                self.ndim, False)
            self.__cpm_dist_shape = (self.nmodes, num_mat_params)
        else:
            raise Exception("CPM type not implemented.")

        self.__pred_matrix_type_int: int = pred_matrix_type
        if self.__pred_matrix_type_int not in GTL_VALID_MAT_OUT_TYPES.values():
            raise Exception("Invalid output matrix type. Found " \
                + f"{self.__pred_matrix_type_int}. Valid options are " \
                + f"{GTL_VALID_MAT_OUT_TYPES.values()}")
    #end def

    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Returns the input free parameters after an activation function has 
        been applied.

        Parameters
        ----------
        x : torch.Tensor
            The reshaped, sub-tensor output from the neural network that
            corresponds to the covariance of the Gaussian.

        Returns
        -------
        res : torch.Tensor
            The free parameters after an activation function has been applied.
        """
        if self.__cpm_decomp_int == GMD_FULL_UU:
            diag_indices = utils.diag_indices_tri(ndim = self.ndim,
                is_lower = False)
            x[:,:,diag_indices] = F.elu(x[:,:,diag_indices], alpha = 1.0) \
                + 1 + utils.epsilon()
        else:
            raise Exception(
                f"CPM type {self.__cpm_decomp_int} not implemented.")

        return x
    #end def

    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build 
        the covariance/precision matrix (or matrices). This function can be 
        used to compute the number of output features from the previous 
        linear layer as `math.prod(cpm_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This 
            argument is optional and is mainly useful for debugging.

        Result
        ------
        res : tuple[int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self.__cpm_dist_shape))
    #end def

    def predict(self, x: Tensor) -> Tensor:
        """
        Performs inference given the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neuraln network.

        Returns
        -------
        res : torch.Tensor
            The prediction of the mixture coeffients given the input tensor.
        """
        x = self.forward(x)

        # Check the output format of the matrix.
        if self.__pred_matrix_type_int \
            in [ GTL_MAT_OUT_PRECISION, GTL_MAT_OUT_COVARIANCE ]:
            x = utils.to_triangular_matrix(self.ndim, x, False)
            x = utils.torch_matmul_4d(x.transpose(-2,-1), x)

            if self.__pred_matrix_type_int == GTL_MAT_OUT_COVARIANCE:
                x = torch.inverse(x)
        #end if

        return x
    #end def

    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for downstream operations.

        Parameters
        ----------
        x : torch.Tensor
            The sub-tensor output from the neural network that corresponds to 
            the free parameters of the covariance matrix.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor with shape depending on the covariance 
            decomposition.
        """
        return x.reshape((-1,) + self.__cpm_dist_shape)
    #end def

    def extra_repr(self) -> str:
        return f"nmodes={self.nmodes}, ndims={self.ndim}, " \
            + "matrix_type=INFO [HARDCODED], " \
            + "matrix_decomposition=CHOLESKY [HARDCODED]" \
            + "predicted_matrix_type=" \
            + F"{GTL_VALID_MAT_OUT_NAMES[self.__pred_matrix_type_int]}"
#end class


class GaussianLayer(_CompositeLayerBase):
    
    def __init__(self, ndim: int, nmodes: int, cpm_decomp: int,
        pred_matrix_type: int) -> None:
        super().__init__(components=OrderedDict([
            ('mu', _GaussianMeanLayer(ndim=ndim, nmodes=nmodes)),
            ('sigma', GaussianCovarianceLayer(ndim=ndim, nmodes=nmodes,
                cpm_decomp=cpm_decomp, pred_matrix_type=pred_matrix_type))
        ]))
    #end def

    def extra_repr(self) -> str:
        return self.components['sigma'].extra_repr()
#end class


class GaussianMixtureLayer(_CompositeLayerBase):
    
    def __init__(self, ndim: int, nmodes: int, cpm_decomp: int,
        pred_matrix_type: int) -> None:
        super().__init__(components=OrderedDict([
            ('mix', _GaussianMixCoeffLayer(ndim=ndim, nmodes=nmodes)),
            ('mu', _GaussianMeanLayer(ndim=ndim, nmodes=nmodes)),
            ('sigma', GaussianCovarianceLayer(ndim=ndim, nmodes=nmodes,
                cpm_decomp=cpm_decomp, pred_matrix_type=pred_matrix_type))
        ]))
    #end def

    def extra_repr(self) -> str:
        return self.components['sigma'].extra_repr()
#end class
