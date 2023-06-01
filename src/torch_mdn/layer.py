from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
import math
import torch
import torch_mdn.utils as utils
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, OrderedDict, Tuple, Union


class MatrixDecompositionType(Enum):
    """Enum for specifying the types of decomposed Gaussian covariance matrices."""

    diagonal = "Diagonal Decomposition"
    """Decomposition of a diagonal covariance matrix."""

    full_LDL = "Full Matrix LDL Decomposition"
    """LDL decomposition of a full covariance matrix."""

    full_UU = "Full Matrix UU Decomposition"
    """UU decomposition of a full covariance matrix."""


class MatrixPredictionType(Enum):
    """Enum for specifying how to output a covariance matrix during prediction/testing."""

    decomposed = "Decomposed"
    """Output the raw/decomposed matrix (the output one would get from training.)"""

    precision = "Precision"
    """Output the precision/infomation matrix."""

    covariance = "Covariance"
    """Output the covariance matrix."""


class _GaussianLayerComponent(nn.Module, ABC):
    """A base class that provides methods for implementing individual Gaussian Mixture components
    (that is, the covariance, mixture coefficients, and mean).
    """

    def __init__(self, ndim: int, nmodes: int) -> None:
        """
        Parameters
        ----------
        ndim : int
            The number of dimensions for the Gaussian.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__()

        # Sanity checks.
        if not isinstance(ndim, int) or ndim < 1:
            raise Exception("ndim must be a positive integer.")
        if not isinstance(nmodes, int) or nmodes < 1:
            raise Exception("nmodes must be a positive integer.")

        self.ndim = ndim
        self.nmodes = nmodes

    @abstractmethod
    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Applies an activation function to the input tensor. Usually called within the forward(...)
        function.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The result of applying the activation function to the tensor `x`.
        """
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build some component.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Result
        ------
        res : tuple[int]
            The expected input shape to this layer.
        """
        raise NotImplementedError("This is an abstract method.")

    def forward(self, x: Tensor) -> Union[Tensor, Iterable[Tensor]]:
        """
        Called when applying a "Gaussian" layer. `x` is usually the output of a feed-forward
        network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The result of "computing" a particular Gaussian component.
        """
        # Reshape the input tensor.
        x = self.reshape_input(x)

        # Then apply the activation function.
        x = self.apply_activation(x)

        return x

    def predict(self, x: Tensor) -> Tensor:
        """
        Performs inference given the input tensor `x`.

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

    @abstractmethod
    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor.
        """
        raise NotImplementedError("This is an abstract method.")


class _GaussianMeanLayer(_GaussianLayerComponent):
    """A class for implementing operations for a Gaussian mean.

    WARNING: This class should never be used standalone.
    """

    def __init__(self, ndim: int, nmodes: int) -> None:
        """
        Parameters
        ----------
        ndim : int
            The number of dimensions for the Gaussian.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)
        self.mu_dist_shape = (self.nmodes, self.ndim)

    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Returns the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The reshaped, sub-tensor output from the neural network that corresponds to the mean
            of a Gaussian.

        Returns
        -------
        res : torch.Tensor
            The unaltered input to the method.
        """
        return x

    def expected_input_shape(self, batch_size: int = 1) -> tuple[int, ...]:
        """
        Computes the expected input shape of the free parameters used to build the mean. This
        function can be used to compute the number of output features from the previous linear
        layer as `math.prod(mu_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        res : tuple[int, ...]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self.mu_dist_shape))

    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for downstream operations.

        Parameters
        ----------
        x : torch.Tensor
            The sub-tensor output from the neural network that corresponds to the mean of a
            Gaussian.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor with shape (batch_size, nmodes, ndim).
        """
        return x.reshape((-1,) + self.mu_dist_shape)


class _GaussianMixCoeffLayer(_GaussianLayerComponent):
    """A class for implementing operations for Gaussian mixture coefficients.

    WARNING: This class should never be used standalone.
    """

    def __init__(self, ndim: int, nmodes: int) -> None:
        """
        Parameters
        ----------
        ndim : int
            The number of dimensions for the Gaussian.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)
        self.mix_dist_shape = (self.nmodes, 1)

    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Returns the mixture weights derived from the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The reshaped, sub-tensor output from the neural network that corresponds to the
            mixture weights of a Gaussian Mixture model.

        Returns
        -------
        res : torch.Tensor
            The mixture weights.
        """
        x = x - torch.max(x, dim=1, keepdim=True)[0]
        x = F.softmax(x, dim=1)

        return x

    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build the mixture
        weights. This function can be used to compute the number of output features from the
        previous linear layer as `math.prod(mix_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        res : tuple[int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self.mix_dist_shape))

    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for downstream operations.

        Parameters
        ----------
        x : torch.Tensor
            The sub-tensor output from the neural network that corresponds to the mixture weights
            of a Gaussian Mixture model.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor with shape (batch_size, nmodes, 1).
        """
        return x.reshape((-1,) + self.mix_dist_shape)


class _CompositeLayerBase(nn.Module):
    """A class for implementing Gaussian layers with multiple components (that is, the Gaussian
    layer [mean and covariance] and the Gaussian Mixture layer [mixcoeff, mean, and covariance]).

    WARNING: This class should never be used standalone.
    """

    def __init__(self, components: OrderedDict[str, _GaussianLayerComponent]) -> None:
        """
        Parameters
        ----------
        components : OrderedDict[str, _GaussianLayerComponent]
            The components for a multiple "component" Gaussian layer.
        """
        super().__init__()
        # Sanity checks.
        if not isinstance(components, OrderedDict):
            raise Exception("components must be an ordered dictionary.")

        for c in components:
            if not isinstance(c, _GaussianLayerComponent):
                raise Exception(
                    "Invalid type found. Expected "
                    + f"_GaussianLayerComponent. Found {type(c)}."
                )

        # An ordered dictionary of Gaussian components.
        self.components = components.copy()

        self.input_slices = list()
        for _, layer in self.components.items():
            width = layer.expected_input_shape()[1]
            if len(self.input_slices) == 0:
                start = 0
            else:
                start = self.input_slices[-1][1]
            self.input_slices.append((start, start + width))

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        """
        Computes the forward for each component object.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        Returns
        -------
        res : tuple of torch.Tensor
            The forward outputs of each component in a tuple.
        """
        return tuple([comp.forward(x) for comp in self.components.values()])

    def predict(self, x: Tensor) -> Tuple[Tensor]:
        """
        Computes the prediction for each component object.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        Returns
        -------
        res : tuple of torch.Tensor
            The prediction outputs of each component in a tuple.
        """
        return tuple([comp.predict(x) for comp in self.components.values()])

    def split_input(self, x: Tensor) -> Iterable[Tensor]:
        """
        Splits the input tensor according to the generated slices.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : tuple[torch.Tensor]
            A tuple of split tensors.
        """
        return tuple(
            [x[:, in_slice[0] : in_slice[1]] for in_slice in self.input_slices]
        )


class GaussianCovarianceLayer(_GaussianLayerComponent):
    """A class that implements the Gaussian covariance layer."""

    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: int,
        nmodes: int = 1,
    ) -> GaussianCovarianceLayer:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : int
            The number of dimensions for the Gaussian.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)

        # Determine the matrix decomposition type.
        self.__matrix_decomp_type = matrix_decomp_type
        if self.__matrix_decomp_type == MatrixDecompositionType.full_UU:
            num_mat_params = utils.num_tri_matrix_params_per_mode(self.ndim, False)
            self.__cpm_dist_shape = (self.nmodes, num_mat_params)
        else:
            raise NotImplementedError(
                f"The matrix decomposition type {self.__matrix_decomp_type} is not implemented."
            )

        # Determine the matrix prediction/inference type.
        self.__predict_matrix_type = predict_matrix_type

    def apply_activation(self, x: Tensor) -> Tensor:
        """
        Returns the input free parameters after an activation function has been applied.

        Parameters
        ----------
        x : torch.Tensor
            The reshaped, sub-tensor output from the neural network that corresponds to the
            covariance of the Gaussian.

        Returns
        -------
        res : torch.Tensor
            The free parameters after an activation function has been applied.
        """
        if self.__matrix_decomp_type == MatrixDecompositionType.full_UU:
            diag_indices = utils.diag_indices_tri(ndim=self.ndim, is_lower=False)
            x[:, :, diag_indices] = (
                F.elu(x[:, :, diag_indices], alpha=1.0) + 1 + utils.epsilon()
            )
        else:
            raise NotImplementedError(
                f"Matrix decomposition type {self.__matrix_decomp_type} is not implemented."
            )

        return x

    def expected_input_shape(self, batch_size: int = 1) -> Tuple[int]:
        """
        Computes the expected input shape of the free parameters used to build the
        covariance/precision matrix (or matrices). This function can be used to compute the number
        of output features from the previous linear layer as `math.prod(cpm_input_shape)` when
        `batch_size = 1`.

        Parameters
        ----------
        batch_size : int
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Result
        ------
        res : tuple[int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self.__cpm_dist_shape))

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
            The predicted matrix given the input tensor.
        """
        x = self.forward(x)

        # Check the output format of the matrix.
        if self.__predict_matrix_type in [
            MatrixPredictionType.covariance,
            MatrixPredictionType.precision,
        ]:
            x = utils.to_triangular_matrix(self.ndim, x, False)
            x = utils.torch_matmul_4d(x.transpose(-2, -1), x)

            if self.__predict_matrix_type == MatrixPredictionType.covariance:
                x = torch.inverse(x)

        return x

    def reshape_input(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for downstream operations.

        Parameters
        ----------
        x : torch.Tensor
            The sub-tensor output from the neural network that corresponds to the free parameters
            of the covariance matrix.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor with shape depending on the covariance decomposition.
        """
        return x.reshape((-1,) + self.__cpm_dist_shape)

    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian covariance layer.

        Returns
        -------
        details : str
            Detailed information about the Gaussian covariance layer.
        """
        return (
            f"nmodes={self.nmodes}, ndims={self.ndim}, "
            + "matrix_type=INFO [HARDCODED], "
            + "matrix_decomposition=CHOLESKY [HARDCODED], "
            + f"predicted_matrix_type={self.__predict_matrix_type.name}"
        )


class GaussianLayer(_CompositeLayerBase):
    """A class that implements the Gaussian (mean and covariance) layer."""

    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: int,
        nmodes: int = 1,
    ) -> GaussianLayer:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : int
            The number of dimensions for the Gaussian.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        # Create the mu and sigma layers.
        mu_layer = _GaussianMeanLayer(ndim=ndim, nmodes=nmodes)
        sigma_layer = GaussianCovarianceLayer(
            matrix_decomp_type=matrix_decomp_type,
            predict_matrix_type=predict_matrix_type,
            ndim=ndim,
            nmodes=nmodes,
        )

        super().__init__(
            components=OrderedDict([("mu", mu_layer), ("sigma", sigma_layer)])
        )

    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian layer.

        Returns
        -------
        details : str
            Detailed information about the Gaussian layer.
        """
        return self.components["sigma"].extra_repr()


class GaussianMixtureLayer(_CompositeLayerBase):
    """A class that implements the Gaussian mixture layer."""

    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: int,
        nmodes: int = 1,
    ) -> GaussianMixtureLayer:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : int
            The number of dimensions for the Gaussian.

        nmodes : int
            The number of modes. Only useful for mixture models.
        """
        # Create the mixture coefficient, mu, and sigma layers.
        mix_layer = _GaussianMixCoeffLayer(ndim=ndim, nmodes=nmodes)
        mu_layer = _GaussianMeanLayer(ndim=ndim, nmodes=nmodes)
        sigma_layer = GaussianCovarianceLayer(
            matrix_decomp_type=matrix_decomp_type,
            predict_matrix_type=predict_matrix_type,
            ndim=ndim,
            nmodes=nmodes,
        )

        super().__init__(
            components=OrderedDict(
                [("mix", mix_layer), ("mu", mu_layer), ("sigma", sigma_layer)]
            )
        )

    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian mixture layer.

        Returns
        -------
        details : str
            Detailed information about the Gaussian mixture layer.
        """
        return self.components["sigma"].extra_repr()
