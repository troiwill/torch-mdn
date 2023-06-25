from collections import OrderedDict
from enum import Enum
import math
from typing import OrderedDict, Tuple

from pydantic import validate_arguments, PositiveInt
import torch
from torch_mdn.ilayer import ILayerComponent, ICompositeLayer
import torch_mdn.utils as utils
from torch import Tensor
import torch.nn.functional as F


class MatrixDecompositionType(Enum):
    """Enum for specifying the types of decomposed Gaussian covariance matrices."""

    diagonal = "Diagonal Decomposition"
    """Decomposition of a diagonal covariance matrix."""

    full_LDL = "Full Matrix LDL Decomposition"
    """LDL decomposition of a full covariance matrix."""

    full_UU = "Full Matrix Cholesky UU Decomposition"
    """UU decomposition of a full covariance matrix."""


class MatrixPredictionType(Enum):
    """Enum for specifying how to output a covariance matrix during prediction/testing."""

    precision = "Precision"
    """Output the precision/infomation matrix."""

    covariance = "Covariance"
    """Output the covariance matrix."""


class _IGaussianLayerComponent(ILayerComponent):
    """A base class that provides methods for implementing individual Gaussian Mixture components
    (that is, the covariance, mixture coefficients, and mean).
    """

    @validate_arguments
    def __init__(self, ndim: PositiveInt, nmodes: PositiveInt) -> None:
        """
        Parameters
        ----------
        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.
        """
        super().__init__()

        self._ndim = ndim
        self._nmodes = nmodes

    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian layer.

        Returns
        -------
        details : str
            Detailed information about the Gaussian layer.
        """
        return f"nmodes={self._nmodes}, ndims={self._ndim}"

    @property
    def ndim(self) -> int:
        """
        Returns the number of dimensions in the Gaussian distribution.

        Returns
        -------
        ndim : int
            The number of dimensions in the Gaussian distribution.
        """
        return self._ndim

    @property
    def nmodes(self) -> int:
        """
        Returns the number of modes in a mixture distribution.

        Returns
        -------
        nmodes : int
            The number of modes in a mixture distribution.
        """
        return self._nmodes


class _GaussianMatrixComponent(_IGaussianLayerComponent):
    """A class that implements the Gaussian matrix."""

    @validate_arguments
    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: PositiveInt,
        nmodes: PositiveInt,
    ) -> None:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)

        # Determine the matrix decomposition type.
        self._matrix_decomp_type = matrix_decomp_type
        if self._matrix_decomp_type == MatrixDecompositionType.full_UU:
            num_mat_params = utils.num_tri_matrix_params_per_mode(self.ndim, False)
            self._cpm_dist_shape = (self.nmodes, num_mat_params)
        else:
            raise NotImplementedError(
                f"The matrix decomposition type {self._matrix_decomp_type.name} is not implemented."
            )

        # Determine the matrix prediction/inference type.
        self._predict_matrix_type = predict_matrix_type

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        if self._matrix_decomp_type == MatrixDecompositionType.full_UU:
            diag_indices = utils.diag_indices_tri(ndim=self.ndim, is_lower=False)
            x[:, :, diag_indices] = (
                F.elu(x[:, :, diag_indices], alpha=1.0) + 1.0 + utils.epsilon()
            )
        else:
            raise NotImplementedError(
                f"Matrix decomposition type {self._matrix_decomp_type.name} is not implemented."
            )

        return x

    @validate_arguments
    def expected_input_shape(self, batch_size: PositiveInt = 1) -> Tuple[int, int]:
        """
        Computes the expected input shape of the free parameters used to build the
        covariance/precision matrix (or matrices). This function can be used to compute the number
        of output features from the previous linear layer as `math.prod(cpm_input_shape)` when
        `batch_size = 1`.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Result
        ------
        res : Tuple[int, int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self._cpm_dist_shape))

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        if self._predict_matrix_type in [
            MatrixPredictionType.covariance,
            MatrixPredictionType.precision,
        ]:
            x = utils.to_triangular_matrix(self.ndim, x, False)
            x = utils.torch_matmul_4d(x.transpose(-2, -1), x)

            if self._predict_matrix_type == MatrixPredictionType.covariance:
                x = torch.inverse(x)

        else:
            raise NotImplementedError(
                f"The prediction type {self._predict_matrix_type.name} is not supported."
            )

        return x

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        return x.reshape((-1,) + self._cpm_dist_shape)

    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian covariance layer.

        Returns
        -------
        details : str
            Detailed information about the Gaussian covariance layer.
        """
        return (
            f"{super().extra_repr()}, "
            + f"matrix_decomposition={self._matrix_decomp_type.name}, "
            + f"predicted_matrix_type={self._predict_matrix_type.name}"
        )


class _GaussianMeanComponent(_IGaussianLayerComponent):
    """A class for implementing operations for a Gaussian mean."""

    @validate_arguments
    def __init__(self, ndim: PositiveInt, nmodes: PositiveInt) -> None:
        """
        Parameters
        ----------
        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)
        self._mu_dist_shape = (self.nmodes, self.ndim)

    @validate_arguments(config={"arbitrary_types_allowed": True})
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

    @validate_arguments
    def expected_input_shape(self, batch_size: PositiveInt = 1) -> Tuple[int, int]:
        """
        Computes the expected input shape of the free parameters used to build the mean. This
        function can be used to compute the number of output features from the previous linear
        layer as `math.prod(mu_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        res : Tuple[int, int]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self._mu_dist_shape))

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        return x.reshape((-1,) + self._mu_dist_shape)

    def extra_repr(self) -> str:
        """
        Not Implemented.
        """
        raise NotImplementedError("The extra_repr() function was not implemented.")


class _GaussianMixCoeffComponent(_IGaussianLayerComponent):
    """A class for implementing operations for Gaussian mixture coefficients.

    WARNING: This class should never be used standalone.
    """

    @validate_arguments
    def __init__(self, ndim: PositiveInt, nmodes: PositiveInt) -> None:
        """
        Parameters
        ----------
        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.
        """
        super().__init__(ndim=ndim, nmodes=nmodes)
        self._mix_dist_shape = (self.nmodes, 1)

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        max_x = torch.max(x, dim=1, keepdim=True).values  # type: ignore
        x = x - max_x  # type: ignore
        x = F.softmax(x, dim=1)  # type: ignore
        return x

    @validate_arguments
    def expected_input_shape(self, batch_size: PositiveInt = 1) -> Tuple[int, int]:
        """
        Computes the expected input shape of the free parameters used to build the mixture
        weights. This function can be used to compute the number of output features from the
        previous linear layer as `math.prod(mix_input_shape)` when `batch_size = 1`.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        res : Tuple[int, ...]
            The expected input shape to this layer.
        """
        return (batch_size, math.prod(self._mix_dist_shape))

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        return x.reshape((-1,) + self._mix_dist_shape)

    def extra_repr(self) -> str:
        """
        Not Implemented.
        """
        raise NotImplementedError("The extra_repr() function was not implemented.")


class GaussianMatrixLayer(_GaussianMatrixComponent):
    """A class that implements the Gaussian covariance/precision matrix layer."""

    @validate_arguments
    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: PositiveInt,
    ) -> None:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : PositiveInt
            The number of dimensions for the Gaussian.
        """
        super().__init__(
            matrix_decomp_type=matrix_decomp_type,
            predict_matrix_type=predict_matrix_type,
            ndim=ndim,
            nmodes=1,
        )


class GaussianLayer(ICompositeLayer):
    """A class that implements the Gaussian (mean and covariance) layer."""

    @validate_arguments
    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: PositiveInt,
    ) -> None:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : PositiveInt
            The number of dimensions for the Gaussian.
        """
        # Create the mu and sigma layers.
        nmodes: int = 1
        mu_layer = _GaussianMeanComponent(ndim=ndim, nmodes=nmodes)
        sigma_layer = _GaussianMatrixComponent(
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
        return self._components["sigma"].extra_repr()


class GaussianMixtureLayer(ICompositeLayer):
    """A class that implements the Gaussian mixture layer."""

    @validate_arguments
    def __init__(
        self,
        matrix_decomp_type: MatrixDecompositionType,
        predict_matrix_type: MatrixPredictionType,
        ndim: PositiveInt,
        nmodes: PositiveInt = 2,
    ) -> None:
        """
        Parameters
        ----------
        matrix_decomp_type : MatrixDecompositionType (enum)
            The covariance/precision matrix decomposition type.

        predict_matrix_type : MatrixPredictionType (enum)
            The prediction matrix type.

        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.
        """
        # Sanity check.
        if nmodes < 2:
            raise ValueError(f"nmodes must be at least 2, but got {nmodes}.")

        # Create the mixture coefficient, mu, and sigma layers.
        mix_layer = _GaussianMixCoeffComponent(ndim=ndim, nmodes=nmodes)
        mu_layer = _GaussianMeanComponent(ndim=ndim, nmodes=nmodes)
        sigma_layer = _GaussianMatrixComponent(
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
        return self._components["sigma"].extra_repr()
