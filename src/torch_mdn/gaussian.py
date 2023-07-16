from enum import Enum
import math
from typing import Literal, Union

from pydantic import validate_arguments, PositiveInt
import torch
from torch_mdn.decompose import (
    MatrixDecompositionBase,
    MatrixDecompositionType,
    CholeskyMatrixDecomposition,
)
from torch_mdn.layer_ops_base import IntraLayerOperationBase
from torch_mdn.mdn import MixtureLayerBase


class MatrixPredictionType(Enum):
    """Enum for specifying how to output a covariance matrix during prediction/testing."""

    precision = "Precision"
    """Output the precision/infomation matrix."""

    covariance = "Covariance"
    """Output the covariance matrix."""


class _GaussianMatrixOperations(IntraLayerOperationBase):
    """A class for implementing operations for a Gaussian matrix."""

    def __init__(
        self,
        ndim: PositiveInt,
        nmodes: PositiveInt,
        prediction_type: MatrixPredictionType,
        decomposition_impl_or_type: Union[
            MatrixDecompositionBase, MatrixDecompositionType
        ],
    ) -> None:
        """
        Parameters
        ----------
        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        nmodes : PositiveInt
            The number of modes in the mixture. Only useful in mixture modeling.

        prediction_type : MatrixPredictionType (enum)
            The prediction type for the matrix.

        decomposition_impl_or_type : Union[MatrixDecompositionBase, MatrixDecompositionType]
            A MatrixDecompositionType or a subclass of MatrixDecompositionBase.
        """
        # Check if the decomposition variable is a MatrixDecompositionType enum.
        if isinstance(decomposition_impl_or_type, MatrixDecompositionType):
            self._decomposition = self._get_decompose_method(
                decomposition_type=decomposition_impl_or_type, ndim=ndim, nmodes=nmodes
            )
        else:
            self._decomposition = decomposition_impl_or_type

        # Call init for super.
        super().__init__(
            free_params_shape=self._decomposition.expected_decomposed_shape,
            predicted_vars_shape=(ndim, ndim) if nmodes == 1 else (nmodes, ndim, ndim),
        )

        # Set other variables.
        self._prediction_type = prediction_type
        self._half_log_twopi = (
            self._decomposition.ndim * 0.5 * torch.log(torch.tensor(2 * math.pi))
        )

    def _compute_free_params_activation(
        self, free_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Uses the decomposition method to apply an activation function to the (reshaped) free
        parameters.

        Parameters
        ----------
        free_params : torch.Tensor
            An appropriately shaped tensor that corresponds to the free parameters of a Gaussian
            (precision or covariance) matrix.

        Returns
        -------
        res : torch.Tensor
            The matrix free parameters after an activation function has been applied.
        """
        return self._decomposition.apply_activation(free_params)

    def _get_decompose_method(
        self,
        decomposition_type: MatrixDecompositionType,
        ndim: PositiveInt,
        nmodes: PositiveInt,
    ) -> MatrixDecompositionBase:
        # Get the implementation for the matrix decomposition.
        decomposition_impl = None
        if decomposition_type == MatrixDecompositionType.cholesky:
            decomposition_impl = CholeskyMatrixDecomposition(ndim=ndim, nmodes=nmodes)
        else:
            raise NotImplementedError(
                f"Received a matrix decomposition type named {decomposition_type.name}, but it is unhandled."
            )
        return decomposition_impl

    def compute_loss_term_operations(
        self, target: torch.Tensor, free_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss term before applying a reduction function (e.g., torch.mean(...)).

        Parameters
        ----------
        target : torch.Tensor
            The residual or error.

        free_params : torch.Tensor
            The appropriately shaped free parameters for the matrix after a forward pass.

        Returns
        -------
        loss : torch.Tensor
            The loss term.
        """
        loss_term = self._decomposition.compute_loss(residual=target, matrix_free_params=free_params)
        return loss_term

    def compute_predict_operations(self, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes a prediction given the free parameters to produce a precision or covariance matrix.

        Parameters
        ----------
        free_params : torch.Tensor
            The free parameters that correspond to the Gaussian matrix.

        Returns
        -------
        result : torch.Tensor
            The predicted precision or covariance matrix.
        """
        # Perform a forward pass on the input.
        free_params = self.compute_forward_operations(free_params=free_params)

        # Use the decomposition method to produce the precision matrix.
        precision_mat = self._decomposition.decomposition_product(
            matrix_free_params=free_params
        )

        # Determine what to return.
        result = None
        if self._prediction_type == MatrixPredictionType.precision:
            result = precision_mat

        elif self._prediction_type == MatrixPredictionType.covariance:
            result = torch.inverse(precision_mat)

        else:  # Ensure we raise an exception if we forgot to handle a prediction case.
            raise RuntimeError(
                f"Unhandled matrix prediction type: {self._prediction_type.name}."
            )

        # Reshape the output tensor.
        result = self._reshape_predicted_vars(result)
        return result


class _GaussianMeanOperations(IntraLayerOperationBase):
    """A class for implementing operations for a Gaussian mean."""

    def __init__(
        self,
        ndim: PositiveInt,
        nmodes: PositiveInt,
    ) -> None:
        super().__init__(
            free_params_shape=(nmodes, ndim),
            predicted_vars_shape=(ndim, 1) if nmodes == 1 else (nmodes, ndim, 1),
        )

    def _compute_free_params_activation(
        self, free_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the input tensor as is.

        Parameters
        ----------
        free_params : torch.Tensor
            An appropriately shaped tensor that corresponds to the free parameters of a Gaussian
            mean.

        Returns
        -------
        res : torch.Tensor
            The unaltered input to the method.
        """
        return free_params

    def compute_loss_term_operations(
        self, target: torch.Tensor, free_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the loss term before applying a reduction function (e.g., torch.mean(...)).

        Parameters
        ----------
        target : torch.Tensor
            The residual or error.

        free_params : torch.Tensor
            The appropriately shaped free parameters for the matrix after a forward pass.

        Returns
        -------
        loss : torch.Tensor
            The loss term.
        """
        return torch.tensor(0.0)

    def compute_predict_operations(self, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes a prediction given the free parameters to produce the mean.

        Parameters
        ----------
        free_params : torch.Tensor
            The free parameters that correspond to the Gaussian mean.

        Returns
        -------
        result : torch.Tensor
            The predicted mean.
        """
        result = self.compute_forward_operations(free_params=free_params)

        # Reshape the output tensor.
        result = self._reshape_predicted_vars(result)
        return result


class GaussianMatrixLayer(MixtureLayerBase):
    """A class that implements the Gaussian matrix layer."""

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        ndim: PositiveInt,
        prediction_type: Literal[
            MatrixPredictionType.precision, MatrixPredictionType.covariance
        ],
        decomposition_impl_or_type: Union[
            MatrixDecompositionBase, MatrixDecompositionType
        ],
    ) -> None:
        """
        Parameters
        ----------
        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        prediction_type : MatrixPredictionType (enum)
            The prediction type for the matrix.

        decomposition_type : Union[MatrixDecompositionBase, MatrixDecompositionType]
            A subclass of MatrixDecompositionBase or a MatrixDecompositionType.
        """
        super().__init__(
            layer_operations=list(
                [
                    _GaussianMatrixOperations(
                        ndim=ndim,
                        nmodes=1,
                        prediction_type=prediction_type,
                        decomposition_impl_or_type=decomposition_impl_or_type,
                    )
                ]
            )
        )

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def loss(self, target: torch.Tensor, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for a Gaussian matrix given its free parameters and the target.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor.

        free_params : torch.Tensor
            The free parameters for the matrix after a forward pass.

        Returns
        -------
        result : Tensor
            The result of computing the loss, including applying a reduction function
            (e.g., torch.mean).

        """
        # Compute the matrix parameters using a forward pass.
        matrix_params = self.forward(free_params=free_params)
        assert isinstance(matrix_params, torch.Tensor)

        # Compute the loss terms for matrix.
        batch_size, _, _ = matrix_params.shape
        loss_terms = self._layer_operations[0].compute_loss_term_operations(
            target=target.view(batch_size, -1, 1), free_params=matrix_params
        )

        # Apply reduction.
        return -1.0 * torch.mean(loss_terms)


class GaussianLayer(MixtureLayerBase):
    """A class that implements the Gaussian (mean and covariance) layer."""

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        ndim: PositiveInt,
        prediction_type: Literal[
            MatrixPredictionType.precision, MatrixPredictionType.covariance
        ],
        decomposition_impl_or_type: Union[
            MatrixDecompositionBase, MatrixDecompositionType
        ],
    ) -> None:
        """
        Parameters
        ----------
        ndim : PositiveInt
            The number of dimensions for the Gaussian.

        prediction_type : MatrixPredictionType (enum)
            The prediction type for the matrix.

        decomposition_type : Union[MatrixDecompositionBase, MatrixDecompositionType]
            A subclass of MatrixDecompositionBase or a MatrixDecompositionType.
        """
        # Create the mu and sigma layers.
        nmodes: int = 1
        super().__init__(
            layer_operations=list(
                [
                    _GaussianMeanOperations(ndim=ndim, nmodes=nmodes),
                    _GaussianMatrixOperations(
                        ndim=ndim,
                        nmodes=nmodes,
                        prediction_type=prediction_type,
                        decomposition_impl_or_type=decomposition_impl_or_type,
                    ),
                ]
            )
        )

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def loss(self, target: torch.Tensor, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for a Gaussian mean and matrix given its free parameters and the target.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor.

        free_params : torch.Tensor
            The free parameters for the matrix after a forward pass.

        Returns
        -------
        result : Tensor
            The result of computing the loss, including applying a reduction function
            (e.g., torch.mean).

        """
        # Compute the matrix parameters using a forward pass.
        mu_params, matrix_params = self.forward(free_params=free_params)

        # Compute the residual.
        batch_size, _, ndim = mu_params.shape
        data_shape = (batch_size, ndim, 1)
        residual = target.view(data_shape) - mu_params.view(data_shape)

        # Compute the loss terms for matrix.
        loss_terms = self._layer_operations[1].compute_loss_term_operations(
            target=residual, free_params=matrix_params
        )

        # Apply reduction.
        return -1.0 * torch.mean(loss_terms)
