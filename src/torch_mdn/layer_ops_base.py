from abc import ABC, abstractmethod
import math
from typing import Tuple

from pydantic import validate_arguments, PositiveInt
import torch


class IntraLayerOperationBase(ABC):
    @validate_arguments
    def __init__(
        self,
        free_params_shape: Tuple[PositiveInt, PositiveInt],
        predicted_vars_shape: Tuple[PositiveInt, ...],
    ) -> None:
        super().__init__()
        self._free_params_shape = tuple(free_params_shape)
        self._predicted_vars_shape = tuple(predicted_vars_shape)

    @abstractmethod
    def _compute_free_params_activation(
        self, free_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies an activation function to the (reshaped) free parameters.

        Parameters
        ----------
        free_params : torch.Tensor
            A reshaped tensor.

        Returns
        -------
        res : torch.Tensor
            The free parameters after an activation function has been applied.
        """

    def _reshape_free_params(self, free_params: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input free parameters.

        Parameters
        ----------
        free_params : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor.
        """
        return free_params.view((-1,) + self._free_params_shape)

    def _reshape_predicted_vars(self, predicted_vars: torch.Tensor) -> torch.Tensor:
        """
        Reshapes the input predicted variables.

        Parameters
        ----------
        predicted_vars : torch.Tensor
            The predicted variables.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor.
        """
        return predicted_vars.reshape((-1,) + self._predicted_vars_shape)

    def compute_expected_input_shape(
        self, batch_size: PositiveInt = 1
    ) -> Tuple[int, int]:
        """
        Computes the expected input shape of the free parameters for the derived class.

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
        return (batch_size, math.prod(self._free_params_shape))

    def compute_forward_operations(self, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass on the input free parameters for the derived class.

        Parameters
        ----------
        free_params : torch.Tensor
            The linear output sub-vector from the previous layer (typically the last linear
            layer of the neural network).

        Returns
        -------
        res : torch.Tensor
            The result of performing a forward pass on the input variable.
        """
        # Reshape the input tensor.
        reshaped_free_params = self._reshape_free_params(free_params=free_params)

        # Then apply the activation function.
        res = self._compute_free_params_activation(free_params=reshaped_free_params)
        return res

    def compute_forward_shape(
        self, batch_size: PositiveInt = 1
    ) -> Tuple[PositiveInt, PositiveInt, PositiveInt]:
        """
        Returns the shape of a tensor after a forward pass.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        res : Tuple[int, int, int]
            The shape after a forward pass.
        """
        return (batch_size,) + self._free_params_shape

    @abstractmethod
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
            The linear output sub-vector from the previous layer (typically the last linear
            layer of the neural network).

        Returns
        -------
        loss_term : torch.Tensor
            The loss term.
        """

    @abstractmethod
    def compute_predict_operations(self, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes a prediction for the derived class given the input free parameters.

        Parameters
        ----------
        free_params : torch.Tensor
            The linear output sub-vector from the previous layer (typically the last linear
            layer of the neural network).

        Returns
        -------
        res : torch.Tensor
            The predicted precision or covariance matrix.
        """

    def compute_prediction_shape(
        self, batch_size: PositiveInt = 1
    ) -> Tuple[PositiveInt, ...]:
        """
        Returns the shape of a tensor after a prediction.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        res : Tuple[int, int, int]
            The shape after a prediction.
        """
        return tuple((batch_size,) + self._predicted_vars_shape)
