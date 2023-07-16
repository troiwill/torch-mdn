from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from pydantic import validate_arguments, PositiveInt
import torch
import torch_mdn.utils
from torch_mdn.layer_ops_base import IntraLayerOperationBase


class MixtureLayerBase(ABC):
    """An abstract class for implementing a mixture layer."""

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def __init__(self, layer_operations: List[IntraLayerOperationBase]) -> None:
        """
        Parameters
        ----------
        layer_operations : List[IntraLayerOperationBase]
            The layer operations.
        """
        super().__init__()  # type: ignore

        # Copy the layer operations.
        self._layer_operations = layer_operations.copy()

        # Compute the indices for the free parameters.
        operation_indices: List[torch.Tensor] = list()
        for operation in self._layer_operations:
            _, width = operation.compute_expected_input_shape(batch_size=1)
            start_index = 0
            if len(operation_indices) > 0:
                start_index = operation_indices[-1][-1] + 1
            operation_indices.append(
                torch_mdn.utils.create_torch_indices(
                    list(range(start_index, start_index + width))
                )
            )

        self._op_free_param_indices = operation_indices.copy()

    def _extract_operation_free_params(
        self, free_params: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Splits the input tensor according to the generated slices.

        Parameters
        ----------
        free_params : torch.Tensor
            The input tensor to extract the sub-free parameters from.

        Returns
        -------
        res : Tuple[torch.Tensor, ...]
            A tuple of split tensors.
        """
        return tuple(
            [
                free_params[:, param_indices]
                for param_indices in self._op_free_param_indices
            ]
        )

    @validate_arguments
    def expected_input_shape(self, batch_size: PositiveInt = 1) -> Tuple[int, int]:
        """
        Computes the expected input shape of the free parameters from the previous layer
        (typically the previous layer is a Linear layer).

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        result : Tuple[int, int]
            The expected input shape.
        """
        dim1 = sum(
            [
                op.compute_expected_input_shape(batch_size=batch_size)[1]
                for op in self._layer_operations
            ]
        )
        return tuple([batch_size, dim1])

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def forward(
        self, free_params: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Computes the forward pass using the free parameters.

        Parameters
        ----------
        free_params : torch.Tensor
            The linear output from the previous layer (typically the last linear
            layer of the neural network).

        Returns
        -------
        forward_output : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The output tensor or tensors from computing a forward pass operation.
        """
        # Split the input free parameters into sub-vectors.
        op_free_params_tuple = self._extract_operation_free_params(
            free_params=free_params
        )

        # Compute the forward passes.
        forward_output = tuple(
            [
                op.compute_forward_operations(free_params=op_free_params)
                for op, op_free_params in zip(
                    self._layer_operations, op_free_params_tuple
                )
            ]
        )

        if len(forward_output) == 1:
            forward_output = forward_output[0]
        return forward_output

    def forward_shapes(
        self, batch_size: PositiveInt = 1
    ) -> Union[Tuple[PositiveInt, ...], Tuple[Tuple[PositiveInt, ...], ...]]:
        """
        Computes the shape of the output after a forward pass.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        forward_shape : Union[Tuple[PositiveInt, ...], Tuple[Tuple[PositiveInt, ...], ...]]
            The output shape.
        """
        return tuple(
            (
                op.compute_forward_shape(batch_size=batch_size)
                for op in self._layer_operations
            )
        )

    @abstractmethod
    def loss(self, target: torch.Tensor, free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss given the target and the free parameters.

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

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(
        self, free_params: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Computes a prediction using the free parameters.

        Parameters
        ----------
        free_params : torch.Tensor
            The linear output from the previous layer (typically the last linear
            layer of the neural network).

        Returns
        -------
        prediction : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The output tensor or tensors from computing a prediction operation.
        """
        # Split the input free parameters into sub-vectors.
        op_free_params_tuple = self._extract_operation_free_params(
            free_params=free_params
        )

        # Compute the prediction(s).
        prediction = tuple(
            [
                op.compute_predict_operations(free_params=op_free_params)
                for op, op_free_params in zip(
                    self._layer_operations, op_free_params_tuple
                )
            ]
        )

        if len(prediction) == 1:
            prediction = prediction[0]
        return prediction

    def prediction_shapes(
        self, batch_size: PositiveInt = 1
    ) -> Union[Tuple[PositiveInt, ...], Tuple[Tuple[PositiveInt, ...], ...]]:
        """
        Computes the shape of the output after a prediction.

        Parameters
        ----------
        batch_size : PositiveInt
            The expected batch size of the input to the forward method. This argument is optional
            and is mainly useful for debugging.

        Returns
        -------
        prediction_shape : Union[Tuple[PositiveInt, ...], Tuple[Tuple[PositiveInt, ...], ...]]
            The output shape.
        """
        return tuple(
            [
                op.compute_prediction_shape(batch_size=batch_size)
                for op in self._layer_operations
            ]
        )


class MDN(torch.nn.Module):
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def __init__(
        self, param_model: torch.nn.Module, dist_layer: MixtureLayerBase
    ) -> None:
        super().__init__()  # type: ignore

        self._param_model = param_model
        self._dist_layer = dist_layer

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def forward(
        self, features: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Performs a forward pass given the input features. Since there is a dedicated method for
        training the model (via self.loss(...)), this method performs the same function as
        self.predict(...).

        Parameters
        ----------
        features : torch.Tensor
            The input features to the model.

        Returns
        -------
        result : torch.Tensor
            The output tensor from a forward pass.
        """
        param_vec = self._param_model.forward(features)
        return self._dist_layer.forward(free_params=param_vec)

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(
        self, features: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Performs a prediction given the input features.

        Parameters
        ----------
        features : torch.Tensor
            The input features to the model.

        Returns
        -------
        result : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            The output tensor or tensors from a prediction.
        """
        with torch.no_grad():
            param_vec = self._param_model.forward(features)
            return self._dist_layer.predict(free_params=param_vec)

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def loss(self, target: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss of the network given the target and input features.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor.

        features : torch.Tensor
            The input features to the model.

        Returns
        -------
        loss : torch.Tensor
            The loss tensor.
        """
        param_vec = self._param_model.forward(features)
        return self._dist_layer.loss(target=target, free_params=param_vec)
