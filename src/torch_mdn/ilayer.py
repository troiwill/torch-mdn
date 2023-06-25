from abc import ABCMeta, abstractmethod
from typing import OrderedDict, Tuple, List

from pydantic import validate_arguments, PositiveInt
from torch import Tensor
import torch.nn as nn


class ILayerComponent(nn.Module, metaclass=ABCMeta):
    """An interface that provides methods for implementing individual "components" for a layer."""

    def __init__(self) -> None:
        super().__init__()  # type: ignore

    @abstractmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def apply_activation(self, x: Tensor) -> Tensor:
        """
        An abstract method that applies an activation function to the input tensor. Usually called
        within the forward(...) function.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The result of applying the activation function to the tensor `x`.
        """

    @abstractmethod
    @validate_arguments
    def expected_input_shape(self, batch_size: PositiveInt = 1) -> Tuple[int, int]:
        """
        An abstract method for computing the expected input shape of the free parameters used to
        build some component.

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

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def forward(self, x: Tensor) -> Tensor:
        """
        A method that is called when applying a layer component. `x` is usually the output of a
        feed-forward network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The result of "computing" a particular layer component.
        """
        # Reshape the input tensor.
        x = self.reshape_input(x)

        # Then apply the activation function.
        x = self.apply_activation(x)
        return x

    @validate_arguments(config={"arbitrary_types_allowed": True})
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
        return self.forward(x=x)

    @abstractmethod
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def reshape_input(self, x: Tensor) -> Tensor:
        """
        An abstract method that reshapes the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : torch.Tensor
            The reshaped tensor.
        """

    @abstractmethod
    def extra_repr(self) -> str:
        """
        Returns detailed information about the layer component.

        Returns
        -------
        details : str
            Detailed information about the layer component.
        """


class ICompositeLayer(nn.Module, metaclass=ABCMeta):
    """An abstract class for combining multiple layer components."""

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def __init__(self, components: OrderedDict[str, ILayerComponent]) -> None:
        """
        Parameters
        ----------
        components : OrderedDict[str, ILayerComponent]
            The components for a multiple layer components.
        """
        super().__init__()  # type: ignore

        # for name, c in components.items():
        #     if not isinstance(c, ILayerComponent):
        #         raise Exception(f"Component name '{name}' must be derived from ILayerComponent, but it is not.")

        # An ordered dictionary of layer components.
        self._components = components.copy()

        self._input_slices: List[Tuple[int, int]] = list()
        for _, layer in self._components.items():
            _, width = layer.expected_input_shape()
            start_idx = 0
            if len(self._input_slices) > 0:
                _, start_idx = self._input_slices[-1]
            self._input_slices.append((start_idx, start_idx + width))

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Computes the forward for each component object.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        Returns
        -------
        res : Tuple[torch.Tensor, ...]
            The forward outputs of each component in a tuple.
        """
        # Split the input tensor.
        split_x = self.split_input(x=x)

        # Perform regular forward pass.
        return tuple(
            [comp.forward(x_i) for x_i, comp in zip(split_x, self._components.values())]
        )

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def predict(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Computes the prediction for each component object.

        Parameters
        ----------
        x : torch.Tensor
            The linear output from the neural network.

        Returns
        -------
        res : Tuple[torch.Tensor, ...]
            The prediction outputs of each component in a tuple.
        """
        # Split the input tensor.
        split_x = self.split_input(x=x)

        # Perform prediction.
        return tuple(
            [comp.predict(x_i) for x_i, comp in zip(split_x, self._components.values())]
        )

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def split_input(self, x: Tensor) -> Tuple[Tensor, ...]:
        """
        Splits the input tensor according to the generated slices.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        res : Tuple[torch.Tensor, ...]
            A tuple of split tensors.
        """
        return tuple(
            [x[:, in_slice[0] : in_slice[1]] for in_slice in self._input_slices]
        )

    @abstractmethod
    def extra_repr(self) -> str:
        """
        Returns detailed information about the Gaussian layer.

        Returns
        -------
        details : str
            Detailed information about the Gaussian layer.
        """
