from abc import abstractmethod, ABC
from enum import Enum
import math
from typing import Tuple

from pydantic import validate_arguments, PositiveInt
import torch
import torch.nn.functional
import torch_mdn.utils


class MatrixDecompositionType(Enum):
    """Enum for specifying the types of decomposed matrices."""

    cholesky = "Cholesky (U^T * U) Decomposition"
    """Cholesky U^T * U decomposition of a full matrix."""


class MatrixDecompositionBase(ABC):
    """An abstract base class for describing matrix decomposition operations."""

    @validate_arguments
    def __init__(
        self,
        ndim: PositiveInt,
        nmodes: PositiveInt,
        expected_decomposed_shape: Tuple[int, int],
    ) -> None:
        """
        ndim : PositiveInt
            The number of dimensions of the Gaussian distribution.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.

        expected_decomposed_shape : Tuple[int, int]
            The expected shape of the matrix free parameters (the decomposed matrix).
        """
        super().__init__()

        self._ndim = ndim
        self._nmodes = nmodes
        self._expected_decomposed_shape = tuple(expected_decomposed_shape)

    @abstractmethod
    def apply_activation(self, matrix_free_params: torch.Tensor) -> torch.Tensor:
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

    @abstractmethod
    def compute_loss(self, residual: torch.Tensor, matrix_free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss function using Cholesky decomposition.

        Parameters
        ----------
        residual : torch.Tensor
            The residual/error that is computed using the target from the data.

        matrix_free_params : torch.Tensor
            The free parameters that are used to build the covariance/precision matrix.

        Returns
        -------
        res : torch.Tensor
            The result of -0.5 * log(2 * pi) + ln(det(Sigma)) - (0.5 * (x^T) * Sigma * (x)).
        """

    @abstractmethod
    def decomposition_product(self, matrix_free_params: torch.Tensor) -> torch.Tensor:
        """
        Performs the product of the decomposed matrix or matrices within the input tensor.

        Parameters
        ----------
        matrix_free_params : torch.Tensor
            The free parameters of the matrix or matrices.

        Returns
        -------
        product : torch.Tensor
            The product of the decomposed matrix free parameters.
        """

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the decomposition method.

        Returns
        -------
        name : str
            The name of the decomposition method.
        """

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns a string describing how to reproduce this decomposition method.

        Returns
        -------
        repr : str
            A string describing how to reproduce this decomposition method.
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string describing the object.

        Returns
        -------
        description : str
            A description of the object.
        """

    @property
    def expected_decomposed_shape(self) -> Tuple[int, int]:
        """
        The expected shape of the matrix free parameters (the decomposed matrix).
        """
        return self._expected_decomposed_shape

    @property
    def ndim(self) -> PositiveInt:
        """
        The number of dimensions in the matrix.
        """
        return self._ndim

    @property
    def nmodes(self) -> PositiveInt:
        """
        The number of modes in the distribution.
        """
        return self._nmodes


class CholeskyMatrixDecomposition(MatrixDecompositionBase):
    """An implementation of the Cholesky matrix decomposition using upper triangular matrices
    (that is, U^T * U)."""

    @validate_arguments
    def __init__(self, ndim: PositiveInt, nmodes: PositiveInt) -> None:
        """
        ndim : PositiveInt
            The number of dimensions of the Gaussian distribution.

        nmodes : PositiveInt
            The number of modes. Only useful for mixture models.
        """
        num_mat_params = torch_mdn.utils.num_tri_matrix_params_per_mode(ndim, False)
        super().__init__(
            ndim=ndim, nmodes=nmodes, expected_decomposed_shape=(nmodes, num_mat_params)
        )

        self._u_diag_indices = torch.tensor(
            torch_mdn.utils.diag_indices_tri(ndim=ndim, is_lower=False),
            dtype=torch.int64,
        )

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def apply_activation(self, matrix_free_params: torch.Tensor) -> torch.Tensor:
        """
        Returns the input free parameters after an activation function has been applied.

        Parameters
        ----------
        matrix_free_params : torch.Tensor
            The reshaped, sub-tensor output from the neural network that corresponds to the
            covariance of the Gaussian.

        Returns
        -------
        res : torch.Tensor
            The free parameters after an activation function has been applied.
        """
        # diag_indices = utils.diag_indices_tri(ndim=self.ndim, is_lower=False)
        matrix_free_params[:, :, self._u_diag_indices] = (
            torch.nn.functional.elu(
                matrix_free_params[:, :, self._u_diag_indices], alpha=1.0
            )
            + 1.0
            + torch_mdn.utils.epsilon()
        )
        return matrix_free_params
    
    @validate_arguments(config={"arbitrary_types_allowed": True})
    def compute_loss(self, residual: torch.Tensor, matrix_free_params: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss function using Cholesky decomposition.

        Parameters
        ----------
        residual : torch.Tensor
            The residual/error that is computed using the target from the data.

        matrix_free_params : torch.Tensor
            The free parameters that are used to build the covariance/precision matrix.

        Returns
        -------
        res : torch.Tensor
            The result of -0.5 * log(2 * pi) + ln(det(Sigma)) - (0.5 * (x^T) * Sigma * (x)).
        """
        # Compute the sum of log(diag_u). This is equivalent to Tr(log(U)).
        diag_u = matrix_free_params.index_select(2, self._u_diag_indices)
        ln_det_sigma = torch.log(diag_u).sum(dim=2, keepdim=True)

        # Compute x^T * U^T * U * x.
        u_mat = torch_mdn.utils.to_triangular_matrix(
            ndim=self._ndim, params=matrix_free_params, is_lower=False
        )
        residual = residual.view(-1, 1, self._ndim, 1)
        quad_sigma = torch_mdn.utils.compute_quad_sigma(u_mat=u_mat, v=residual)

        return (-0.5 * self._ndim * math.log(2.0 * math.pi)) + ln_det_sigma - (0.5 * quad_sigma)

    def decomposition_product(self, matrix_free_params: torch.Tensor) -> torch.Tensor:
        """
        Performs the product of the upper triangular matrix or matrices within the input tensor.

        Parameters
        ----------
        matrix_free_params : torch.Tensor
            The free parameters of the upper triangular matrix or matrices.

        Returns
        -------
        product : torch.Tensor
            The product.
        """
        u_mat = torch_mdn.utils.to_triangular_matrix(
            self._ndim, matrix_free_params, False
        )
        product = torch_mdn.utils.torch_matmul_4d(u_mat.transpose(-2, -1), u_mat)
        return product

    def name(self) -> str:
        """
        Returns the name of this decomposition method.

        Returns
        -------
        name : str
            The name of this decomposition method.
        """
        return MatrixDecompositionType.cholesky.value

    def __repr__(self) -> str:
        """
        Returns a string describing how to reproduce this decomposition method.

        Returns
        -------
        repr : str
            A string describing how to reproduce this decomposition method.
        """
        return f"{__class__.__name__}(ndim={self.ndim}, nmodes={self.nmodes})"

    def __str__(self) -> str:
        """
        Returns a string describing this decomposition method.

        Returns
        -------
        description : str
            A description of this decomposition method.
        """
        return f"{self.name()} for {self.nmodes} {self.ndim}-dimensional matrices."
