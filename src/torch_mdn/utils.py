"""
Module providing utility functions for other code in this package.
"""

from typing import Tuple, List

from pydantic import (
    validate_arguments,
    PositiveInt,
)
import torch
from torch import Tensor


@validate_arguments(config={"arbitrary_types_allowed": True})
def compute_quad_sigma(u_mat: Tensor, v: Tensor) -> Tensor:
    """
    Computes the quadratic (v^T) * (U^T * U) * (v) for the loss function, where U is an upper
    triangular matrix.

    Parameters
    ----------
    u_mat : torch.Tensor
        An upper triangular matrix.

    v : torch.Tensor
        A residual/error vector.

    Returns
    -------
    res : Tensor
        The result of (x^T) * Sigma * (x) using an upper triangular matrix.
    """
    # Sanity checks for upper triangular matrix.
    if u_mat.shape[:-1] != v.shape[:-1]:
        raise RuntimeError(f"u_mat.shape[:-1] ({u_mat.shape[:-1]}) != v.shape[:-1] ({v.shape[:-1]})")
    
    if u_mat.shape[-2] != u_mat.shape[-1]:
        raise RuntimeError(f"The last two dimensions are not the same: {u_mat.shape[-2:]}")

    # Compute ||U * v||^2_2, where v = (x - mu).
    ur = torch.matmul(u_mat, v)
    ur = torch.square(ur)
    ur = ur.sum(dim=2, keepdim=False)
    return ur


@validate_arguments
def create_torch_indices(indices: List[int]) -> Tensor:
    """
    Creates a PyTorch tensor of Int64 with the input list of integers.

    Parameters
    ----------
    indices : List[int]
        The list of integers.

    Returns
    -------
    res : torch.Tensor
        The indices as a tensor.
    """
    return torch.tensor(indices, dtype=torch.int64)


@validate_arguments
def diag_indices_tri(ndim: PositiveInt, is_lower: bool) -> Tuple[int, ...]:
    """
    Computers the diagonal index values for a triangular matrix.

    Parameters
    ----------
    ndim : PositiveInt
        The number of dimension of the triangular matrix. E.g., 3 for a 3 x 3 matrix.

    is_lower : bool
        Specifies if the matrix is a lower triangular (True) or upper triangular (False).

    Returns
    -------
    res : Tuple[int, ...]
        A tuple of integers representing the diagonal indices for a lower or upper triangular
        matrix.
    """
    # Stores the diagonal indices for a triangular matrix.
    diag_indices = list([0] * ndim)

    # Compute the diagonal indices only if ndim is at least 2.
    if ndim >= 2:
        start_idx, expected_end_idx = 0, int(((ndim * ndim) + ndim) * 0.5) - 1
        index_gaps = (
            list(range(2, ndim + 1, 1)) if is_lower else list(range(ndim, 1, -1))
        )

        diag_indices[0] = start_idx
        for i, index_gap in enumerate(index_gaps):
            diag_indices[i + 1] = diag_indices[i] + index_gap

        # Sanity check.
        if diag_indices[-1] != expected_end_idx:
            raise ValueError(
                f"Expected last index to be {expected_end_idx} but received {diag_indices[-1]}."
            )

    return tuple(diag_indices)


def epsilon() -> float:
    """
    Epsilon number for PyTorch for a floating-point number.

    Returns
    -------
    epsilon : float
        Returns a small floating-point number.
    """
    return float(torch.finfo(torch.float32).eps)


@validate_arguments
def num_tri_matrix_params_per_mode(ndim: PositiveInt, is_unit_tri: bool) -> int:
    """
    Compute the number of free parameters for one triangular matrix.

    Parameters
    ----------
    ndim : PositiveInt
        The number of dimensions in the data.

    is_unit_tri : bool
        Specifies if the resulting triangular matrix is unit (has a diagonal of ones) or not. If
        the matrix is unit, `is_unit_tri` must be `True`. Otherwise, it is `False`.

    Returns
    -------
    res : int
        The number of free parameters for a (unit or non-unit) triangular matrix with ndim
        dimensions.
    """
    num_params = int(ndim * (ndim + 1) * 0.5)
    if is_unit_tri:
        num_params = num_params - ndim
    return num_params


@validate_arguments(config={"arbitrary_types_allowed": True})
def to_triangular_matrix(ndim: PositiveInt, params: Tensor, is_lower: bool) -> Tensor:
    """
    Builds a triangular matrix using a set of free parameters.

    WARNING: This function only builds non-unit triangular matrices.

    Parameters
    ----------
    ndim : PositiveInt
        The number of dimensions in the data.

    params : torch.Tensor
        The free parameters from the output of a function (e.g., a neural network). The free
        parameters will be reorganized to build a non-unit triangular matrix.

    is_lower : bool
        Specifies if the resulting triangular matrix is a lower (True) or upper (False) triangular
        matrix.

    Returns
    -------
    res : torch.Tensor
        Returns a triangular matrix with dimensions (L, M, N, N), where N is ndim.
    """
    # Sanity checks.
    if len(params.size()) != 3:
        raise ValueError(
            f"len(tuple( params.size() )) must be 3, but got length {len(params.size())}."
        )

    # Allocate the triangular matrix.
    batch, nmodes, n_params = tuple(params.size())
    expected_n_params = num_tri_matrix_params_per_mode(ndim=ndim, is_unit_tri=False)
    if n_params != expected_n_params:
        raise ValueError(
            f"params.size()[2] must be {expected_n_params}, but got {n_params}."
        )

    tri_mat = torch.zeros(
        (batch, nmodes, ndim, ndim), dtype=params.dtype, device=params.device
    )

    if is_lower is True:
        i, j = torch.tril_indices(ndim, ndim)
    else:
        i, j = torch.triu_indices(ndim, ndim)
    tri_mat[:, :, i, j] = params
    return tri_mat


@validate_arguments(config={"arbitrary_types_allowed": True})
def torch_matmul_4d(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Performs matrix-matrix multiplication for two 4D matrices, where the last two dimensions of
    each matrix is (N,N).

    Parameters
    ----------
    tensor1 : torch.Tensor
        The first 4-D matrix.

    tensor2 : torch.Tensor
        The second 4-D matrix.

    Returns
    -------
    res : torch.Tensor
        The matrix-matrix multiplication of a and b, where the third and fourth dimensions are
        multiplied.
    """
    # Sanity checks.
    if len(tensor1.size()) != 4:
        raise ValueError(
            f"tensor1.size() must have length 4, but got length {len(tensor1.size())}."
        )
    if len(tensor2.size()) != 4:
        raise ValueError(
            f"tensor2.size() must have length 4, but got length {len(tensor2.size())}."
        )
    if tensor1.size()[2:] != tensor2.size()[2:]:
        raise ValueError(
            f"a.size()[2:] ({tensor1.size()[2:]}) != b.size()[2:] ({tensor2.size()[2:]})."
        )
    return torch.einsum("abcd, abde -> abce", tensor1, tensor2) # type: ignore
