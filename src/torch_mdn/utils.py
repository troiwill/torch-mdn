import torch
from torch import Tensor
from typing import Tuple


def diag_indices_tri(ndim: int, is_lower: bool) -> Tuple[int]:
    """
    Computers the diagonal index values for a triangular matrix.

    Parameters
    ----------
    ndim : int
        The number of dimension of the triangular matrix. E.g., 3 for a 3 x 3 matrix.

    is_lower : bool
        Specifies if the matrix is a lower triangular (True) or upper triangular (False).

    Returns
    -------
    res : Tuple[int]
        A tuple of integers representing the diagonal indices for a lower or upper triangular
        matrix.
    """
    if not isinstance(ndim, int):
        raise TypeError("`ndim` must be an integer.")
    if ndim <= 0:
        raise ValueError(f"`ndim` must be a positive integer, but got {ndim}.")
    if not isinstance(is_lower, bool):
        raise Exception("`is_lower` must be a Boolean value.")

    # Stores the diagonal indices for a triangular matrix.
    diag_indices = list([0] * ndim)

    # Compute the diagonal indices only if ndim is at least 2.
    if ndim >= 2:
        start_idx, expected_end_idx = 0, int(((ndim * ndim) + ndim) * 0.5) - 1
        index_gaps = (
            list(range(2, ndim + 1, 1)) if is_lower else list(range(ndim, 1, -1))
        )

        diag_indices[0] = start_idx
        for i, ig in enumerate(index_gaps):
            diag_indices[i + 1] = diag_indices[i] + ig

        # Sanity check.
        if diag_indices[-1] != expected_end_idx:
            raise ValueError(
                f"Expected last index to be {expected_end_idx} "
                + f"but received {diag_indices[-1]}."
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
    return torch.finfo(torch.float32).eps


def num_tri_matrix_params_per_mode(ndim: int, is_unit_tri: bool) -> int:
    """
    Compute the number of free parameters for one triangular matrix.

    Parameters
    ----------
    ndim : int
        The number of dimensions in the data.

    is_unit_tri : bool
        Specifies if the resulting triangular matrix is unit (has a diagonal of ones) or not.

    Returns
    -------
    res : int
        The number of free parameters for a (unit or non-unit) triangular matrix with ndim
        dimensions.
    """
    if not isinstance(ndim, int):
        raise TypeError("`ndim` must be an integer.")
    if ndim <= 0:
        raise ValueError(f"`ndim` must be a positive integer, but got {ndim}.")
    if not isinstance(is_unit_tri, bool):
        raise Exception("`is_unit_tri` must be a Boolean value.")

    num_params = int(ndim * (ndim + 1) * 0.5)
    if is_unit_tri:
        num_params = num_params - ndim
    return num_params


def to_triangular_matrix(ndim: int, params: Tensor, is_lower: bool) -> Tensor:
    """
    Builds a triangular matrix using a set of free parameters. WARNING: this function only builds
    non-unit triangular matrices.

    Parameters
    ----------
    ndim : int
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
    # Allocate the triangular matrix.
    batch, nmodes, _ = list(params.size())
    tri_mat = torch.zeros(
        (batch, nmodes, ndim, ndim), dtype=params.dtype, device=params.device
    )

    if is_lower == True:
        i, j = torch.tril_indices(ndim, ndim)
    else:
        i, j = torch.triu_indices(ndim, ndim)
    tri_mat[:, :, i, j] = params
    return tri_mat


def torch_matmul_4d(a: Tensor, b: Tensor) -> Tensor:
    """
    Performs matrix-matrix multiplication for two 4D matrices, where the last two dimensions of
    each matrix is (N,N).

    Parameters
    ----------
    a : torch.Tensor
        The first 4-D matrix.

    b : torch.Tensor
        The second 4-D matrix.

    Returns
    -------
    res : torch.Tensor
        The matrix-matrix multiplication of a and b, where the third and fourth dimensions are
        multiplied.
    """
    return torch.einsum("abcd, abde -> abce", a, b)
