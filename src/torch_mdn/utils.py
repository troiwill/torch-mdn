import torch
from torch import Tensor
from typing import Tuple, List


def diag_indices_tri(ndim: int, is_lower: bool) -> Tuple[List[int]]:
    """
    Returns the diagonal index values for a triangular matrix.
    """
    ndim = int(ndim)
    if ndim <= 0:
        raise Exception("`ndim` must be a positive integer.")
    if not isinstance(is_lower, bool):
        raise Exception("`is_lower` must be a Boolean value.")

    # Stores the diagonal indices for a triangular matrix.
    diag_indices = list([0] * ndim)

    # Compute the diagonal indices only if ndim is at least 2.
    if ndim >= 2:
        start_idx, expected_end_idx = 0, int(((ndim * ndim) + ndim) * 0.5) - 1
        index_gaps = list(range(2, ndim + 1, 1)) \
            if is_lower else list(range(ndim, 1, -1))

        diag_indices[0] = start_idx
        for i, ig in enumerate(index_gaps):
            diag_indices[i+1] = diag_indices[i] + ig

        # Sanity check.
        if diag_indices[-1] != expected_end_idx:
            raise Exception(f'Expected last index to be {expected_end_idx} ' \
                + f'but received {diag_indices[-1]}.')
    #end if

    return tuple(diag_indices)
#end def

def epsilon() -> float:
    """
    Epsilon number for PyTorch for a floating-point number.
    """
    return torch.finfo(torch.float32).eps

def num_tri_matrix_params_per_mode(ndim: int, is_unit_tri: bool) -> int:
    """
    Compute the number of free parameters for each mode depending on the 
    number of dimensions (ndim) and if the matrix is triangular or not.
    """
    num_params = int(ndim * (ndim + 1) * 0.5)
    if is_unit_tri:
        num_params = num_params - ndim
    return num_params
#end def

def to_triangular_matrix(ndim: int, params: Tensor, is_lower: bool) -> Tensor:
    """
    Builds a triangular matrix using a set of free parameters with ndim 
    dimensions.
    """
    # Allocate the triangular matrix.
    batch, nmodes, _ = list(params.size())
    tri_mat = torch.zeros((batch, nmodes, ndim, ndim),
        dtype = params.dtype, device = params.device)

    i, j = torch.tril_indices(ndim, ndim) if is_lower \
        else torch.triu_indices(ndim, ndim)
    tri_mat[:, :, i, j] = params
    return tri_mat
#end def

def torch_matmul_4d(a: Tensor, b: Tensor) -> Tensor:
    """
    Performs matrix-matrix multiplication for two 4D matrices, where the last
    two dimensions of each matrix is (N,N).
    """
    return torch.einsum('abcd, abde -> abce', a, b)
#end def
