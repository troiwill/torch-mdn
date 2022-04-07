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

    start_index, end_index = 0, int(((ndim * ndim) + ndim) * 0.5) - 1
    index_gap = 2 if is_lower else -1 * ndim

    diag_indices = list([0] * ndim)
    diag_indices[0] = start_index
    diag_indices[-1] = end_index
    for i in range(1, ndim):
        diag_indices[i] = diag_indices[i - 1] + index_gap
        index_gap = index_gap + 1

    return tuple(diag_indices)
#end def

def epsilon() -> float:
    return torch.finfo(torch.float32).eps

def num_tri_matrix_params_per_mode(ndim: int, is_unit_tri: bool) -> int:
    num_params = int(ndim * (ndim + 1) * 0.5)
    if is_unit_tri:
        num_params = num_params - ndim
    return num_params
#end def

def to_triangular_matrix(ndim: int, params: Tensor) -> Tensor:
    # Allocate the triangular matrix.
    batch, nmodes, _ = list(params.size())
    tri_mat = torch.zeros((batch, nmodes, ndim, ndim),
        dtype = params.dtype, device = params.device)

    i, j = torch.tril_indices(ndim, ndim)
    tri_mat[:, :, i, j] = params
    return tri_mat
#end def
