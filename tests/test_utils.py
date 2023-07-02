"""Module to test code in utils.py"""

import numpy as np
from pydantic import ValidationError
import pytest
import sys

SRC_PATH = "../src"
if SRC_PATH not in sys.path:
    sys.path.append("../src")
import torch
from torch_mdn.utils import (
    diag_indices_tri,
    epsilon,
    num_tri_matrix_params_per_mode,
    to_triangular_matrix,
    torch_matmul_4d,
    create_torch_indices,
)


class TestTorchIndices:
    def test_validation_error_type(self):
        with pytest.raises(ValidationError):
            _ = create_torch_indices(["hello", 0.3])  # type: ignore

    def test_type(self):
        indices = create_torch_indices([0, 13, 5, 2])
        assert indices.dtype == torch.int64

    def test_indices_len1(self):
        indices = create_torch_indices([0])
        assert len(indices) == 1

    def test_indices_len4(self):
        indices = create_torch_indices([0, 1, 2, 4])
        assert len(indices) == 4


class TestDiagIndicesTri:
    def test_validation_error_ndim(self):
        with pytest.raises(ValidationError):
            _ = diag_indices_tri(ndim="hello", is_lower=True)  # type: ignore

    def test_value_error_ndim(self):
        with pytest.raises(ValidationError):
            _ = diag_indices_tri(ndim=-1, is_lower=True)

    def test_validation_error_is_lower(self):
        with pytest.raises(ValidationError):
            _ = diag_indices_tri(ndim=1, is_lower="hello")  # type: ignore

    def test_return_type(self):
        res = diag_indices_tri(ndim=2, is_lower=False)
        assert isinstance(res, tuple) is True
        assert all([isinstance(x, int) for x in res]) is True

    def test_upper_tri_ndim1(self):
        res = diag_indices_tri(ndim=1, is_lower=False)

        assert len(res) == 1
        assert res[0] == 0

    def test_upper_tri_ndim2(self):
        res = diag_indices_tri(ndim=2, is_lower=False)

        assert len(res) == 2
        assert res == (0, 2)

    def test_upper_tri_ndim3(self):
        res = diag_indices_tri(ndim=3, is_lower=False)

        assert len(res) == 3
        assert res == (0, 3, 5)

    def test_lower_tri_ndim1(self):
        res = diag_indices_tri(ndim=1, is_lower=True)

        assert len(res) == 1
        assert res[0] == 0

    def test_lower_tri_ndim2(self):
        res = diag_indices_tri(ndim=2, is_lower=True)

        assert len(res) == 2
        assert res == (0, 2)

    def test_lower_tri_ndim3(self):
        res = diag_indices_tri(ndim=3, is_lower=True)

        assert len(res) == 3
        assert res == (0, 2, 5)


class TestEpsilon:
    def test_return_type(self):
        res = epsilon()
        assert isinstance(res, float) is True

    def test_return_value(self):
        res = epsilon()
        assert res <= 1e-6


class TestNumTriMatrixParamsPerMode:
    def test_validation_error_ndim(self):
        with pytest.raises(ValidationError):
            _ = num_tri_matrix_params_per_mode(ndim="kd", is_unit_tri=True)  # type: ignore

    def test_value_error_ndim(self):
        with pytest.raises(ValidationError):
            _ = num_tri_matrix_params_per_mode(ndim=0, is_unit_tri=True)

    def test_validation_error_is_unit_tri(self):
        with pytest.raises(ValidationError):
            _ = num_tri_matrix_params_per_mode(ndim=1, is_unit_tri=23)  # type: ignore

    def test_return_type(self):
        res = num_tri_matrix_params_per_mode(ndim=1, is_unit_tri=True)
        assert isinstance(res, int) is True

    def test_ndim1_unit_tri(self):
        res = num_tri_matrix_params_per_mode(ndim=1, is_unit_tri=True)
        assert res == 0

    def test_ndim2_unit_tri(self):
        res = num_tri_matrix_params_per_mode(ndim=2, is_unit_tri=True)
        assert res == 1

    def test_ndim3_unit_tri(self):
        res = num_tri_matrix_params_per_mode(ndim=3, is_unit_tri=True)
        assert res == 3

    def test_ndim1_non_unit_tri(self):
        res = num_tri_matrix_params_per_mode(ndim=1, is_unit_tri=False)
        assert res == 1

    def test_ndim2_non_unit_tri(self):
        res = num_tri_matrix_params_per_mode(ndim=2, is_unit_tri=False)
        assert res == 3

    def test_ndim3_non_unit_tri(self):
        res = num_tri_matrix_params_per_mode(ndim=3, is_unit_tri=False)
        assert res == 6


class TestToTriangularMatrix:
    def test_type_error_ndim(self):
        with pytest.raises(ValidationError):
            _ = to_triangular_matrix(ndim=12.0, params="23", is_lower=10)  # type: ignore

    def test_value_error_ndim(self):
        with pytest.raises(ValidationError):
            _ = to_triangular_matrix(ndim=-1, params="23", is_lower=10)  # type: ignore

    def test_type_error_params(self):
        with pytest.raises(ValidationError):
            _ = to_triangular_matrix(ndim=1, params="23", is_lower=10)  # type: ignore

    def test_params_size_length(self):
        with pytest.raises(ValidationError):
            wrong_params = torch.randn(10, 3)
            _ = to_triangular_matrix(ndim=1, params=wrong_params, is_lower=10)  # type: ignore

    def test_type_error_is_lower(self):
        with pytest.raises(ValidationError):
            wrong_params = torch.tensor(np.random.random(size=(10, 3, 5)))
            _ = to_triangular_matrix(ndim=1, params=wrong_params, is_lower=10)  # type: ignore

    def test_params_wrong_size(self):
        with pytest.raises(ValueError):
            wrong_params = torch.tensor(np.random.random(size=(10, 3, 5)))
            _ = to_triangular_matrix(ndim=1, params=wrong_params, is_lower=True)

    def test_params_ndim1_is_lower_tri(self):
        correct_params = torch.tensor(np.arange(6).reshape(3, 2, 1))  # type: ignore
        res = to_triangular_matrix(ndim=1, params=correct_params, is_lower=True)
        assert torch.equal(res, correct_params.reshape((3, 2, 1, 1))) == True

    def test_params_ndim2_is_lower_tri(self):
        tri_params = torch.tensor(np.arange(18).reshape(3, 2, 3))  # type: ignore
        res = to_triangular_matrix(ndim=2, params=tri_params, is_lower=True)
        correct_params = torch.tensor(
            np.array(
                [
                    [[[0, 0], [1, 2]], [[3, 0], [4, 5]]],
                    [[[6, 0], [7, 8]], [[9, 0], [10, 11]]],
                    [[[12, 0], [13, 14]], [[15, 0], [16, 17]]],
                ]
            )
        )
        assert torch.equal(res, correct_params) is True

    def test_params_ndim3_is_lower_tri(self):
        tri_params = torch.tensor(np.arange(12).reshape(2, 1, 6))  # type: ignore
        res = to_triangular_matrix(ndim=3, params=tri_params, is_lower=True)
        correct_params = torch.tensor(
            np.array(
                [
                    [[[0, 0, 0], [1, 2, 0], [3, 4, 5]]],
                    [[[6, 0, 0], [7, 8, 0], [9, 10, 11]]],
                ]
            )
        )
        assert torch.equal(res, correct_params) is True

    def test_params_ndim1_is_upper_tri(self):
        tri_params = torch.tensor(np.arange(6).reshape(3, 2, 1))  # type: ignore
        res = to_triangular_matrix(ndim=1, params=tri_params, is_lower=False)
        correct_params = torch.tensor(
            np.array(
                [
                    [[[0]], [[1]]],
                    [[[2]], [[3]]],
                    [[[4]], [[5]]],
                ]
            )
        )
        assert torch.equal(res, correct_params) is True

    def test_params_ndim2_is_upper_tri(self):
        tri_params = torch.tensor(np.arange(18).reshape(3, 2, 3))  # type: ignore
        res = to_triangular_matrix(ndim=2, params=tri_params, is_lower=False)
        correct_params = torch.tensor(
            np.array(
                [
                    [[[0, 1], [0, 2]], [[3, 4], [0, 5]]],
                    [[[6, 7], [0, 8]], [[9, 10], [0, 11]]],
                    [[[12, 13], [0, 14]], [[15, 16], [0, 17]]],
                ]
            )
        )
        assert torch.equal(res, correct_params) is True

    def test_params_ndim3_is_upper_tri(self):
        tri_params = torch.tensor(np.arange(12).reshape(2, 1, 6))  # type: ignore
        res = to_triangular_matrix(ndim=3, params=tri_params, is_lower=False)
        correct_params = torch.tensor(
            np.array(
                [
                    [[[0, 1, 2], [0, 3, 4], [0, 0, 5]]],
                    [[[6, 7, 8], [0, 9, 10], [0, 0, 11]]],
                ]
            )
        )
        assert torch.equal(res, correct_params) is True


class TestTorchMatmul4D:
    def test_type_error_a(self):
        with pytest.raises(ValidationError):
            _ = torch_matmul_4d(tensor1=12, tensor2=12)  # type: ignore

    def test_type_error_b(self):
        with pytest.raises(ValidationError):
            a = torch.tensor(
                np.array([[[[1, 3], [1, 1]], [[2, 1], [1, 3]], [[2, 2], [4, 3]]]])
            )
            _ = torch_matmul_4d(tensor1=a, tensor2=12)  # type: ignore

    def test_value_error_a_length(self):
        with pytest.raises(ValueError):
            a = torch.tensor(
                np.array([[[1, 3], [1, 1]], [[2, 1], [1, 3]], [[2, 2], [4, 3]]])
            )
            _ = torch_matmul_4d(tensor1=a, tensor2=a)

    def test_value_error_b_length(self):
        with pytest.raises(ValueError):
            a = torch.tensor(
                np.array([[[[1, 3], [1, 1]], [[2, 1], [1, 3]], [[2, 2], [4, 3]]]])
            )
            b = torch.tensor(
                np.array([[[1, 3], [1, 1]], [[2, 1], [1, 3]], [[2, 2], [4, 3]]])
            )
            _ = torch_matmul_4d(tensor1=a, tensor2=b)

    def test_value_error_ab_sizes(self):
        with pytest.raises(ValueError):
            a = torch.tensor(
                np.array([[[[1, 3], [1, 1]], [[2, 1], [1, 3]], [[2, 2], [4, 3]]]])
            )
            b = torch.tensor(
                np.array(
                    [
                        [
                            [[2, 3, 1], [4, 3, 1]],
                            [[0, 9, 2], [5, 4, 2]],
                            [[6, 8, 5], [7, 2, 6]],
                        ]
                    ]
                )
            )
            _ = torch_matmul_4d(tensor1=a, tensor2=b)

    def test_matmul_identity1(self):
        rand = torch.tensor(np.random.random(size=(5, 9, 2, 2)))
        identity = torch.tensor(np.tile(np.eye(2).reshape(1, 1, 2, 2), (5, 9, 1, 1)))
        product = torch_matmul_4d(tensor1=rand, tensor2=identity)
        assert torch.equal(rand, product) is True

    def test_matmul_identity2(self):
        rand = torch.tensor(np.random.random(size=(5, 9, 2, 2)))
        identity = torch.tensor(np.tile(np.eye(2).reshape(1, 1, 2, 2), (5, 9, 1, 1)))
        product = torch_matmul_4d(tensor1=identity, tensor2=rand)
        assert torch.equal(rand, product) is True

    def test_matmul_is_same_as_numpy_matmul(self):
        a = np.array([[[[1, 3], [1, 1]], [[2, 1], [1, 3]], [[2, 2], [4, 3]]]])
        b = np.array([[[[2, 3], [4, 3]], [[0, 9], [5, 4]], [[6, 8], [7, 2]]]])
        np_product = a @ b
        product = torch_matmul_4d(tensor1=torch.tensor(a), tensor2=torch.tensor(b))
        assert torch.equal(product, torch.tensor(np_product)) is True
