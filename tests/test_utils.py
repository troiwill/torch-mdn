import numpy as np
import pytest
import torch
from torch_mdn.utils import diag_indices_tri, epsilon, num_tri_matrix_params_per_mode, to_triangular_matrix


class TestDiagIndicesTri:
    def test_type_error_ndim(self):
        with pytest.raises(TypeError):
            _ = diag_indices_tri(ndim=0.2, is_lower=True)

    def test_value_error_ndim(self):
        with pytest.raises(ValueError):
            _ = diag_indices_tri(ndim=0, is_lower=True)

    def test_type_error_is_lower(self):
        with pytest.raises(TypeError):
            _ = diag_indices_tri(ndim=1, is_lower=0)

    def test_return_type(self):
        res = diag_indices_tri(ndim=2, is_lower=False)
        assert isinstance(res, tuple)
        assert all([isinstance(x, int) for x in res]) == True

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
        assert isinstance(res, float)

    def test_return_value(self):
        res = epsilon()
        assert res <= 1e-6


class TestNumTriMatrixParamsPerMode:
    def test_type_error_ndim(self):
        with pytest.raises(TypeError):
            _ = num_tri_matrix_params_per_mode(ndim="kd", is_unit_tri=True)

    def test_value_error_ndim(self):
        with pytest.raises(ValueError):
            _ = num_tri_matrix_params_per_mode(ndim=0, is_unit_tri=True)

    def test_type_error_ndim(self):
        with pytest.raises(TypeError):
            _ = num_tri_matrix_params_per_mode(ndim=1, is_unit_tri=23)

    def test_return_type(self):
        res = num_tri_matrix_params_per_mode(ndim=1, is_unit_tri=True)
        assert isinstance(res, int)

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
        with pytest.raises(TypeError):
            _ = to_triangular_matrix(ndim=12.0, params="23", is_lower=10)

    def test_value_error_ndim(self):
        with pytest.raises(ValueError):
            _ = to_triangular_matrix(ndim=-1, params="23", is_lower=10)

    def test_type_error_params(self):
        with pytest.raises(TypeError):
            _ = to_triangular_matrix(ndim=1, params="23", is_lower=10)

    def test_params_size_length(self):
        with pytest.raises(ValueError):
            wrong_params = torch.tensor(np.random.random(size=(10, 3)))
            _ = to_triangular_matrix(ndim=1, params=wrong_params, is_lower=10)

    def test_type_error_is_lower(self):
        with pytest.raises(TypeError):
            wrong_params = torch.tensor(np.random.random(size=(10, 3, 5))) 
            _ = to_triangular_matrix(ndim=1, params=wrong_params, is_lower=10)

    def test_params_wrong_size(self):
        with pytest.raises(ValueError):
            wrong_params = torch.tensor(np.random.random(size=(10, 3, 5))) 
            _ = to_triangular_matrix(ndim=1, params=wrong_params, is_lower=True)

    def test_params_correct_size_ndim1_is_lower_tri(self):
        correct_params = torch.tensor(np.arange(6).reshape(3,2,1))
        res = to_triangular_matrix(ndim=1, params=correct_params, is_lower=True)
        assert res.size() == correct_params.reshape((3, 2, 1, 1)).size()
