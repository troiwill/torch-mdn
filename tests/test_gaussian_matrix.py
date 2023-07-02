from pydantic import ValidationError
import pytest
import torch
from torch_mdn.gaussian import (
    MatrixDecompositionType,
    MatrixPredictionType,
    GaussianMatrixLayer,
)


class Test_GaussianMatrixLayer:
    def test_validation_error_matrix_decomp_type(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                matrix_decomp_type="hello", # type: ignore
                predict_matrix_type="hello", # type: ignore
                ndim=-1,
            )

    def test_validation_error_predict_matrix_type(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                matrix_decomp_type=MatrixDecompositionType.full_UU,
                predict_matrix_type="hello", # type: ignore
                ndim=-1,
            )

    def test_validation_error_ndim_type(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                matrix_decomp_type=MatrixDecompositionType.full_UU,
                predict_matrix_type=MatrixPredictionType.covariance,
                ndim="hello", # type: ignore
            )

    def test_validation_error_ndim_value(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                matrix_decomp_type=MatrixDecompositionType.full_UU,
                predict_matrix_type=MatrixPredictionType.covariance,
                ndim=-1,
            )

    def test_apply_activation_decomposition_type_diagonal_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ = GaussianMatrixLayer(
                matrix_decomp_type=MatrixDecompositionType.diagonal,
                predict_matrix_type=MatrixPredictionType.covariance,
                ndim=3,
            )

    def test_apply_activation_decomposition_type_LDL_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ = GaussianMatrixLayer(
                matrix_decomp_type=MatrixDecompositionType.full_LDL,
                predict_matrix_type=MatrixPredictionType.covariance,
                ndim=3,
            )

    def test_ndim1(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        assert layer.ndim == 1

    def test_ndim3(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        assert layer.ndim == 3

    def test_forward_ndim1(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        input_x = torch.tensor([[6.0]])
        assert layer.forward(x=input_x) == torch.tensor([[[7.0]]])

    def test_forward_ndim3(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        input_x = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
        assert (
            torch.allclose(
                layer.forward(x=input_x),
                torch.tensor(
                    [
                        [
                            [0.36787957, 0.0000, 1.0000, 3.0000, 3.0000, 5.0000],
                        ]
                    ]
                ),
            )
            is True
        )

    def test_predict_precision_ndim1(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.precision,
            ndim=1,
        )
        input_x = torch.tensor([[6.0]])
        assert (
            torch.allclose(layer.predict(x=input_x), torch.tensor([[[49.0]]])) is True
        )

    def test_predict_precision_ndim3(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.precision,
            ndim=3,
        )
        input_x = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
        assert (
            torch.allclose(
                layer.predict(x=input_x),
                torch.tensor(
                    [
                        [
                            [0.13533537, 0.0, 0.36787957],
                            [0.0, 9.0, 9.0],
                            [0.36787957, 9.0, 35.0],
                        ]
                    ]
                ),
            )
            is True
        )

    def test_predict_covariance_ndim1(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        input_x = torch.tensor([[6.0]])
        assert (
            torch.allclose(
                layer.predict(x=input_x), torch.tensor([[[0.02040816326530612]]])
            )
            is True
        )

    def test_predict_covariance_ndim3(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        input_x = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
        assert (
            torch.allclose(
                layer.predict(x=input_x),
                torch.tensor(
                    [
                        [
                            [7.6846147, 0.10873125, -0.10873125],
                            [0.10873171, 0.15111111, -0.04],
                            [-0.10873171, -0.04, 0.04],
                        ]
                    ]
                ),
            )
            is True
        )

    def test_expected_input_shape_ndim1(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        assert layer.expected_input_shape() == (1, 1)

    def test_expected_input_shape_ndim3(self):
        layer = GaussianMatrixLayer(
            matrix_decomp_type=MatrixDecompositionType.full_UU,
            predict_matrix_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        assert layer.expected_input_shape() == (1, 6)
