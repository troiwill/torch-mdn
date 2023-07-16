from pydantic import ValidationError
import pytest
import sys

SRC_PATH = "../src"
if SRC_PATH not in sys.path:
    sys.path.append("../src")
import torch
from torch_mdn.gaussian import (
    MatrixDecompositionType,
    MatrixPredictionType,
    GaussianMatrixLayer,
)


class Test_GaussianMatrixLayer:
    def test_validation_error_decomposition_impl_or_type(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                decomposition_impl_or_type="hello",  # type: ignore
                prediction_type="hello",  # type: ignore
                ndim=-1,
            )

    def test_validation_error_prediction_type(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                decomposition_impl_or_type=MatrixDecompositionType.cholesky,
                prediction_type="hello",  # type: ignore
                ndim=-1,
            )

    def test_validation_error_ndim_type(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                decomposition_impl_or_type=MatrixDecompositionType.cholesky,
                prediction_type=MatrixPredictionType.covariance,
                ndim="hello",  # type: ignore
            )

    def test_validation_error_ndim_value(self):
        with pytest.raises(ValidationError):
            _ = GaussianMatrixLayer(
                decomposition_impl_or_type=MatrixDecompositionType.cholesky,
                prediction_type=MatrixPredictionType.covariance,
                ndim=-1,
            )

    # def test_ndim1(self):
    #     layer = GaussianMatrixLayer(
    #         decomposition_impl_or_type=MatrixDecompositionType.cholesky,
    #         prediction_type=MatrixPredictionType.covariance,
    #         ndim=1,
    #     )
    #     assert layer.ndim == 1

    # def test_ndim3(self):
    #     layer = GaussianMatrixLayer(
    #         decomposition_impl_or_type=MatrixDecompositionType.cholesky,
    #         prediction_type=MatrixPredictionType.covariance,
    #         ndim=3,
    #     )
    #     assert layer.ndim == 3

    def test_forward_ndim1(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        input_x = torch.tensor([[6.0]])
        assert layer.forward(free_params=input_x) == torch.tensor([[[7.0]]])

    def test_forward_ndim3(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        input_x = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
        free_params = layer.forward(free_params=input_x)
        assert isinstance(free_params, torch.Tensor)
        assert (
            torch.allclose(
                free_params,
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
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.precision,
            ndim=1,
        )
        input_x = torch.tensor([[6.0]])
        assert (
            torch.allclose(layer.predict(free_params=input_x), torch.tensor([[[49.0]]]))
            is True
        )

    def test_predict_precision_ndim3(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.precision,
            ndim=3,
        )
        input_x = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
        assert (
            torch.allclose(
                layer.predict(free_params=input_x),
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
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        input_x = torch.tensor([[6.0]])
        assert (
            torch.allclose(
                layer.predict(free_params=input_x),
                torch.tensor([[[0.02040816326530612]]]),
            )
            is True
        )

    def test_predict_covariance_ndim3(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        input_x = torch.tensor([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])
        assert (
            torch.allclose(
                layer.predict(free_params=input_x),
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
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        assert layer.expected_input_shape() == (1, 1)

    def test_expected_input_shape_ndim3(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        assert layer.expected_input_shape() == (1, 6)

    def test_forward_shape_ndim1(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        assert layer.forward_shapes() == ((1, 1, 1),)

    def test_forward_shape_ndim3(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        assert layer.forward_shapes() == ((1, 1, 6),)

    def test_prediction_shape_ndim1(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=1,
        )
        assert layer.prediction_shapes() == ((1, 1, 1),)

    def test_prediction_shape_ndim3(self):
        layer = GaussianMatrixLayer(
            decomposition_impl_or_type=MatrixDecompositionType.cholesky,
            prediction_type=MatrixPredictionType.covariance,
            ndim=3,
        )
        assert layer.prediction_shapes() == ((1, 3, 3),)
