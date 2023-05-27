from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset as DatasetBase


class Dataset(DatasetBase):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> Dataset:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Grapher:
    """A class used to graph data quickly for the examples in this library."""

    @staticmethod
    def graph_1d(x: np.ndarray, y: np.ndarray, title: str = "Data") -> None:
        """Graphs 1D data in a scatter plot.

        x : np.ndarray
            The x values.

        y : np.ndarray
            The y values.

        title : str
            The title of the graph.
        """
        plt.scatter(x, y, s=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.show()
        plt.close()


class UnimodalData:
    """A class used to generate different unimodal data."""

    @staticmethod
    def generate_linear_noise(
        a: float = 9.3, stdv: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates noisy data using a linear function: y = ax + ((0.25x)^2) * noise, where noise
        ~ N(0, stdv) and x in [0.0, 10.0].

        Parameters
        ----------
        a : float
            The scalar coefficient for the linear function.

        stdv : float
            The scalar standard deviation for the noise.

        Returns
        -------
        x, y : tuple[ np.ndarray, np.ndarray ]
            The scalar x and y values
        """
        x = np.linspace(0.0, 10.0, 10000)
        noise = np.random.normal(0.0, stdv, size=x.shape[0])
        b = np.square(0.25 * x) * noise
        y = (a * x) + b

        return x, y

    @staticmethod
    def generate_linear_noise_with_cap(
        a: float = 9.3, stdv: float = 1.0, x_cap: float = 4.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates noisy data using a linear function: y = ax + ((0.25x)^2) * noise, where noise
        ~ N(0, stdv) and x in [0.0, 10.0]. However, from [`x_cap`, `X_MAX`], the function becomes
        y = ax + ((0.25*`x_cap`)^2) * noise.

        Parameters
        ----------
        a : float
            The scalar coefficient for the linear function.

        stdv : float
            The scalar standard deviation for the noise.

        x_cap : float
            The x value at which the noise becomes constant.

        Returns
        -------
        x, y : tuple[ np.ndarray, np.ndarray ]
            The scalar x and y values
        """
        X_MIN = 0.0
        X_MAX = 10.0
        assert X_MIN < x_cap < X_MAX

        x1 = np.linspace(X_MIN, x_cap, int(x_cap - X_MIN) * 1000, endpoint=False)
        noise1 = np.random.normal(0.0, stdv, size=x1.shape[0])
        b1 = np.square(0.25 * x1) * noise1

        x2 = np.linspace(x_cap, X_MAX, int(X_MAX - x_cap) * 1000)
        noise2 = np.random.normal(0.0, stdv, size=x2.shape[0])
        b2 = np.square(0.25 * x_cap) * noise2

        x = np.hstack((x1.flatten(), x2.flatten()))
        b = np.hstack((b1.flatten(), b2.flatten()))

        y = (a * x) + b

        return x, y
