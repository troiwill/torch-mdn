# PyTorch MDN

PyTorch MDN (torch-mdn) is a set of classes and functions for building and evaluating a Mixture Density Network in PyTorch.

## Setting up the environment
Run the following commands to set up the environment.

```
cd ${HOME}
mkdir -p repos && cd repos
git clone https://github.com/troiwill/torch-mdn.git
conda create -n torch-mdn python=3.8 pytorch==1.10.2 matplotlib

# Activate the conda environment and install Python dependencies.
conda activate torch-mdn
pip install pydantic==1.10.10
```

## Installing the torch_mdn Python package

Installing this Python package requires a two-step process. First, you must build the package. Assuming you set up the environment as mentioned above, run the following commands:
```
conda activate torch_mdn
cd ${HOME}/repos/torch-mdn
python -m build
```

Once you built the package, use pip to install the wheel file (`*.whl`). For example, `pip install <TORCH_MDN_WHEEL>.whl`. If you do not want to install this package in the current conda environment, deactivate the environment first (using `conda deactivate`) and then install the package in the appropriate environment.

## Testing the Library

We use the `pytest` package to sanity check the components in the library. To install `pytest`, run:
```
pip install pytest
```

To run all the tests, run:
```
cd tests
pytest
```

To run individual test files, run:
```
cd tests
pytest test_<file name>.py # for example, test_utils.py
```
