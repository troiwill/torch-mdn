# PyTorch MDN

PyTorch MDN (torch-mdn) is a set of classes and functions for building and evaluating a Mixture Density Network in PyTorch.

## Setting up the environment
There are two methods you can use to set up the environment.

### Method #1 (Recommended)
Run the following conda command:
```
cd ${HOME}
mkdir -p repos && cd repos
git clone https://github.com/troiwill/torch-mdn.git
conda create -n torch_mdn python=3.8 pytorch matplotlib
```
This should create a conda environment with the same dependencies mentioned in the [environment.yml](env/environment.yml) file.

### Method #2
Clone the repository and then source the [activate.sh](env/activate.sh) script.
```
cd ${HOME}
mkdir -p repos && cd repos
git clone https://github.com/troiwill/torch-mdn.git
cd torch-mdn/env
source activate.sh
```
Please note that sourcing the [activate.sh](env/activate.sh) script will also activate the environment and add `torch_mdn` to the `PYTHONPATH` environment variable.

**Note:** After you set up the environment using Method #1 or #2, install the build tool via the following command: `pip install build`.

## Installing the torch_mdn Python package

Installing this Python package requires a two-step process. First, you must build the package. Assuming you set up the environment as mentioned above, run the following commands:
```
conda activate torch_mdn
cd ${HOME}/repos/torch-mdn
python -m build
```

Once you built the package, use pip to install the wheel file (`*.whl`). For example, `pip install <TORCH_MDN_WHEEL>.whl`. If you do not want to install this package in the current conda environment, deactivate the environment first (using `conda deactivate`) and then install the package in the appropriate environment.
