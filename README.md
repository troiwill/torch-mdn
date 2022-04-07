# PyTorch MDN

PyTorch MDN (torch-mdn) is a set of classes and functions for building and evaluating a Mixture Density Network in PyTorch.

## Setting up the environment
There are two methods you can use to set up the environment.

### Method #1
Clone the repository and then source the [activate.sh](env/activate.sh) script.
```
cd ${HOME}
mkdir -p repos && cd repos
git clone https://github.com/troiwill/torch-mdn.git
cd torch-mdn/env
source activate.sh
```
Please note that sourcing the [activate.sh](env/activate.sh) script will also activate the environment as well.

### Method #2
Run the following conda command:
```
conda create -n torch_mdn python=3.8 pytorch matplotlib
```
This should create a conda environment with the same dependencies mentioned in the [environment.yml](env/environment.yml) file.
