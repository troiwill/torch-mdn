#!/bin/bash

REPO_DIR_NAME="torch-mdn"
PACKAGE_NAME="torch_mdn"

REPO_PATH="${HOME}/repos/${REPO_DIR_NAME}"
DIR_FOR_ENV="${REPO_PATH}/env"
ENV_NAME="${PACKAGE_NAME}"
ENV_PATH="${DIR_FOR_ENV}/${ENV_NAME}"

# Check if conda is installed.
if [[ ! $(command conda -h ) ]]; then
    printf "Please install conda before proceeding.\n"
    exit 1
fi

# Determine if the conda environment is installed.
EXISTS=0
if [[ ! $(conda info --envs | grep ${PACKAGE_NAME}) ]]; then
    printf "The conda environment $ENV_NAME does not exist.\n"

    RESPONSE=""
    while [[ "$RESPONSE" != "y" && "$RESPONSE" != "n" ]]; do
        printf "Would you like me to create the environment? (y/n) "
        read RESPONSE
        printf "\n"
    done

    if [[ "$RESPONSE" == "y" ]]; then
        printf "Creating a conda environment for $ENV_NAME.\n"
        conda env create -f environment.yml
        printf "Done!"
    fi
else
    EXISTS=1
fi

# Activate the environment if it exists.
if [[ $EXISTS == 0 ]]; then
    printf "Cannot activate $ENV_NAME because it does not exist.\n"
else
    export VTORCH_MDN="${REPO_PATH}"
    export PYTHONPATH=$VTORCH_MDN:$PYTHONPATH
    conda activate $ENV_NAME
fi
