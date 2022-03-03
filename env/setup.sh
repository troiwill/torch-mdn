#!/bin/bash

REPO_DIR_NAME="torch-mdn"
PACKAGE_NAME="torch_mdn"

REPO_PATH="${HOME}/repos/${REPO_DIR_NAME}"
DIR_FOR_ENV="${REPO_PATH}/env"
ENV_NAME="${PACKAGE_NAME}"
ENV_PATH="${DIR_FOR_ENV}/${ENV_NAME}"

pushd "${DIR_FOR_ENV}"

# Check if python3 is installed.
if [[ ! $(command python3 -h) ]]; then
    printf "Please install python3 before proceeding.\n"
    return 0
fi

# Check if virtualenv is installed.
if [[ ! $(command python3 -m virtualenv -h) ]]; then
    printf "Installing virtualenv.\n"
    python3 -m pip install virtualenv
    printf "Done!\n"
fi

# Check if the environment was set up.
EXISTS=0

if [[ ! -d "${ENV_PATH}" ]]; then
    printf "The $ENV_NAME environment does not exist.\n"

    RESPONSE=""
    while [[ "$RESPONSE" != "y" && "$RESPONSE" != "n" ]]; do
        printf "Would you like me to create the environment? (y/n) "
        read RESPONSE
        printf "\n"
    done

    if [[ "$RESPONSE" == "y" ]]; then
        printf "Creating a virtual environment for $ENV_NAME.\n"
        python3 -m virtualenv ${ENV_NAME} && EXISTS=1
        printf "Done!" # "Once the environment is activated, run ''bash install_packages.sh''\n"
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
    source ${ENV_NAME}/bin/activate
fi

popd
