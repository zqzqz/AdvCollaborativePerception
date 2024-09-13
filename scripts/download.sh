#!/usr/bin/env bash -l
set -e

echo "########################## Download ##########################"
if [ ! -d models ]; then
    mkdir models
fi

if [ ! -d data ]; then
    mkdir data
fi

if [ ! -f models/model_experiment.zip ]; then
    # Website link: https://drive.google.com/file/d/1exCVzj7I7f201MOvCHFldBMsjA9oBF6D/view?usp=sharing
    pip install gdown
    gdown -O models/model_experiment.zip 1exCVzj7I7f201MOvCHFldBMsjA9oBF6D
    cd models && unzip model_experiment.zip && cd -
fi

if [ ! -f data/data_experiment.zip ]; then
    # Website link: https://drive.google.com/file/d/1KW1Ya8CGvJuqB5dUBmH968zDFbBFgNtm/view?usp=sharing
    gdown -O data/data_experiment.zip 1KW1Ya8CGvJuqB5dUBmH968zDFbBFgNtm
    cd data && unzip data_experiment.zip && cd -
fi

if [ ! -f data/OPV2V/test.zip ]; then
    # Website link: https://drive.google.com/file/d/1fuYK-oNA0FpZtT8rUiEETOCNmtO3FCfS/view?usp=sharing
    gdown -O data/OPV2V/test.zip 1fuYK-oNA0FpZtT8rUiEETOCNmtO3FCfS
    cd data/OPV2V && unzip test.zip && cd -
fi
