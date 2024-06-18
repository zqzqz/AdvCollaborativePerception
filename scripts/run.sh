#!/usr/bin/env bash -l
set -e

env_name="advCP"

echo "########################## Conda Env ##########################"
if conda env list | grep -qE "^$env_name\s"; then
    echo "Conda environment '$env_name' exists."
else
    echo "Conda environment '$env_name' does not exist. creating ..."
    conda env create --name $env_name --file environment.yml
fi

echo "Installing pytorch, cuda, and related packages ..."
conda install -y --name $env_name pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=10.2 -c pytorch
pip install spconv-cu102

echo "########################## Download ##########################"
if [ ! -d models ]; then
	mkdir models
fi

if [ ! -d data ]; then
	mkdir data
fi

if [ ! -f models/model_experiment.zip ]; then
	conda run --live-stream -n $env_name gdown -O models/model_experiment.zip 1exCVzj7I7f201MOvCHFldBMsjA9oBF6D
	cd models && unzip model_experiment.zip && cd -
fi

if [ ! -f data/data_experiment.zip ]; then
	conda run --live-stream -n $env_name gdown -O data/data_experiment.zip 1KW1Ya8CGvJuqB5dUBmH968zDFbBFgNtm
	cd data && unzip data_experiment.zip && cd -
fi

if [ ! -f data/OPV2V/test.zip ]; then
	conda run --live-stream -n $env_name gdown -O data/OPV2V/test.zip 1fuYK-oNA0FpZtT8rUiEETOCNmtO3FCfS
	cd data/OPV2V && unzip test.zip && cd -
fi

echo "########################## Dependency Setup ##########################"
echo "Setting up IoU cuda operator ..."
cd mvp/perception/cuda_op 
conda run --live-stream -n $env_name python setup.py install
cd -

echo "Setting up OpenCOOD ..."
cd third_party/OpenCOOD
conda run --live-stream -n $env_name python opencood/utils/setup.py build_ext --inplace
conda run --live-stream -n $env_name python opencood/pcdet_utils/setup.py build_ext --inplace
cd -

echo "########################## All Evaluation ##########################"
echo "Running all evaluation tasks (this may take a long time) ..."
echo "If the evaluation is interrupted, run python scripts/evaluation.py to resume."
conda run --live-stream -n $env_name python scripts/evaluate.py
