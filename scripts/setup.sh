#!/usr/bin/env bash -l
set -e

pip install spconv-cu114 

echo "########################## Dependency Setup ##########################"
echo "Setting up IoU cuda operator ..."
# Navigate to the directory and run the installation
cd mvp/perception/cuda_op 
python setup.py install
cd -

echo "Setting up OpenCOOD ..."
# Navigate and build OpenCOOD related extensions
cd third_party/OpenCOOD
python opencood/utils/setup.py build_ext --inplace
python opencood/pcdet_utils/setup.py build_ext --inplace
cd -

echo "download data"
bash  scripts/download.sh