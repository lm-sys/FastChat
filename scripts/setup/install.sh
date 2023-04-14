#!/bin/zsh

# Run with 'source scripts/setup/install.sh'

conda deactivate
rm -rf /Users/panayao/mambaforge/envs/ml
pip3 cache purge

CONDA_SUBDIR=osx-arm64 conda create -n ml python=3.10 -c conda-forge
conda activate ml
conda env config vars set CONDA_SUBDIR=osx-arm64
echo "set CONDA_SUBDIR=osx-arm64 in 'ml' conda env ..."

conda deactivate
conda activate ml
echo "Finished installing conda 'ml' env !!!"
pip3 install --no-cache-dir -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install --no-cache-dir -e .
