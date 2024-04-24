#!/bin/bash

module load eth_proxy gcc/8.2.0 python_gpu/3.11.2
#eval "$(micromamba shell hook --shell )"
#micromamba activate env2

#if [ ! -d "venv/" ]; then
#    python3 -m venv venv
#    echo "Created virtual environment"
#fi

source /cluster/home/cguerner/python_venvs/usagebasedprobing/bin/activate
python -m pip install -r requirements.txt 
echo "Installed requirements"
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

