#!/bin/bash

module load eth_proxy gcc/8.2.0 python_gpu/3.10.4

eval "$(ssh-agent -s)"
ssh-add $HOME/.ssh/id_ed25519_euler_remote

if [ ! -d "venv/" ]; then
    python3 -m venv venv
    echo "Created virtual environment."
fi

export TA_CACHE_DIR="/scratch/$USER/.cache"

source venv/bin/activate
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116