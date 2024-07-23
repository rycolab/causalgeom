#!/bin/bash

#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=16G                 # per node!!
#SBATCH --job-name=preprocess_cebab
#SBATCH --output=preprocess_cebab.out
#SBATCH --error=preprocess_cebab.err


source ./init_euler.sh

python src/data/cebab/preprocess_cebab.py -concept food
python src/data/cebab/preprocess_cebab.py -concept ambiance
python src/data/cebab/preprocess_cebab.py -concept service
python src/data/cebab/preprocess_cebab.py -concept noise

python src/data/cebab/preprocess_cebab.py -concept food -binary
python src/data/cebab/preprocess_cebab.py -concept ambiance -binary
python src/data/cebab/preprocess_cebab.py -concept service -binary
python src/data/cebab/preprocess_cebab.py -concept noise -binary
