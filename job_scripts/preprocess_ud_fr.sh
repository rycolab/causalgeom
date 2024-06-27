#!/bin/bash

#SBATCH -n 4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=64G                 # per node!!
#SBATCH --job-name=preprocess_ud_fr
#SBATCH --output=preprocess_ud_fr.out
#SBATCH --error=preprocess_ud_fr.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

module load eth_proxy gcc/8.2.0 python_gpu/3.10.4

export TA_CACHE_DIR="/scratch/$USER/.cache"

source venv/bin/activate

python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_gsd -split train
python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_gsd -split dev
python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_gsd -split test

python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_partut -split train
python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_partut -split dev
python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_partut -split test

python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_rhapsodie -split train
python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_rhapsodie -split dev
python src/data/ud/ud_french_gender_preprocess.py -dataset ud_fr_rhapsodie -split test