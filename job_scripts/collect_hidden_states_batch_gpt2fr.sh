#!/bin/bash

#SBATCH -n 4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=16G                 
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=collect_hs_gpt2fr
#SBATCH --output=collect_hs_gpt2fr.out
#SBATCH --error=collect_hs_gpt2fr.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

source ./init_euler.sh

python src/data/collect_hidden_states.py -dataset ud_fr_gsd -split train -model gpt2-base-french -concept gender
python src/data/collect_hidden_states.py -dataset ud_fr_gsd -split dev -model gpt2-base-french -concept gender
python src/data/collect_hidden_states.py -dataset ud_fr_gsd -split test -model gpt2-base-french -concept gender

python src/data/collect_hidden_states.py -dataset ud_fr_partut -split train -model gpt2-base-french -concept gender
python src/data/collect_hidden_states.py -dataset ud_fr_partut -split dev -model gpt2-base-french -concept gender
python src/data/collect_hidden_states.py -dataset ud_fr_partut -split test -model gpt2-base-french -concept gender

python src/data/collect_hidden_states.py -dataset ud_fr_rhapsodie -split train -model gpt2-base-french -concept gender
python src/data/collect_hidden_states.py -dataset ud_fr_rhapsodie -split dev -model gpt2-base-french -concept gender
python src/data/collect_hidden_states.py -dataset ud_fr_rhapsodie -split test -model gpt2-base-french -concept gender
