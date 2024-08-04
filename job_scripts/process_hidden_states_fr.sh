#!/bin/bash

#SBATCH -n 1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=64G                 # per node!!
#SBATCH --job-name=process_hidden_states_fr
#SBATCH --output=process_hidden_states_fr.out
#SBATCH --error=process_hidden_states_fr.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

source ./init_euler.sh

python src/data/process_hidden_states.py -dataset ud_fr_gsd -split train -model gpt2-base-french -outtype full -concept gender 
python src/data/process_hidden_states.py -dataset ud_fr_gsd -split dev -model gpt2-base-french -outtype full -concept gender 
python src/data/process_hidden_states.py -dataset ud_fr_gsd -split test -model gpt2-base-french -outtype full -concept gender 

python src/data/process_hidden_states.py -dataset ud_fr_partut -split train -model gpt2-base-french -outtype full -concept gender 
python src/data/process_hidden_states.py -dataset ud_fr_partut -split dev -model gpt2-base-french -outtype full -concept gender 
python src/data/process_hidden_states.py -dataset ud_fr_partut -split test -model gpt2-base-french -outtype full -concept gender 

python src/data/process_hidden_states.py -dataset ud_fr_rhapsodie -split train -model gpt2-base-french -outtype full -concept gender 
python src/data/process_hidden_states.py -dataset ud_fr_rhapsodie -split dev -model gpt2-base-french -outtype full -concept gender 
python src/data/process_hidden_states.py -dataset ud_fr_rhapsodie -split test -model gpt2-base-french -outtype full -concept gender 

