#!/bin/bash

#SBATCH -n 4
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=16G                 
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=collect_hs_llama2
#SBATCH --output=collect_hs_llama2.out
#SBATCH --error=collect_hs_llama2.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES


source ./init_euler.sh

#python src/data/collect_hidden_states.py -dataset CEBaB -concept food -split train -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept food -split dev -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept food -split test -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept noise -split train -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept noise -split dev -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept noise -split test -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept ambiance -split train -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept ambiance -split dev -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept ambiance -split test -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept service -split train -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept service -split dev -model llama2 -batch_size 64
#python src/data/collect_hidden_states.py -dataset CEBaB -concept service -split test -model llama2 -batch_size 64

python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept food -split train -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept food -split dev -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept food -split test -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept noise -split train -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept noise -split dev -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept noise -split test -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept ambiance -split train -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept ambiance -split dev -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept ambiance -split test -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept service -split train -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept service -split dev -model llama2 -batch_size 64
python src/data/collect_hidden_states.py -dataset CEBaB_binary -concept service -split test -model llama2 -batch_size 64

#python src/data/collect_hidden_states.py -dataset linzen -concept number -model llama2 -batch_size 16
