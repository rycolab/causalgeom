#!/bin/bash

#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=64G                 # per node!!
#SBATCH --job-name=process_hidden_states
#SBATCH --output=process_hidden_states.out
#SBATCH --error=process_hidden_states.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

source ./init_euler.sh

#python src/data/process_hidden_states.py -dataset CEBaB -concept food -split train -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept food -split dev -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept food -split test -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept noise -split train -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept noise -split dev -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept noise -split test -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept ambiance -split train -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept ambiance -split dev -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept ambiance -split test -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept service -split train -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept service -split dev -model llama2 -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept service -split test -model llama2 -outtype full

#python src/data/process_hidden_states.py -dataset CEBaB -concept food -split train -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept food -split dev -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept food -split test -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept noise -split train -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept noise -split dev -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept noise -split test -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept ambiance -split train -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept ambiance -split dev -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept ambiance -split test -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept service -split train -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept service -split dev -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset CEBaB -concept service -split test -model gpt2-large -outtype full

python src/data/process_hidden_states.py -dataset CEBaB_binary -concept food -split train -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept food -split dev -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept food -split test -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept noise -split train -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept noise -split dev -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept noise -split test -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept ambiance -split train -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept ambiance -split dev -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept ambiance -split test -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept service -split train -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept service -split dev -model llama2 -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept service -split test -model llama2 -outtype full

python src/data/process_hidden_states.py -dataset CEBaB_binary -concept food -split train -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept food -split dev -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept food -split test -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept noise -split train -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept noise -split dev -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept noise -split test -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept ambiance -split train -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept ambiance -split dev -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept ambiance -split test -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept service -split train -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept service -split dev -model gpt2-large -outtype full
python src/data/process_hidden_states.py -dataset CEBaB_binary -concept service -split test -model gpt2-large -outtype full


#python src/data/process_hidden_states.py -dataset linzen -concept number -model gpt2-large -outtype full
#python src/data/process_hidden_states.py -dataset linzen -concept number -model llama2 -outtype full
