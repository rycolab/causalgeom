#!/bin/bash

#SBATCH -n 4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=16G      
#SBATCH --gpus=2
#SBATCH --gres=gpumem:20g           
#SBATCH --job-name=embedder
#SBATCH --output=embedder.out
#SBATCH --error=embedder.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES


source ./init_euler.sh

python src/data/spacy_wordlists/embedder.py -concept food -model gpt2-large
python src/data/spacy_wordlists/embedder.py -concept food -model llama2
python src/data/spacy_wordlists/embedder.py -concept ambiance -model gpt2-large
python src/data/spacy_wordlists/embedder.py -concept ambiance -model llama2
python src/data/spacy_wordlists/embedder.py -concept noise -model gpt2-large
python src/data/spacy_wordlists/embedder.py -concept noise -model llama2
python src/data/spacy_wordlists/embedder.py -concept service -model gpt2-large
python src/data/spacy_wordlists/embedder.py -concept service -model llama2

python src/data/spacy_wordlists/embedder.py -concept gender -model gpt2-base-french
python src/data/spacy_wordlists/embedder.py -concept number -model gpt2-large
python src/data/spacy_wordlists/embedder.py -concept number -model llama2



