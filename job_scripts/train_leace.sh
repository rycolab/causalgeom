#!/bin/bash

#SBATCH -n 4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=16G                 
#SBATCH --job-name=train_leace
#SBATCH --output=train_leace.out
#SBATCH --error=train_leace.err

source ./init_euler.sh

python src/algorithms/training_runner.py -concept food -model gpt2-large -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000
python src/algorithms/training_runner.py -concept ambiance -model gpt2-large -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000
python src/algorithms/training_runner.py -concept noise -model gpt2-large -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000
python src/algorithms/training_runner.py -concept service -model gpt2-large -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000

python src/algorithms/training_runner.py -concept food -model llama2 -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000
python src/algorithms/training_runner.py -concept ambiance -model llama2 -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000
python src/algorithms/training_runner.py -concept noise -model llama2 -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000
python src/algorithms/training_runner.py -concept service -model llama2 -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -val_nsamples 100000

python src/algorithms/training_runner.py -concept number -model gpt2-large -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -train_nsamples 100000 -val_nsamples 100000
python src/algorithms/training_runner.py -concept number -model llama2 -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -train_nsamples 100000 -val_nsamples 100000

python src/algorithms/training_runner.py -concept gender -model gpt2-base-french -proj_source natural_concept -seed 110 -out_folder june27 -nruns 3 -train_nsamples 100000 -val_nsamples 100000
