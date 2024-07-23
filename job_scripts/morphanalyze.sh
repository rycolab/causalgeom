#!/bin/bash

#SBATCH -n 4
#SBATCH --account=es.sachan
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=64G                 # per node!!
#SBATCH --job-name=spacy_analyze_fr
#SBATCH --output=spacy_analyze_fr.out
#SBATCH --error=spacy_analyze_fr.err

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

module load eth_proxy gcc/8.2.0 python_gpu/3.10.4

export TA_CACHE_DIR="/scratch/$USER/.cache"

source venv/bin/activate

python src/spacy_morpho.py -language fr


sbatch --account=es_cott --ntasks=4 --time=24:00:00 --mem-per-cpu=64G --tmp=64G --gpus=1 --gres=gpumem:10g --wrap "python src/data/spacy_wordlists/morphoanalyze.py -language fr"
sbatch --account=es_cott --ntasks=4 --time=24:00:00 --mem-per-cpu=64G --tmp=64G --gpus=1 --gres=gpumem:10g --wrap "python src/data/spacy_wordlists/morphoanalyze.py -language en"

