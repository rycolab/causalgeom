#!/bin/bash

#SBATCH --job-name=filter_generations
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-22
#SBATCH --output=/cluster/work/cotterell/cguerner/usagebasedprobing/out/slurm_outputs/filter_generations_3105/filter_gens_%A_%a.out

export DISABLE_TQDM=True

source ./init_euler.sh

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

list_of_jobs=()

# 5 x 2 x 2 = 20 jobs
for concept in 'food' 'service' 'ambiance' 'noise' 'number'; do
#for concept in 'food' 'service' 'ambiance' 'noise'; do #NONUMBER
for model_name in 'gpt2-large' 'llama2'; do
#model_name='llama2'
for nucleus in true false; do
#nucleus=true
if [ "$nucleus" = true ]; then
jobargs="-model $model_name -concept $concept -nucleus"
else
jobargs="-model $model_name -concept $concept"
fi
list_of_jobs+=("${jobargs}")
done
done
done

# two jobs
for nucleus in true false; do
#nucleus=false
if [ "$nucleus" = true ]; then
jobargs="-model gpt2-base-french -concept gender -nucleus"
else
jobargs="-model gpt2-base-french -concept gender"
fi
list_of_jobs+=("${jobargs}")
done

num_jobs=${#list_of_jobs[@]}

job_id=${SLURM_ARRAY_TASK_ID}

if [ ${job_id} -ge ${num_jobs} ] ; then
echo "Invalid job id; qutting"
exit 2
fi

echo "-------- STARTING JOB ${job_id}/${num_jobs}"

args=${list_of_jobs[${job_id}]}

python src/data/GenerationsFilter.py ${args}

echo "Job completed"


