#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=16G
#SBATCH --gpus=3
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=/cluster/work/cotterell/cguerner/usagebasedprobing/out/slurm_outputs/clem_new_generate_llama2/generate_%A_%a.out

export DISABLE_TQDM=True

source ./init_euler.sh

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

list_of_jobs=()

# 3 x 2
for export_index in 1 2 3; do
model_name='llama2'
for nucleus in true false; do
if [ "$nucleus" = true ]; then
jobargs="-model $model_name -export_index $export_index -nucleus"
else
jobargs="-model $model_name -export_index $export_index"
fi
list_of_jobs+=("${jobargs}")
done
done


num_jobs=${#list_of_jobs[@]}

job_id=${SLURM_ARRAY_TASK_ID}

if [ ${job_id} -ge ${num_jobs} ] ; then
echo "Invalid job id; qutting"
exit 2
fi

echo "-------- STARTING JOB ${job_id}/${num_jobs}"

args=${list_of_jobs[${job_id}]}

python src/data/generate.py ${args} 

echo "Job completed"


