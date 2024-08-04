#!/bin/bash

#SBATCH --job-name=mteval_distribs
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G                 
#SBATCH --time=4:00:00
#SBATCH --array=0-33
#SBATCH --output=/cluster/work/cotterell/cguerner/usagebasedprobing/out/slurm_outputs/micomputer_all/micomputer_eval_%A_%a.out

source ./init_euler.sh

export DISABLE_TQDM=True

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

list_of_jobs=()

mt_eval_output_folder='otherwords_mt_eval'
run_output_folder='leace05042024'
source='natural'
#for nucleus in true false; do

# 2 x 5 x 3 = 30
for model_name in 'gpt2-large' 'llama2'; do

#for concept in 'food' 'service' 'ambiance' 'noise' 'number'; do
concept='number'

run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${concept}/${model_name}/${run_output_folder}"

for run_path in "$run_dir"/*; do

echo "Computing distributions for run file ${run_path}"

jobargs="-model $model_name -concept $concept -source $source -out_folder $mt_eval_output_folder -run_path $run_path"
list_of_jobs+=("${jobargs}")

done
done
#done
#done
#done

# GENDER RUNS: 3
model_name='gpt2-base-french'
concept='gender'

run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${concept}/${model_name}/${run_output_folder}"

for run_path in "$run_dir"/*; do

echo "Computing distributions for run file ${run_path}"

jobargs="-model $model_name -concept $concept -source $source -out_folder $mt_eval_output_folder -run_path $run_path"
list_of_jobs+=("${jobargs}")

done
#done
#done


num_jobs=${#list_of_jobs[@]}

job_id=${SLURM_ARRAY_TASK_ID}

if [ ${job_id} -ge ${num_jobs} ] ; then
echo "Invalid job id; qutting"
exit 2
fi

echo "-------- STARTING JOB ${job_id}/${num_jobs}"

args=${list_of_jobs[${job_id}]}

python src/evals/MIComputer.py ${args}

echo "Job completed"
