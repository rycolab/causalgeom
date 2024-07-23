#!/bin/bash

#SBATCH --job-name=mt_llama
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=30G                 
#SBATCH --gpus=rtx_4090:3
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=/cluster/work/cotterell/cguerner/usagebasedprobing/out/slurm_outputs/mt_eval_llama2_june27/multitoken_eval_%A_%a.out

source ./init_euler.sh

export DISABLE_TQDM=True

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

list_of_jobs=()

model_name='llama2'
nsamples=100
msamples=1000
output_folder='june27'
n_other_words=1
torch_dtype='bfloat16'
batch_size=6
run_output_folder='june27'
eval_source='test_concept'
proj_source='natural_concept'

### CEBAB
# 4 x 3 = 12

for concept in 'food' 'ambiance' 'noise' 'service'; do

run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${run_output_folder}/${concept}/${model_name}/${proj_source}"

for run_path in "$run_dir"/*; do

echo "Computing distributions for run file ${run_path}"

jobargs="-model $model_name -concept $concept -eval_source $eval_source -nsamples $nsamples -msamples $msamples -out_folder $output_folder -run_path $run_path -n_other_words $n_other_words -batch_size $batch_size -torch_dtype $torch_dtype"

list_of_jobs+=("${jobargs}")
done
done

# Number
# 3
concept='number'

run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${run_output_folder}/${concept}/${model_name}/${proj_source}"

for run_path in "$run_dir"/*; do

echo "Computing distributions for run file ${run_path}"

jobargs="-model $model_name -concept $concept -eval_source $eval_source -nsamples $nsamples -msamples $msamples -out_folder $output_folder -run_path $run_path -n_other_words $n_other_words -batch_size $batch_size -torch_dtype $torch_dtype"

list_of_jobs+=("${jobargs}")
done

#done
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

python src/evals/mt_eval_runner.py ${args}

echo "Job completed"
