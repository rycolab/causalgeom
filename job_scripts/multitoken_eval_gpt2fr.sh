#!/bin/bash

#SBATCH --job-name=mt_gpt2fr
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G                 
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/cluster/work/cotterell/cguerner/usagebasedprobing/out/slurm_outputs/mt_eval_gpt2fr_june27/multitoken_eval_%A_%a.out

source ./init_euler.sh

export DISABLE_TQDM=True

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

list_of_jobs=()

model_name='gpt2-base-french'
nsamples=100
msamples=1000
run_output_folder='june27'
output_folder='june27'
n_other_words=1
batch_size=16
eval_source='test_concept'
proj_source='natural_concept'
concept='gender'

# 2 x 3 = 6

#for proj_source in 'gen_nucleus_all' 'gen_ancestral_all'; do

run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${run_output_folder}/${concept}/${model_name}/${proj_source}"

for run_path in "$run_dir"/*; do

echo "Computing distributions for run file ${run_path}"

#if [ "$run_output_folder" = "leacegen_ancestral" ] ; then
#	source='gen_ancestral_all'
#else
#	source='gen_nucleus_all'
#fi

jobargs="-model $model_name -concept $concept -eval_source $eval_source -nsamples $nsamples -msamples $msamples -out_folder $output_folder -run_path $run_path -batch_size $batch_size -n_other_words $n_other_words"

list_of_jobs+=("${jobargs}")
done
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
