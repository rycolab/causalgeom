#!/bin/bash

#SBATCH --job-name=int_llama
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G                 
#SBATCH --gpus=rtx_4090:3
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/cluster/work/cotterell/cguerner/usagebasedprobing/out/slurm_outputs/int_eval_llama2_june15/multitoken_eval_%A_%a.out

source ./init_euler.sh

export DISABLE_TQDM=True

echo "Running [ ${0} ${@} ] on $(hostname), starting at $(date)"
echo "Job id = ${SLURM_JOB_ID}, task id = ${SLURM_ARRAY_TASK_ID}"
echo "PWD = $(pwd)"

set -exu

list_of_jobs=()

model_name='llama2'
concept='number'
int_source='test'
nsamples=100
msamples=50
output_folder='int_june27'
batch_size=6
torch_dtype='bfloat16'

# Number - GENERATED
# 2 x 3 = 6
#run_output_folder='june2'

#for proj_source in 'gen_nucleus_all' 'gen_ancestral_all'; do

#run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${run_output_folder}/${concept}/${model_name}/${proj_source}"

#for run_path in "$run_dir"/*; do

#echo "Computing distributions for run file ${run_path}"

#jobargs="-model $model_name -concept $concept -int_source $int_source -nsamples $nsamples -msamples $msamples -run_path $run_path -out_folder $output_folder -batch_size $batch_size -torch_dtype $torch_dtype"

#list_of_jobs+=("${jobargs}")
#done
#done

# Number - GENERATED
# 3
run_output_folder='june27'
proj_source='natural_concept'

run_dir="/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/${run_output_folder}/${concept}/${model_name}/${proj_source}"

for run_path in "$run_dir"/*; do

echo "Computing distributions for run file ${run_path}"

jobargs="-model $model_name -concept $concept -int_source $int_source -nsamples $nsamples -msamples $msamples -run_path $run_path -out_folder $output_folder -batch_size $batch_size -torch_dtype $torch_dtype"

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

python src/evals/MultiTokenIntervenor.py ${args}

echo "Job completed"
