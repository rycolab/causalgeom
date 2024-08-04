sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2fr --wrap "python src/data/generate.py -model gpt2-base-french -export_index 18" 
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2fr --wrap "python src/data/generate.py -model gpt2-base-french -export_index 19" 
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2fr --wrap "python src/data/generate.py -model gpt2-base-french -export_index 20"

sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2fr_nuc --wrap "python src/data/generate.py -model gpt2-base-french -nucleus -export_index 189"
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2fr_nuc --wrap "python src/data/generate.py -model gpt2-base-french -nucleus -export_index 190"
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2fr_nuc --wrap "python src/data/generate.py -model gpt2-base-french -nucleus -export_index 191"

sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2l --wrap "python src/data/generate.py -model gpt2-large -export_index 18" #52926232
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2l --wrap "python src/data/generate.py -model gpt2-large -export_index 319" #52938283
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2l --wrap "python src/data/generate.py -model gpt2-large -export_index 320" #52938295

sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2l_nuc --wrap "python src/data/generate.py -model gpt2-large -nucleus -export_index 189" #running
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2l_nuc --wrap "python src/data/generate.py -model gpt2-large -nucleus -export_index 190"
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=1 --gres=gpumem:20g --job-name=gen_gpt2l_nuc --wrap "python src/data/generate.py -model gpt2-large -nucleus -export_index 191"

sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=3 --gres=gpumem:20g --job-name=gen_llama2 --wrap "python src/data/generate.py -model llama2 -export_index 101" #52926407
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=3 --gres=gpumem:20g --job-name=gen_llama2 --wrap "python src/data/generate.py -model llama2 -export_index 111" #52938286
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=3 --gres=gpumem:20g --job-name=gen_llama2 --wrap "python src/data/generate.py -model llama2 -export_index 121" #52938297

sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=3 --gres=gpumem:20g --job-name=gen_llama2_nuc --wrap "python src/data/generate.py -model llama2 -nucleus -export_index 143" #52926460
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=3 --gres=gpumem:20g --job-name=gen_llama2_nuc --wrap "python src/data/generate.py -model llama2 -nucleus -export_index 123" #52938290
sbatch --ntasks=4 --time=24:00:00 --mem-per-cpu=16G --tmp=16G --gpus=3 --gres=gpumem:20g --job-name=gen_llama2_nuc --wrap "python src/data/generate.py -model llama2 -nucleus -export_index 133"
