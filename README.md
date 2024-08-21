
# A Geometric Notion of Causal Probing

This repository contains code for training and evaluation of A Causal Notion of Geometric Probing.

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

## Getting The Data

### Linzen
TBC

### UD Treebanks
Clone the following three UD french treebank repositories:
- https://github.com/UniversalDependencies/UD_French-GSD
- https://github.com/UniversalDependencies/UD_French-ParTUT
- https://github.com/UniversalDependencies/UD_French-Rhapsodie

## Setting Paths

Set the path variables in ```src/paths.py``` to match your folder structure, including where the data sources have been cloned.

## Data Preprocessing

Execute the following bash scripts and Python files in the order given:
- job_scripts/preprocess_cebab.sh
- job_scripts/preprocess_ud_fr.sh
- src/data/linzen/linzen_data.py
- job_scripts/morphoanalyze.sh
- src/data/spacy_wordlists/create_wordlists_en_cebab.py
- src/data/spacy_wordlists/create_wordlists_en_number.py
- src/data/spacy_wordlists/create_wordlists_fr_gender.py
- job_scripts/embedder.sh
- job_scripts/collect_hidden_states_batch_gpt2fr.sh
- job_scripts/collect_hidden_states_batch_gpt2l.sh
- job_scripts/collect_hidden_states_batch_llama2.sh
- job_scripts/process_hidden_states.sh
- job_scripts/process_hidden_states_fr.sh

## Generation to obtain (x, h, c) pairs

Execute the following bash scripts and Python files in the order given:
- job_scripts/generate.sh OR array_generate_gpts.sh and array_generate_llama.sh
- job_scripts/filter_generations.sh

## Training and Evaluating LEACE

Execute the following bash scripts and Python files in the order given:
- job_scripts/train_leace.sh
- job_scripts/multitoken_eval*.sh (three total)
- job_scripts/corr_multitoken_eval*.sh (three total)
- job_scripts/int_eval_*.sh (three total)
- src/evals/agg_mi_evals.sh
- src/evals/agg_int_evals.sh
- src/evals/control_plot.py

# Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{guerner-causalprobing-2023,
    title = "A Geometric Notion of Causal Probing",
    author = "Guerner, Clément  and
      Liu, Tianyu  and
      Svete, Anej and
      Warstadt, Alex and
      Cotterell, Ryan",
    journal={arXiv preprint arXiv:2307.15054},
    year={2023},
    url={https://arxiv.org/abs/2307.15054}
}
```