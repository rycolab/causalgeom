
# causalgeom

Stuff to run:

1. Data:
- job_scripts/preprocess_cebab.sh
- job_scripts/preprocess_ud_fr.sh
- data/linzen/linzen_data.py
- job_scripts/morphoanalyze.sh
- data/spacy_wordlists/create_wordlists_en_cebab.py
- data/spacy_wordlists/create_wordlists_en_number.py
- data/spacy_wordlists/create_wordlists_fr_gender.py
- job_scripts/embedder.sh
- job_scripts/generate.sh OR array_generate_gpts.sh and array_generate_llama.sh
- job_scripts/filter_generations.sh
- job_scripts/collect_hidden_states_batch_gpt2fr.sh
- job_scripts/collect_hidden_states_batch_gpt2l.sh
- job_scripts/collect_hidden_states_batch_llama2.sh
- job_scripts/process_hidden_states.sh
- job_scripts/process_hidden_states_fr.sh

2. LEACE and eval:
- job_scripts/train_leace.sh
- job_scripts/multitoken_eval*.sh (three total)
- job_scripts/corr_multitoken_eval*.sh (three total)
- job_scripts/int_eval_*.sh (three total)
- evals/agg_mi_evals.sh
- evals/agg_int_evals.sh
- TODO: add plotting scripts

