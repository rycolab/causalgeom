#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import torch

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT


from utils.lm_loaders import get_tokenizer, get_V, get_model
from utils.cuda_loaders import get_device

#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



#%%#################
# Generating Text  #
####################
#from transformers import AutoTokenizer
#from transformers import LogitsWarper, LogitsProcessorList, TopPLogitsWarper
from evals.kl_eval import load_run_output

#torch.cuda.empty_cache()

model_name = "gpt2-large"

device = get_device()
#device = "cpu"
model = get_model(model_name).to(device)
tokenizer = get_tokenizer(model_name)

#TODO: batch this, check that once EOS has been generated only EOS is then generated
prompt_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)

if model_name == "gpt2-large":
    run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")
else:
    run_output = None

P, I_P = load_run_output(run_output)
I_P = torch.tensor(I_P, device=device).float()

#%%

def generate_sequence_until_eos(model, prompt_ids, device="cpu", P=None):
    tokens = prompt_ids
    token_h_pairs = []
    sm = torch.nn.Softmax(0)#.to(device)
    while (tokens[-1][-1] != tokenizer.eos_token_id or (tokens.shape[1] == prompt_ids.shape[1])):# and tokens.shape[1] < 10:
        if tokens.shape[1] > 1024:
            input_tokens = tokens[:,-1024:]
        else:
            input_tokens = tokens
        with torch.no_grad():
            output = model(
                input_ids=input_tokens,
                output_hidden_states=True
            )
        #final_logits = output.logits[-1]#.cpu()
        hs = output.hidden_states[-1][-1,-1,:].squeeze().squeeze()#.cpu()
        if I_P is not None:
            hs = torch.matmul(I_P, hs)
        check_logits = model.lm_head(hs)
        #assert (torch.isclose(final_logits[-1,:], check_logits, atol = 1e-05)).all().item(), "Logits not equal"
        probs = sm(check_logits)
        sampled_token = torch.multinomial(probs, 1)
        tokens = torch.cat((tokens, sampled_token.unsqueeze(0)), dim=-1)
        logging.info(f"Sampled token {sampled_token}, number of tokens so far: {tokens.shape[1]}")
        token_h_pairs.append((hs.cpu(), sampled_token.item()))
    return tokens.cpu(), token_h_pairs

#tokens, token_h_pairs = generate_sequence_until_eos(model, prompt_ids, device)
#I_P_tokens, I_P_token_h_pairs = generate_sequence_until_eos(model, prompt_ids, device, P=I_P)

#%%
def generate_multiple_sequences(model, prompt_ids, device="cpu", P=None):
    export_index = 0
    n_generations = 3
    all_token_h_pairs = []
    all_lengths = []
    for i in range(n_generations):
        _, token_h_pairs = generate_sequence_until_eos(model, prompt_ids, device)
        all_lengths.append(len(token_h_pairs))
        all_token_h_pairs += token_h_pairs
    export_path = os.path.join(OUT, f"generated_text/{model_name}/generations_{export_index}.pkl")
    with open(export_path, 'wb') as f:
        pickle.dump(all_token_h_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

#%%


#%%#################
# Computing new MI #
####################
from utils.dataset_loaders import load_processed_data
from scipy.special import softmax, kl_div
from scipy.stats import entropy
from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from utils.dataset_loaders import load_hs, load_model_eval
from evals.kl_eval import load_run_output

model_name = "gpt2-large" #"bert-base-uncased"
concept_name = "number"
#run_output = os.path.join(OUT, "run_output/linzen/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")
run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")

logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

hs = load_hs(concept_name, model_name)
other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept_name, model_name)
P, I_P = load_run_output(run_output)

#%%
from evals.kl_eval import get_all_distribs, get_all_marginals, get_lemma_marginals
from evals.kl_eval import compute_kls_one_sample
h = hs[13]

base_distribs, P_distribs, I_P_distribs = get_all_distribs(
    h, P, I_P, other_emb, l0_emb, l1_emb
)

distrib = base_distribs
cond_all_marginals = get_all_marginals(
    distrib["l0"], distrib["l1"], distrib["other"]
)
print(cond_all_marginals)

distrib = I_P_distribs
cond_all_marginals = get_all_marginals(
    distrib["l0"], distrib["l1"], distrib["other"]
)
print(cond_all_marginals)


#compute_kls_one_sample(h, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals )
# %%
from evals.kl_eval import compute_overall_mi
all_marginals = [
    concept_marginals["p_0_incl_other"], 
    concept_marginals["p_1_incl_other"], 
    concept_marginals["p_other_incl_other"]
]

compute_overall_mi(concept_marginals, distrib["l0"], distrib["l1"], distrib["other"])

#%%#################
# Camembert        #
####################
model = get_model("camembert-base")
tokenizer = get_tokenizer("camembert-base")
print(model)

inputs = tokenizer("La capitale de la France est <mask>", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs, output_hidden_states=True)

# %%
from data.dataset_loaders import load_model_eval

#model_name = "gpt2-base-french"
#dataset_name = "ud_fr_gsd"
model_name = "gpt2"
dataset_name = "linzen"
model = get_model(model_name)
word_emb, l0_emb, l1_emb, lemma_prob, concept_prob = load_model_eval(dataset_name, model_name)
# %%
from scipy.stats import entropy
entropy(concept_prob)
