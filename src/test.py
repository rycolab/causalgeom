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
import random 

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST
from utils.cuda_loaders import get_device
from evals.kl_eval import load_run_output
from utils.dataset_loaders import load_processed_data
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#####################
# Creating Nice Graphs #
########################
model_name = "gpt2-large"
concept_name = "number"

#resdir = os.path.join(RESULTS, f"{concept_name}/{model_name}")
#I_P_fth_res_path = os.path.join(resdir, f"fth_res_I_P.csv")

#I_P_kls = pd.read_csv(I_P_fth_res_path)

raw_resdir = os.path.join(OUT, "raw_results")
concept_fth_raw_res_path = os.path.join(raw_resdir, f"kl_mi_concept_{model_name}_{concept_name}.csv")
concept_kls = pd.read_csv(concept_fth_raw_res_path)
other_fth_raw_res_path = os.path.join(raw_resdir, f"kl_mi_other_{model_name}_{concept_name}.csv")
other_kls = pd.read_csv(other_fth_raw_res_path)

#%%
I_P_kls = I_P_kls[I_P_kls["distance_metric"] == "kl"]


#%%#################
# Computing correct MI #
####################
from models.fit_kde import load_data

model_name = "gpt2-large"
I_P = "I_P"
X, weights = load_data(model_name, I_P)

#%%
#TODO: OR SAMPLE FROM THE KERNEL
nsamples = 200
probs = weights / np.sum(weights)
idx = np.arange(0, X.shape[0])
sampled_idx = np.random.choice(idx, size=nsamples, p=probs)
X_samples = X[sampled_idx]

#%%
kde_path = os.path.join(MODELS, f"kde/kde_{model_name}_{I_P}.pkl")
with open(kde_path, "rb") as f:
    kde = pickle.load(f)

#%%
logp_hs = kde.score_samples(X_samples)

#%%#################
# Computing new MI #
####################
"""
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

with open(run_output, 'rb') as f:      
    run = pickle.load(f)

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
#%%
from scipy.stats import entropy

entropy(concept_prob)
"""