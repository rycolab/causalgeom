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

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V, get_model


#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

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

#%%###############################
# Loading gender datasets        #
##################################
from utils.dataset_loaders import load_processed_data
from scipy.special import softmax, kl_div
from scipy.stats import entropy
from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from utils.dataset_loaders import load_hs, load_model_eval
from evals.kl_eval import load_run_output

model_name = "bert-base-uncased"
concept_name = "number"
run_output = os.path.join(OUT, "run_output/linzen/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")

logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

hs = load_hs(concept_name, model_name)
other_emb, l0_emb, l1_emb, pair_probs, concept_marginals = load_model_eval(concept_name, model_name)
P, I_P = load_run_output(run_output)

#%%
from evals.kl_eval import get_all_distribs
from evals.kl_eval import compute_kls_one_sample
h = hs[1]

base_distribs, P_distribs, I_P_distribs = get_all_distribs(
    h, P, I_P, other_emb, l0_emb, l1_emb
)

compute_kls_one_sample(h, P, I_P, other_emb, l0_emb, l1_emb, pair_probs, concept_marginals )
# %%
