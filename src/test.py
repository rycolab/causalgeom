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

from scipy.special import softmax, kl_div

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V
from evals.kl_eval import load_hs, load_model_eval, load_run_output,\
    get_all_distribs, normalize_pairs
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Main             #
####################
#if __name__ == '__main__':


model_name = "bert-base-uncased"
dataset_name = "linzen"
gpt_run_output = os.path.join(OUT, "run_output/gpt2/230302/run_gpt2_k_1_plr_0.001_clflr_0.0003_bs_256_0_1.pkl")
bert_run_output = os.path.join(OUT, "run_output/bert-base-uncased/230303/run_bert_k_1_plr_0.003_clflr_0.003_bs_256_0_1.pkl")
if model_name == "bert-base-uncased":
    run_output = bert_run_output
elif model_name == "gpt2":
    run_output = gpt_run_output
else:
    run_output = None

logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")

hs = load_hs(dataset_name, model_name)
word_emb, sg_emb, pl_emb, verb_probs = load_model_eval(model_name)
#P, I_P = load_run_output(gpt_run_output)

#kls = compute_kls(hs, P, I_P, word_emb, sg_emb, pl_emb, verb_probs)
#kls.to_csv(os.path.join(OUT, "run_kls.csv"))


#%%
with open(run_output, 'rb') as f:      
    run = pickle.load(f)

#%%
P = run["output"]["P_burn"]
I_P = run["output"]["I_P_burn"]

#%%
from evals.kl_eval import compute_kls
burn_kl_eval = compute_kls(
    hs, P, I_P, 
    word_emb, sg_emb, pl_emb, verb_probs
)
burn_kl_means = burn_kl_eval.loc["mean",:]
burn_kl_eval.to_csv(os.path.join(OUT, "run_kls.csv"))

#%%
from numpy.random import normal
from evals.kl_eval import compute_kl, renormalize, compute_tvd

h = hs[5]
base_distribs = get_all_distribs(h, word_emb, sg_emb, pl_emb)

all_split = np.hstack([base_distribs["sg"], base_distribs["pl"], base_distribs["words"]])

sd = 1
noise = normal(0, sd, all_split.shape[0])
compute_kl(renormalize(all_split), renormalize(all_split*noise))

#%%
compute_tvd(renormalize(all_split), renormalize(all_split + noise))

#%%

#%%
from numpy.random import dirichlet

h = hs[4]
base_distribs = get_all_distribs(h, word_emb, sg_emb, pl_emb)

all_split = np.hstack([base_distribs["sg"], base_distribs["pl"], base_distribs["words"]])
alpha=99999999999999999999999999999
noise = dirichlet(alpha*all_split,3)



#%%


compute_kl(all_split, noise[1])

#%%
from scipy.stats import entropy

h = hs[2]

#%%
base_distribs = get_all_distribs(h, word_emb, sg_emb, pl_emb)
P_distribs = get_all_distribs(P @ h, word_emb, sg_emb, pl_emb)
I_P_distribs = get_all_distribs(I_P @ h, word_emb, sg_emb, pl_emb)

base_pair_probs = normalize_pairs(base_distribs["sg"], base_distribs["pl"])
P_pair_probs = normalize_pairs(P_distribs["sg"], P_distribs["pl"])
I_P_pair_probs = normalize_pairs(I_P_distribs["sg"], I_P_distribs["pl"])

# %%
pairwise_ent = np.mean(np.apply_along_axis(entropy, 1, verb_probs))
base_ent = np.mean(np.apply_along_axis(entropy, 1, base_pair_probs))
P_ent = np.mean(np.apply_along_axis(entropy, 1, P_pair_probs))
I_P_ent = np.mean(np.apply_along_axis(entropy, 1, I_P_pair_probs))

# %%
SG_PL_PROB = os.path.join(DATASETS, "processed/linzen_word_lists/sg_pl_prob.pkl")
with open(SG_PL_PROB, 'rb') as f:      
    #data_sg_pl_prob = pickle.load(f).to_numpy()
    data_sg_pl_prob = pickle.load(f)

data_ent = entropy(data_sg_pl_prob.to_numpy())


#sg_prob = base_distribs["sg"]
#pl_prob = base_distribs["pl"]
def get_sg_pl_prob(sg_prob, pl_prob):
    total_sg_prob = np.sum(sg_prob)
    total_pl_prob = np.sum(pl_prob)
    total_prob = total_sg_prob + total_pl_prob
    sg_pl_prob = np.hstack([total_sg_prob, total_pl_prob]) / total_prob
    return sg_pl_prob

base_sg_pl_prob = get_sg_pl_prob(base_distribs["sg"], base_distribs["pl"])
P_sg_pl_prob = get_sg_pl_prob(P_distribs["sg"], P_distribs["pl"])
I_P_sg_pl_prob = get_sg_pl_prob(I_P_distribs["sg"], I_P_distribs["pl"])

# %%
# compute the XMI: trueprob * probthat model outputs
ent_base = data_sg_pl_prob @ (-1 * np.log(base_sg_pl_prob))
ent_I_P = data_sg_pl_prob @ (-1 * np.log(I_P_sg_pl_prob))

entcond_base = np.array([0,1]) @ (-1 * np.log(base_sg_pl_prob))
entcond_I_P = np.array([0,1]) @ (-1 * np.log(I_P_sg_pl_prob))

