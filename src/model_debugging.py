#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import pickle
from tqdm import tqdm, trange
import shutil

from scipy.special import softmax, kl_div
from scipy.stats import entropy

from utils.lm_loaders import get_model, get_tokenizer, get_V
from evals.kl_eval import load_model_eval, get_distribs, compute_overall_mi, \
     get_sg_pl_prob, get_all_distribs, get_all_pairwise_distribs, compute_kls, \
        load_hs, normalize_pairs
from algorithms.rlace.rlace import init_classifier


from paths import DATASETS, OUT

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# TESTING ENV      #
####################
model_name = "bert-base-uncased" # gpt2
dataset_name = "linzen"
#batchfile=os.path.join(OUT, f"hidden_states/linzen/{model_name}/batch_0.pkl")
gpt2_run_output = os.path.join(OUT, "run_output/gpt2/230411_pca/run_gpt2_Pms11,16,21,26,31,36_Pg0.5_clfms21,31_clfg0.5_0_1.pkl")
bert_run_output = os.path.join(OUT, "run_output/bert-base-uncased/230310/run_bert_k_1_0_1.pkl")

if model_name == "gpt2":
    run_output = gpt2_run_output
elif model_name == "bert-base-uncased":
    run_output = bert_run_output
else:
    raise ValueError("Incorrect model name")

#%%#################
# LOADING DATA     #
####################
#with open(batchfile, 'rb') as f:      
#    batch_data = pickle.load(f)

hs = load_hs(dataset_name, model_name)
word_emb, sg_emb, pl_emb, verb_probs, sg_pl_probs = load_model_eval(model_name, add_space=(model_name == "gpt2"))
TOKENIZER = get_tokenizer(model_name)

with open(run_output, 'rb') as f:      
    run = pickle.load(f)

P = run["output"]["P"]
I_P = run["output"]["I_P"]
if model_name == "gpt2":
    X_pca = run["X_pca"]
else:
    X_pca = None

#%%######################
# TESTING DIAG EVAL     #
#########################

#model_name = "bert-base-uncased"
if model_name == "gpt2":
    DATASET = os.path.join(DATASETS, f"processed/linzen_{model_name}_ar.pkl")
elif model_name == "bert-base-uncased":
    DATASET = os.path.join(DATASETS, f"processed/linzen_{model_name}_masked.pkl")
else:
    DATASET = None

with open(DATASET, 'rb') as f:      
    data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])

X = np.array([x for x in data["h"]])
U = np.array([x for x in data["u"]])
y = np.array([yi for yi in data["y"]])
del data

train_lastind = 50000
val_lastind = train_lastind + 20000
test_lastind = val_lastind + 20000
idx = np.arange(0, X.shape[0])
np.random.shuffle(idx)
X_train, X_val, X_test = X[idx[:train_lastind]], X[idx[train_lastind:val_lastind]], X[idx[val_lastind:test_lastind]]
U_train, U_val, U_test = U[idx[:train_lastind]], U[idx[train_lastind:val_lastind]], U[idx[val_lastind:test_lastind]]
y_train, y_val, y_test = y[idx[:train_lastind]], y[idx[train_lastind:val_lastind]], y[idx[val_lastind:test_lastind]]

#svm = init_classifier()
#svm.fit(X_train, y_train)

#svm.score(X_train, y_train)
#svm.score(X_val, y_val)
#svm.score(X_test, y_test)

#%%######################
# TESTING WHOLE KL EVAL #
#########################
kl_eval = compute_kls(h, P, I_P, word_emb, sg_emb, pl_emb, verb_probs, 
    sg_pl_probs, X_pca=X_pca
)

#%%##############################
# TESTING INDIVIDUAL H DISTRIBS #
#################################
h = hs[500006]
#base_distribs, P_distribs, I_P_distribs = get_all_distribs(h, P, I_P, word_emb, sg_emb, pl_emb)
#base_pairs, P_pairs, I_P_pairs = get_all_pairwise_distribs(base_distribs, P_distribs, I_P_distribs)

base_distribs, P_distribs, I_P_distribs = get_all_distribs(h, P, I_P, word_emb, sg_emb, pl_emb, X_pca)
base_pairs, P_pairs, I_P_pairs = get_all_pairwise_distribs(base_distribs, P_distribs, I_P_distribs)
#base_pairs = normalize_pairs(base_distribs["sg"], base_distribs["pl"])

#%%
TOKENIZER = get_tokenizer(model_name)
V = get_V(model_name)
probs = softmax(V @ hs[1000001])
np.argpartition(probs,-5)[-5:]

#%%############################
# TESTING MULTIPLE H DISTRIBS #
###############################
base_mis, P_mis, I_P_mis = [], [], []
base_ents, P_ents, I_P_ents = [], [], []
concept_ent = entropy(sg_pl_probs)
h = hs[1000000:1000200]
for i in trange(h.shape[0]):
    base_distribs, P_distribs, I_P_distribs = get_all_distribs(h[i], P, I_P, word_emb, sg_emb, pl_emb, X_pca)
    base_pairs, P_pairs, I_P_pairs = get_all_pairwise_distribs(base_distribs, P_distribs, I_P_distribs)
    base_pairs = normalize_pairs(base_distribs["sg"], base_distribs["pl"])
    base_sgpl = get_sg_pl_prob(base_distribs["sg"], base_distribs["pl"])
    P_sgpl = get_sg_pl_prob(P_distribs["sg"], P_distribs["pl"])
    I_P_sgpl = get_sg_pl_prob(I_P_distribs["sg"], I_P_distribs["pl"])
    base_ent = entropy(base_sgpl)
    P_ent = entropy(P_sgpl)
    I_P_ent = entropy(I_P_sgpl)
    base_ents.append(base_ent)
    P_ents.append(P_ent)
    I_P_ents.append(I_P_ent)
    base_mis.append(concept_ent - base_ent)
    P_mis.append(concept_ent - P_ent)
    I_P_mis.append(concept_ent - I_P_ent)

print(f"\n Base ent mean: {np.mean(base_ents)}, std: {np.std(base_ents)}")
print(f"\n P ent mean: {np.mean(P_ents)}, std: {np.std(P_ents)}")
print(f"\n I_P ent mean: {np.mean(I_P_ents)}, std: {np.std(I_P_ents)}")
print(f"\n Base MI mean: {np.mean(base_mis)}, std: {np.std(base_mis)}")
print(f"\n P MI mean: {np.mean(P_mis)}, std: {np.std(P_mis)}")
print(f"\n I_P MI mean: {np.mean(I_P_mis)}, std: {np.std(I_P_mis)}")

#%%
import seaborn as sns 

sns.scatterplot(x=base_ents, y=I_P_ents)

#%%############################
# TESTING MULTIPLE H DISTRIBS #
###############################
h = hs[500000:500200]

count = 0
indices = []
for i in trange(hs.shape[0]):
    if (h == hs[i]).all():
        count += 1
        indices.append(i)
print(count/i)

#%%############################
# DATA TESTING (DELETE) #
###############################
from data.dataset_loaders import load_dataset

data = load_dataset("linzen", "gpt2", "train")

# %%
