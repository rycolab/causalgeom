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

from paths import DATASETS, OUT, RESULTS


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

#%%#############################
# Computing Generation p(x, h) #
################################
model_name = "gpt2-large"
I_P = "I_P"
generations_folder = os.path.join(OUT, f"generated_text/{model_name}/{I_P}")
files = os.listdir(generations_folder)

#%%
def identify_matches(all_hs, new_hs):
    all_index = 0
    match_index = torch.ones(new_hs.shape[0]) * -1
    for i, new_h in enumerate(new_hs):
        stop = False
        while not stop:
            close = torch.isclose(all_hs[all_index,:], new_h, 1E-5).all().item()
            if close:
                match_index[i] = all_index
                stop = True
            elif ((new_h[0] > all_hs[all_index,0]).item() and 
                all_index < (all_hs.shape[0] - 1)):
                all_index += 1
            else:
                stop = True
    return match_index


def compute_updates(all_counts_shape, new_hs, new_counts, match_indices):
    count_update = torch.zeros(all_counts_shape)
    new_h_list = []
    new_count_list = []
    for index, new_h, count in zip(match_indices, new_hs, new_counts):
        index_ = int(index.item())
        count_ = int(count.item())
        if index_ != -1:
            count_update[index_] = count_
        else:
            new_h_list.append(new_h)
            new_count_list.append(count_)
    return count_update, torch.vstack(new_h_list), torch.tensor(new_count_list)

def merge_unique_counts(agg_hs, agg_counts, update_hs, update_counts):
    matched_indices = identify_matches(agg_hs, update_hs)
    count_update, new_hs, new_counts = compute_updates(
        agg_counts.shape, update_hs, update_counts, matched_indices
    )

    all_hs = torch.vstack((agg_hs, new_hs))
    all_counts = torch.hstack((agg_counts + count_update, new_counts))
    all_merged = torch.hstack((all_hs, all_counts.unsqueeze(1)))

    all_sorted = torch.unique(all_merged, dim=0)
    hs_sorted, counts_sorted = all_sorted[:,:-1], all_sorted[:,-1]
    return hs_sorted, counts_sorted

#%%
#testfile = files[0]
#count_non_unique = 0
all_hs, all_counts = None, None
tempcount = 0
tempdir = os.path.join(OUT, "p_h/tempdir")
tempbatches = 10
for i, filepath in enumerate(tqdm(files)):
    #filepath = files[0]
    with open(os.path.join(generations_folder, filepath), "rb") as f:
        data = pickle.load(f)

    hs = [x[0] for x in data]
    hs = torch.vstack(hs)
    unique_hs, counts = torch.unique(hs, return_counts=True, dim=0)
    if all_hs is None and all_counts is None:
        all_hs, all_counts = unique_hs, counts
    elif (i+1)%tempbatches == 0 or i == len(files)-1:
        all_hs, all_counts = merge_unique_counts(all_hs, all_counts, unique_hs, counts)
        tempfile = os.path.join(
            tempdir, 
            f"temp{tempcount}.pkl"
        )
        with open(tempfile, 'wb') as f:
            pickle.dump((all_hs, all_counts), f, protocol=pickle.HIGHEST_PROTOCOL)
        all_hs, all_counts = None, None
        tempcount+=1
    else:
        all_hs, all_counts = merge_unique_counts(all_hs, all_counts, unique_hs, counts)

#%%
tempfiles = os.listdir(tempdir)
all_hs, all_counts = None, None
for i, filepath in enumerate(tqdm(tempfiles)):
    #filepath = files[0]
    with open(os.path.join(tempdir, filepath), "rb") as f:
        file_hs, file_counts = pickle.load(f)

    if all_hs is None and all_counts is None:
        all_hs, all_counts = file_hs, file_counts
    else:
        all_hs, all_counts = merge_unique_counts(
            all_hs, all_counts, file_hs, file_counts
        )

#%%
counts, countcounts = torch.unique(all_counts, return_counts=True)
print("Count of counts:")
for count, countcount in zip(counts, countcounts):
    print(f"{int(count)}: {countcount}")

#%%% tests
"""
agg_hs = torch.tensor([[1,2,3],[4,3,2],[7,8,9]]) 
update_hs = torch.tensor([[-1, 1, 2],[4,3,2], [7,11,12], [11,12,13]])
agg_counts = torch.tensor([1,1,99])
update_counts = torch.tensor([5,2,1,3])

matched_indices = identify_matches(agg_hs, update_hs)
count_update, new_hs, new_counts = compute_updates(
    agg_counts.shape, update_hs, update_counts, matched_indices
)

all_hs = torch.vstack((agg_hs, new_hs))
all_counts = torch.hstack((agg_counts + count_update, new_counts))
all_merged = torch.hstack((all_hs, all_counts.unsqueeze(1)))

all_sorted = torch.unique(all_merged, dim=0)
hs_sorted, counts_sorted = all_sorted[:,:-1], all_sorted[:,-1]
#counts_sort_index = torch.unique(sort_index, dim=1).squeeze()
#sorted_counts = all_counts[sort_index]

#%%
"""
"""
merge_unique_counts(test1, counttest1, test2, counttest2)

#new_counts = counttest

#%%
test1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]) 
test2 = torch.tensor([[-1,2,3], [1,2,3], [4,6,7], [10,11,12]])

out = identify_matches(test2, test1)

#%%
all_index = 0
    match_status = torch.zeros(new_hs.shape[0])
    #all_counts_copy = all_counts.clone()
    for i, (new_h, new_count) in enumerate(zip(new_hs, counts)):
        stop = False
        while not matched:
            #if all_index < all_hs.shape[0]:
            close = torch.isclose(all_hs[all_index,:], new_h, 1E-5).all().item()
            #else:
            #    match_status[i] = False
            #    matched = True
            #    break
            if close:
                #all_counts[all_index] += new_count
                match_index[i] = all_index
                stop = True
            elif ((new_h[0] > all_hs[all_index,0]).item() and 
                all_index < (all_hs.shape[0] - 1)):
                all_index += 1
            else:
                #match_status[i] = False
                stop = True

#%%
match_status = torch.zeros(unique_hs.shape[0])
all_counts_copy = all_counts.clone()
for i, (new_h, new_count) in enumerate(zip(unique_hs, counts)):
    for j, (all_h, all_count) in enumerate(tqdm(zip(all_hs, all_counts))):
        close = torch.isclose(new_h, all_h, 1E-5).all().item()
        if close:
            all_counts[j] += new_count
            match_status[i] = True
        elif (new_h[0] > all_h[0]).item():
            continue
        else:
            match_status[i] = False
            break

#if unique_hs.shape != hs.shape:
#    count_non_unique += 1

#%%
hs = {}
for h, _ in data:
    #count = hs.get(h, 0)
    solved = False
    for k, c in hs.items():
        if torch.isclose(h, k, 1E-5).all().item():
            hs[k] = c + 1
            solved = True
            break
        else: 
            continue
    if not solved:
        hs[h] = 1

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