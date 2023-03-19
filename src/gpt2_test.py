#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import pickle
from tqdm import tqdm
import shutil

from utils.lm_loaders import get_model, get_tokenizer, get_V
from evals.kl_eval import load_model_eval, get_distribs, compute_overall_mi

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

# %%
model_name="gpt2"
batchfile="/cluster/work/cotterell/cguerner/usagebasedprobing/out/hidden_states/linzen/gpt2/batch_2.pkl"
with open(batchfile, 'rb') as f:      
    batch_data = pickle.load(f)

word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob = load_model_eval(model_name, add_space=True)

# %%
mi=[]
for j in range(len(batch_data)):
    sample=batch_data[j]
    #for i in range(sample["verb_hs"].shape[0]):
    h=sample["verb_hs"][0]
    base_distribs = get_distribs(h, word_emb, sg_emb, pl_emb)
    mi.append(compute_overall_mi(sg_pl_prob, base_distribs["sg"], base_distribs["pl"]))

print(np.mean(mi))
print(np.std(mi))


#%%
tok = get_tokenizer("gpt2")

#%%
tok()

#%%
from paths import DATASETS
import pandas as pd
# Load dataset
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

#%%
h = X[0]

#%%
from evals.kl_eval import load_model_eval, get_distribs, compute_overall_mi

word_emb, sg_emb, pl_emb, verb_probs, sg_pl_prob = load_model_eval("gpt2")
base_distribs = get_distribs(h, word_emb, sg_emb, pl_emb)

compute_overall_mi(sg_pl_prob, base_distribs["sg"], base_distribs["pl"])

#%%
dim=h.shape[0]
I = np.eye(dim, dim)

from evals.kl_eval import compute_kls_one_sample

kls = compute_kls_one_sample(h, I, I, word_emb, sg_emb, pl_emb, verb_probs, 
    sg_pl_prob, faith=False, er_kls=False)

# %%
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch 

clf = torch.nn.Linear(768, 768)
optimizer = SGD(clf.parameters(), lr=0.01)
scheduler = MultiStepLR(optimizer, [5,10], verbose=True)
# %%
for i in range(20):
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
# %%
