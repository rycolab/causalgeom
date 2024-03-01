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
#import torch
import random 
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import trange
from transformers import TopPLogitsWarper, LogitsProcessorList
import torch 

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS


from evals.mi_eval import prep_generated_data, compute_inner_loop_qxhs
from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.eval_utils import load_run_Ps, load_run_output, load_model_eval,\
    renormalize
from data.filter_generations import load_generated_hs_wff
from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%

#TOMORROW:
# - process all hidden states created overnight
# - re-run LEACE training
# - replace this with a CEBaB food run with gpt2-large (can be old)
# - import the new unfiltered token lists
# - use whatever processed file you want to do a first implementation of the algo
MODEL_NAME = "gpt2-large"
CONCEPT = "number"
NUCLEUS = True
SINGLE_TOKEN = False
MSAMPLES = 10
RUN_PATH = os.path.join(OUT, 
    "run_output/number/gpt2-large/leace29022024/run_leace_number_gpt2-large_2024-02-29-18:30:00_0_3.pkl"
)

#%%
run = load_run_output(RUN_PATH)
P, I_P, _ = load_run_Ps(RUN_PATH)

# test set version of the eval
V, l0_tl, l1_tl = load_model_eval(MODEL_NAME, CONCEPT, SINGLE_TOKEN)

#p_x = load_p_x(MODEL_NAME, NUCLEUS)
p_c, l0_hs_wff, l1_hs_wff, all_hs = prep_generated_data(MODEL_NAME, NUCLEUS)

#%%
X_dev, y_dev, facts_dev, foils_dev, cxt_toks_dev = \
    run["X_val"], run["y_val"], run["facts_val"], run["foils_val"], run["cxt_toks_val"]
X_test, y_test, facts_test, foils_test, cxt_toks_test = \
    run["X_test"], run["y_test"], run["facts_test"], run["foils_test"], run["cxt_toks_test"]

#%%


model = get_model(MODEL_NAME)
tokenizer = get_tokenizer(MODEL_NAME)
device = get_device()


if MODEL_NAME == "gpt2-large":
    model = model.to(device)

# %%
test_obs_index = 0
h = X_test[test_obs_index]
cxt_pad = cxt_toks_test[test_obs_index]
cxt = cxt_pad[cxt_pad != -1]

#%% new h
with torch.no_grad():
    cxt_input = torch.tensor(cxt).to(device)
    output = model(
        input_ids=cxt_input, 
        #attention_mask=attention_mask, 
        labels=cxt_input,
        output_hidden_states=True
    )
    cxt_pkv = output["past_key_values"]

#%%

first_qxh = compute_inner_loop_qxhs(
    "hbot", h, all_hs, P, I_P, V, MSAMPLES, processor=None
).mean(axis=0)


#%%
from itertools import zip_longest

test_tl = l0_tl
cxt_plus_tl = [np.append(cxt, x).tolist() for x in test_tl]
cxt_plus_tl_batched = torch.tensor(list(zip_longest(*cxt_plus_tl, fillvalue=tokenizer.pad_token_id))).T

#%%
with torch.no_grad():
    cxt_plus_tl_input = cxt_plus_tl_batched.to(device)
    output = model(
        input_ids=cxt_plus_tl_input, 
        #attention_mask=attention_mask, 
        labels=cxt_plus_tl_input,
        output_hidden_states=True,
        #past_key_values= cxt_pkv
    )

#%%
p_words = []
word_index = 9
#for ind, word_tok in enumerate(test_tl):
cxt_last_index = cxt.shape[0] - 1
cxt_plus_word_tok = cxt_plus_tl_batched[word_index] #
word_tok = test_tl[word_index]
hidden_states = output["hidden_states"][-1][word_index].cpu().numpy()

#%%
i = 0 
p_word = 1
while i < len(word_tok):
    logging.info(f"Loop iteration: {i}")
    h = hidden_states[cxt_last_index + i]
    logging.info(f"Hs index: {cxt_last_index + i} / {hidden_states.shape[0]}")
    qxh = compute_inner_loop_qxhs(
        "hbot", h, all_hs, P, I_P, V, MSAMPLES, processor=None
    ).mean(axis=0)
    token_prob = qxh[word_tok[i]]
    logging.info(f"Token prob: {token_prob}")
    p_word = p_word * token_prob
    logging.info(f"Word prob: {p_word}")
    i += 1


#%%
#p_w = first_qxh[word_tok[0]]
#i = 1
#cxt_loop = torch.tensor(np.append(cxt.copy(), word_tok[0]))#.to(device)
#while i < len(word_tok):
#    with torch.no_grad():
#        output = model(
#            input_ids=cxt_loop, 
##            #attention_mask=attention_mask, 
#            labels=cxt_loop,
#            output_hidden_states=True
#        )


# %%
new_word_tokens = []
other_tokens = []
for token, token_id in tokenizer.vocab.items():
    if token.startswith("Ġ"):
        new_word_tokens.append((token, token_id))
    else:
        other_tokens.append((token, token_id))

# %%
