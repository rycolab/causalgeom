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

data = load_processed_data("gender", "gpt2-base-french")
# %%
y_train = data[2]
fact_train = data[3]
foil_train = data[4]



