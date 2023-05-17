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

X_train_gsd, U_train_gsd, y_train_gsd = load_processed_data("ud_fr_gsd", "camembert-base", "train")
X_train_partut, U_train_partut, y_train_partut = load_processed_data("ud_fr_partut", "camembert-base", "train")
X_train_rhapsodie, U_train_rhapsodie, y_train_rhapsodie = load_processed_data("ud_fr_rhapsodie", "camembert-base", "train")

X_dev_gsd, U_dev_gsd, y_dev_gsd = load_processed_data("ud_fr_gsd", "camembert-base", "dev")
X_dev_partut, U_dev_partut, y_dev_partut = load_processed_data("ud_fr_partut", "camembert-base", "dev")
X_dev_rhapsodie, U_dev_rhapsodie, y_dev_rhapsodie = load_processed_data("ud_fr_rhapsodie", "camembert-base", "dev")

X_test_gsd, U_test_gsd, y_test_gsd = load_processed_data("ud_fr_gsd", "camembert-base", "test")
X_test_partut, U_test_partut, y_test_partut = load_processed_data("ud_fr_partut", "camembert-base", "test")
X_test_rhapsodie, U_test_rhapsodie, y_test_rhapsodie = load_processed_data("ud_fr_rhapsodie", "camembert-base", "test")


def load_indiv_processed_data_gender(model_name, dataset_name):
    udfrgsd_train_path = get_processed_data_path(dataset_name, model_name, "train")
    udfrgsd_dev_path = get_processed_data_path(dataset_name, model_name, "dev")
    udfrgsd_test_path = get_processed_data_path(dataset_name, model_name, "test")


    return 

def load_processed_data_gender(model_name):
    

    udfrpartut_train_path = get_processed_data_path("ud_fr_partut", model_name, "train")
    udfrpartut_dev_path = get_processed_data_path("ud_fr_partut", model_name, "dev")
    udfrpartut_test_path = get_processed_data_path("ud_fr_partut", model_name, "test")

    udfrrhapsodie_train_path = get_processed_data_path("ud_fr_rhapsodie", model_name, "train")
    udfrrhapsodie_dev_path = get_processed_data_path("ud_fr_rhapsodie", model_name, "dev")
    udfrrhapsodie_test_path = get_processed_data_path("ud_fr_rhapsodie", model_name, "test")
    

def load_processed_data_new(concept_name, model_name):
    if concept_name == "number" and model_name in GPT2_LIST:
        DATASET = os.path.join(DATASETS, f"processed/{dataset_name}/ar/{dataset_name}_{model_name}_ar.pkl")
    elif concept_name == "number" and model_name in BERT_LIST:
        DATASET = os.path.join(DATASETS, f"processed/{dataset_name}/masked/{dataset_name}_{model_name}_masked.pkl")
    elif concept_name == "gender" and model_name in GPT2_LIST:
        load_gender_processed(model_name)

