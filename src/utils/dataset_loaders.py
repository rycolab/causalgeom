import os
import sys

import numpy as np 
import pandas as pd
import csv 
import pickle

#sys.path.append('..')
sys.path.append('./src/')

from data.embed_wordlists.embedder import get_outfile_paths
from utils.lm_loaders import GPT2_LIST, BERT_LIST
from paths import DATASETS, FR_DATASETS

LINZEN_PREPROCESSED = os.path.join(DATASETS, "preprocessed/linzen_preprocessed.tsv")
UD_FRENCH_GSD_PREPROCESSED = os.path.join(DATASETS, "preprocessed/ud_fr_gsd")
UD_FRENCH_ParTUT_PREPROCESSED = os.path.join(DATASETS, "preprocessed/ud_fr_partut")
UD_FRENCH_Rhapsodie_PREPROCESSED = os.path.join(DATASETS, "preprocessed/ud_fr_rhapsodie")

#%%##############################
# Loading Preprocessed Datasets #
#################################
#%% LINZEN LOADERS
def load_linzen_ar():
    data = []
    with open(LINZEN_PREPROCESSED) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            unmasked_text = line[1]
            verb = line[3]
            iverb = line[4]
            verb_pos = line[5]
            vindex = int(line[6])
            if vindex > 0:
                verb = " " + verb
                iverb = " " + iverb
            pre_verb_text = " ".join(unmasked_text.split(" ")[:vindex])
            verb_text = " ".join(unmasked_text.split(" ")[:(vindex+1)])
            iverb_text = " ".join(unmasked_text.split(" ")[:vindex] + [iverb])
            sample = dict(
                pre_tgt_text=pre_verb_text,
                fact_text=verb_text,
                foil_text=iverb_text,
                fact=verb,
                foil=iverb,
                tgt_label=verb_pos
            )
            data.append(sample)
    return data

def load_linzen_masked():
    data = []
    with open(LINZEN_PREPROCESSED) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            #masked = line[2].replace("***mask***", "[MASK]")
            masked = line[2]
            #mask_index = masked.split().index("[MASK]")
            sample = dict(
                masked=masked,
                #mask_index=mask_index,
                fact=line[3],
                foil=line[4],
                tgt_label=line[5]
            )
            data.append(sample)
    return data

def load_linzen(model_type):
    if model_type == "ar":
        return load_linzen_ar()
    elif model_type == "masked":
        return load_linzen_masked()
    else:
        raise ValueError("Invalid model type for Linzen")


#%% UD 
def load_udfr_ar(split_path):
    data = pd.read_pickle(split_path)
    data.drop(data[data["ar_flag"] != True].index, inplace=False)
    data["fact"] = " " + data["adj"]
    data["foil"] = " " + data["adj_gender_foil"]
    data = data[["pre_tgt_text", "fact_text", "foil_text", "fact", "foil", 
        "gender"]]
    data.columns = ["pre_tgt_text", "fact_text", "foil_text", "fact", 
        "foil", "tgt_label"]
    return data.to_dict("records")

def load_udfr_masked(split_path):
    data = pd.read_pickle(split_path)
    data = data[["masked", "adj", "adj_gender_foil", "gender"]]
    data.columns = ["masked", "fact", "foil", "tgt_label"]
    return data.to_dict("records")

def load_udfr(model_type, split_path):
    if model_type == "ar":
        return load_udfr_ar(split_path)
    elif model_type == "masked":
        return load_udfr_masked(split_path)
    else:
        raise ValueError("Invalid model type for French UD data")

def get_udfr_dataset_folder(dataset_name):
    if dataset_name == "ud_fr_gsd": 
        return UD_FRENCH_GSD_PREPROCESSED
    elif dataset_name == "ud_fr_partut":
        return UD_FRENCH_ParTUT_PREPROCESSED
    elif dataset_name == "ud_fr_rhapsodie":
        return UD_FRENCH_Rhapsodie_PREPROCESSED
    else:
        raise ValueError(f"Dataset name not supported: {dataset_name}")

#%%####################################
# Preprocessed Dataset General Loader #
#######################################
def get_model_type(model_name):
    if model_name in GPT2_LIST:
        return "ar"
    elif model_name in BERT_LIST:
        return "masked"
    else: 
        raise ValueError(f"Model {model_name} not supported")

def load_dataset(dataset_name, model_name, split=None):
    """ Dataset name: ["linzen", "ud_fr_gsd"]
    Model name: GPT2 + BERT models
    Split: ["train", "dev", "test"] (only for UD)
    """
    model_type = get_model_type(model_name)
    if dataset_name == "linzen":
        return load_linzen(model_type)
    #TODO: complete this second if 
    elif dataset_name in FR_DATASETS:
        assert split in ["train", "dev", "test"], "Must specify split"
        dataset_folder = get_udfr_dataset_folder(dataset_name)
        return load_udfr(
            model_type, os.path.join(dataset_folder, f"{split}.pkl")
        )
    else:
        raise ValueError("invalid dataset name")


#%%##############################
# Loading Processed HS Datasets #
#################################
def load_dataset_pickle(path):
    with open(path, 'rb') as f:      
        data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])
    
    X = np.array([x for x in data["h"]])
    U = np.array([x for x in data["u"]])
    y = np.array([yi for yi in data["y"]])
    return X, U, y

def get_processed_data_path(dataset_name, model_name, split=None):
    if model_name in GPT2_LIST and split is None:
        return os.path.join(DATASETS, f"processed/{dataset_name}/ar/{dataset_name}_{model_name}_ar.pkl")
    elif model_name in GPT2_LIST and split is not None:
        return os.path.join(DATASETS, f"processed/{dataset_name}/ar/{dataset_name}_{model_name}_ar_{split}.pkl")
    elif model_name in BERT_LIST and split is None:
        return os.path.join(DATASETS, f"processed/{dataset_name}/masked/{dataset_name}_{model_name}_masked.pkl")
    elif model_name in BERT_LIST and split is not None:
        return os.path.join(DATASETS, f"processed/{dataset_name}/masked/{dataset_name}_{model_name}_masked_{split}.pkl")
    else:
        return None

def load_processed_data(dataset_name, model_name, split=None):
    dataset_path = get_processed_data_path(dataset_name, model_name, split)
    return load_dataset_pickle(dataset_path)

def sample_hs(hs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    ind = idx[:nsamples]
    return hs[ind]

def load_hs(dataset_name, model_name, nsamples=None):
    hs, _, _ = load_processed_data(dataset_name, model_name)

    if nsamples is not None:
        return sample_hs(hs, nsamples)
    else:
        return hs

#%%##############################
# Loading Model Word Lists.     #
#################################
def load_model_eval(dataset_name, model_name):
    word_emb_path, lemma_p_path, lemma_0_path, lemma_1_path = get_outfile_paths(dataset_name, model_name)
    
    if dataset_name == "linzen":
        concept_prob_path = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/sg_pl_prob.pkl")
    elif dataset_name in ["ud_fr_gsd"]:
        concept_prob_path = os.path.join(DATASETS, "processed/fr/word_lists/p_fr_gender.pkl")
    
    with open(concept_prob_path, 'rb') as f:
        concept_prob = pickle.load(f).to_numpy().tolist()
    
    word_emb = np.load(word_emb_path)
    lemma_prob = np.load(lemma_p_path)
    l0_emb = np.load(lemma_0_path)
    l1_emb = np.load(lemma_1_path)
    return word_emb, l0_emb, l1_emb, lemma_prob, concept_prob