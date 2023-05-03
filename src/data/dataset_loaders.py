import os
import sys

import numpy as np 
import pandas as pd
import csv 
import pickle

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS

LINZEN_PREPROCESSED = os.path.join(DATASETS, "preprocessed/linzen_preprocessed.tsv")
UD_FRENCH_GSD = os.path.join(DATASETS, "preprocessed/ud_fr_gsd")

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
    elif model_tupe == "masked":
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
    elif model_tupe == "masked":
        return load_udfr_masked(split_path)
    else:
        raise ValueError("Invalid model type for French UD data")


#%%####################################
# Preprocessed Dataset General Loader #
#######################################
def get_model_type(model_name):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-base-french"]:
        return "ar"
    elif model_name == "bert-base-uncased":
        return "masked"
    else: 
        raise ValueError(f"Model {model_name} not supported")

def load_dataset(dataset_name, model_name, split=None):
    """ Dataset name: ["linzen", "ud_fr_gsd"]
    Model name: ["gpt2", "bert-base-uncased"]
    Split: ["train", "dev", "test"] (only for UD)
    """
    model_type = get_model_type(model_name)
    if dataset_name == "linzen":
        return load_linzen(model_type)
    #TODO: complete this second if 
    elif dataset_name == "ud_fr_gsd":
        assert split in ["train", "dev", "test"], "Must specify split"
        return load_udfr(
            model_type, os.path.join(UD_FRENCH_GSD, f"{split}.pkl")
        )
    else:
        raise ValueError("invalid dataset name")


#%%##############################
# Loading Processed HS Datasets #
#################################
def load_processed_data(dataset_name, model_name):
    if model_name.startswith("gpt2"):
        DATASET = os.path.join(DATASETS, f"processed/{dataset_name}/ar/{dataset_name}_{model_name}_ar.pkl")
    elif model_name == "bert-base-uncased":
        DATASET = os.path.join(DATASETS, f"processed/{dataset_name}/masked/{dataset_name}_{model_name}_masked.pkl")
    else:
        DATASET = None

    with open(DATASET, 'rb') as f:      
        data = pd.DataFrame(pickle.load(f), columns = ["h", "u", "y"])
    
    X = np.array([x for x in data["h"]])
    U = np.array([x for x in data["u"]])
    y = np.array([yi for yi in data["y"]])
    return X, U, y

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
    SG_PL_PROB = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/sg_pl_prob.pkl")
    WORD_EMB = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_word_embeds.npy")
    VERB_P = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_verb_p.npy")
    SG_EMB = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_sg_embeds.npy")
    PL_EMB = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_pl_embeds.npy")

    word_emb = np.load(WORD_EMB)
    verb_p = np.load(VERB_P)
    sg_emb = np.load(SG_EMB)
    pl_emb = np.load(PL_EMB)
    with open(SG_PL_PROB, 'rb') as f:      
        sg_pl_prob = pickle.load(f).to_numpy()
    return word_emb, sg_emb, pl_emb, verb_p, sg_pl_prob