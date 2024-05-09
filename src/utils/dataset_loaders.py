import os
import sys

import numpy as np 
import pandas as pd
import csv 
import pickle
from datasets import load_dataset
from itertools import zip_longest
#sys.path.append('..')
sys.path.append('./src/')

from data.embed_wordlists.embedder import get_emb_outfile_paths, get_token_list_outfile_paths
from utils.lm_loaders import GPT2_LIST, BERT_LIST, SUPPORTED_AR_MODELS
from paths import DATASETS, FR_DATASETS, HF_CACHE

LINZEN_PREPROCESSED = os.path.join(DATASETS, "preprocessed/linzen_preprocessed.tsv")
UD_FRENCH_GSD_PREPROCESSED = os.path.join(DATASETS, "preprocessed/ud_fr_gsd")
UD_FRENCH_ParTUT_PREPROCESSED = os.path.join(DATASETS, "preprocessed/ud_fr_partut")
UD_FRENCH_Rhapsodie_PREPROCESSED = os.path.join(DATASETS, "preprocessed/ud_fr_rhapsodie")

#%%##############################
# Loading HuggingFace datasets  #
#################################
def load_wikipedia(language):
    return load_dataset(
            "wikipedia", f"20220301.{language}", cache_dir=HF_CACHE
        )["train"]

#%%##############################
# Loading Preprocessed Datasets #
#################################
#%% LINZEN LOADERS
def load_linzen_ar(model_name):
    data = []
    with open(LINZEN_PREPROCESSED) as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for line in tsv_file:
            unmasked_text = line[1]
            verb = line[3]
            iverb = line[4]
            verb_pos = line[5]
            vindex = int(line[6])
            if model_name in GPT2_LIST and vindex > 0:
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

def load_linzen(model_type, model_name):
    if model_type == "ar":
        return load_linzen_ar(model_name)
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

#%% CEBaB
def load_CEBaB(concept, split):
    fpath = os.path.join(
        DATASETS, 
        f"preprocessed/CEBaB/{concept}/CEBaB_{concept}_{split}.pkl"
    )
    with open(fpath, "rb") as f:
        df = pickle.load(f).to_dict("records")
    return df

#%%####################################
# Preprocessed Dataset General Loader #
#######################################
def get_model_type(model_name):
    if model_name in SUPPORTED_AR_MODELS:
        return "ar"
    elif model_name in BERT_LIST:
        return "masked"
    else: 
        raise ValueError(f"Model {model_name} not supported")

def load_preprocessed_dataset(dataset_name, model_name, concept=None, split=None):
    """ Dataset name: ["linzen", "ud_fr_gsd"]
    Model name: GPT2 + BERT models + llama2
    Split: ["train", "dev", "test"]
    """
    model_type = get_model_type(model_name)
    if dataset_name == "linzen":
        return load_linzen(model_type, model_name)
    #TODO: complete this second if 
    elif dataset_name in FR_DATASETS:
        assert split in ["train", "dev", "test"], "Must specify split"
        dataset_folder = get_udfr_dataset_folder(dataset_name)
        return load_udfr(
            model_type, os.path.join(dataset_folder, f"{split}.pkl")
        )
    elif dataset_name == "CEBaB":
        assert split in ["train", "dev", "test"], "Must specify split"
        assert concept in ["food", "ambiance", "service", "noise"], \
            "Must specify CEBaB concept"
        return load_CEBaB(concept, split)
    else:
        raise ValueError("invalid dataset name")


#%%##############################
# Loading Processed HS Datasets #
#################################
def load_dataset_pickle(path, dataset_name):
    with open(path, 'rb') as f:     
        data = pd.DataFrame(pickle.load(f))
        assert data.shape[1] == 6, "Out of date processed dataset"
        data.columns = ["h", "u", "y", "fact", "foil", "cxt_tok"]        
    
    X = np.array([x for x in data["h"]])
    U = np.array([x for x in data["u"]])
    y = np.array([yi for yi in data["y"]])
    #fact = np.array([fact for fact in data["fact"]]).flatten()
    fact = np.array(list(zip_longest(*data["fact"], fillvalue=-1))).T
    #foil = np.array([foil for foil in data["foil"]]).flatten()
    foil = np.array(list(zip_longest(*data["foil"], fillvalue=-1))).T
    cxt_tok = np.array(list(zip_longest(*data["cxt_tok"], fillvalue=-1))).T
    #X = np.vstack(data["h"])
    #U = np.vstack(data["u"])
    #y = np.vstack(data["y"]).flatten()
    #fact = np.vstack(data["fact"]).flatten()
    #foil = np.vstack(data["foil"]).flatten()
    #cxt_tok = np.vstack(data["cxt_tok"])
    #attention_mask = np.vstack(data["attention_mask"])
    #return X, U, y, fact, foil, cxt_tok, attention_mask
    return X, U, y, fact, foil, cxt_tok

def get_processed_dataset_path(dataset_name, model_name, concept=None, split=None):
    if model_name in SUPPORTED_AR_MODELS:
        dataset_dir = os.path.join(DATASETS, f"processed/{dataset_name}/ar")
        file_name = f"{dataset_name}_{model_name}_ar.pkl"
        if concept is not None:
            file_name = file_name[:-len(f".pkl")] + f"_{concept}.pkl"
        if split is not None:
            file_name = file_name[:-len(f".pkl")] + f"_{split}.pkl"
        return os.path.join(dataset_dir, file_name)
    elif model_name in BERT_LIST and split is None:
        dataset_dir = os.path.join(DATASETS, f"processed/{dataset_name}/masked")
        file_name = f"{dataset_name}_{model_name}.pkl"
        if concept is not None:
            file_name = file_name[:-len(f".pkl")] + f"_{concept}.pkl"
        if split is not None:
            file_name = file_name[:-len(f".pkl")] + f"_{split}.pkl"
        return os.path.join(dataset_dir, file_name)
    else:
        raise NotImplementedError(f"Model name {model_name} and dataset {dataset_name} not implemented")
    

def load_processed_dataset(dataset_name, model_name, concept=None, split=None):
    dataset_path = get_processed_dataset_path(dataset_name, model_name, concept=concept, split=split)
    return load_dataset_pickle(dataset_path, dataset_name)

def load_gender_split(model_name, split_name):

    X_gsd, U_gsd, y_gsd, fact_gsd, foil_gsd, cxt_tok_gsd = load_processed_dataset(
        "ud_fr_gsd", model_name, concept="gender", split=split_name
    )
    X_partut, U_partut, y_partut, fact_partut, foil_partut, cxt_tok_partut = load_processed_dataset(
        "ud_fr_partut", model_name, concept="gender", split=split_name
    )
    X_rhapsodie, U_rhapsodie, y_rhapsodie, fact_rhapsodie, foil_rhapsodie, cxt_tok_rhapsodie = load_processed_dataset(
        "ud_fr_rhapsodie", model_name, concept="gender", split=split_name
    )

    X = np.vstack([X_gsd, X_partut, X_rhapsodie])
    U = np.hstack([U_gsd, U_partut, U_rhapsodie])
    y = np.hstack([y_gsd, y_partut, y_rhapsodie])
    
    all_fact = [*fact_gsd] + [*fact_partut] + [*fact_rhapsodie]
    fact = np.array(list(zip_longest(*all_fact, fillvalue=-1))).T
    
    all_foil = [*foil_gsd] + [*foil_partut] + [*foil_rhapsodie]
    foil = np.array(list(zip_longest(*all_foil, fillvalue=-1))).T
    
    all_cxt_tok = [*cxt_tok_gsd] + [*cxt_tok_partut] + [*cxt_tok_rhapsodie]
    cxt_tok = np.array(list(zip_longest(*all_cxt_tok, fillvalue=-1))).T
    return X, U, y, fact, foil, cxt_tok

def load_gender_processed(model_name):
    X_train, U_train, y_train, fact_train, foil_train, cxt_tok_train = load_gender_split(model_name, "train")
    #X_dev, U_dev, y_dev = load_gender_split(model_name, "dev")
    X_dev, U_dev, y_dev, fact_dev, foil_dev, cxt_tok_dev = load_gender_split(model_name, "dev")
    #X_test, U_test, y_test = load_gender_split(model_name, "test")
    X_test, U_test, y_test, fact_test, foil_test, cxt_tok_test = load_gender_split(model_name, "test")
    
    #stacking into one
    X = np.vstack([X_train, X_dev, X_test])
    U = np.hstack([U_train, U_dev, U_test])
    y = np.hstack([y_train, y_dev, y_test])
    
    all_fact = [*fact_train] + [*fact_dev] + [*fact_test]
    fact = np.array(list(zip_longest(*all_fact, fillvalue=-1))).T
    
    all_foil = [*foil_train] + [*foil_dev] + [*foil_test]
    foil = np.array(list(zip_longest(*all_foil, fillvalue=-1))).T
    
    all_cxt_tok = [*cxt_tok_train] + [*cxt_tok_dev] + [*cxt_tok_test]
    cxt_tok = np.array(list(zip_longest(*all_cxt_tok, fillvalue=-1))).T
    return X, U, y, fact, foil, cxt_tok

def load_CEBaB_processed(concept_name, model_name):
    X_train, U_train, y_train, facts_train, foils_train, cxt_toks_train = load_processed_dataset(
        "CEBaB", model_name, concept_name, split="train"
    )
    X_dev, U_dev, y_dev, facts_dev, foils_dev, cxt_toks_dev = load_processed_dataset(
        "CEBaB", model_name, concept_name, split="dev"
    )
    X_test, U_test, y_test, facts_test, foils_test, cxt_toks_test = load_processed_dataset(
        "CEBaB", model_name, concept_name, split="test"
    )
    return {
        "X_train": X_train, 
        "U_train": U_train,
        "y_train": y_train,
        "facts_train": facts_train,
        "foils_train": foils_train,
        "cxt_toks_train": cxt_toks_train,
        "X_dev": X_dev, 
        "U_dev": U_dev,
        "y_dev": y_dev,
        "facts_dev": facts_dev,
        "foils_dev": foils_dev,
        "cxt_toks_dev": cxt_toks_dev,
        "X_test": X_test, 
        "U_test": U_test,
        "y_test": y_test,
        "facts_test": facts_test,
        "foils_test": foils_test,
        "cxt_toks_test": cxt_toks_test,
    }

def load_processed_data(concept_name, model_name):
    # TODO fix this trash this function returns different things depending on the args.
    if concept_name == "number":
        return load_processed_dataset("linzen", model_name, "number")
    elif concept_name == "gender":
        return load_gender_processed(model_name)
    elif concept_name in ["food", "ambiance", "service", "noise"]:
        return load_CEBaB_processed(concept_name, model_name)
    else: 
        raise ValueError("Concept name and model name pair not supported.")

#%% loading only the hs
def sample_hs(hs, nsamples=200):
    idx = np.arange(0, hs.shape[0])
    np.random.shuffle(idx)
    ind = idx[:nsamples]
    return hs[ind]

def load_hs(concept_name, model_name, nsamples=None):
    hs, _, _ = load_processed_data(concept_name, model_name)

    if nsamples is not None:
        return sample_hs(hs, nsamples)
    else:
        return hs

def load_other_hs(concept_name, model_name, nsamples=None):
    if concept_name == "gender":
        language = "fr"
    elif concept_name == "number":
        language = "en"
    else:
        raise ValueError(f"Unsupported concept_name {concept_name}")

    other_hs_file = os.path.join(
        DATASETS, 
        f"processed/{language}/other_hidden_states/{model_name}.pkl"
    )
    with open(other_hs_file, 'rb') as f:
        hs = pickle.load(f)

    if nsamples is not None:
        hs = sample_hs(hs, nsamples)
    
    if model_name in BERT_LIST:
        hs = np.hstack((hs, np.ones((hs.shape[0], 1))))
    
    return hs

