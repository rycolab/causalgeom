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

#sys.path.append('../../')
sys.path.append('./src/')

from paths import DATASETS, OUT
from utils.lm_loaders import get_tokenizer, get_V

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%#################
# Arguments        #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Embedding word and target lists')
    argparser.add_argument(
        "-dataset", 
        type=str,
        choices=["linzen", "ud_fr_gsd"],
        default="linzen",
        help="Dataset to extract counts from"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=["bert-base-uncased", "gpt2", "gpt2-medium", "gpt2-large"],
        help="MultiBERTs checkpoint for tokenizer and model"
    )
    return argparser.parse_args()

def get_wordlist_paths(dataset_name, model_name):
    wordlist_path = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/wordlist.csv")
    verblist_path = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/verb_list_final.pkl")
    return wordlist_path, verblist_path

def get_outfile_paths(dataset_name, model_name):
    word_emb_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_word_embeds.npy")
    verb_p_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_verb_p.npy")
    sg_emb_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_sg_embeds.npy")
    pl_emb_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_pl_embeds.npy")
    return word_emb_outfile, verb_p_outfile, sg_emb_outfile, pl_emb_outfile

def define_add_space(model_name):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large"]:
        return True
    else: 
        return False

#%%#################
# Helpers         #
####################
def tokenize_word(tokenizer, word, masked, add_space):
    if type(word) != str:
        return []
    if add_space:
        word = " "+word
    if masked:
        return tokenizer(word)["input_ids"][1:-1]
    else:
        return tokenizer(word)["input_ids"]
    
def tokenize_word_handler(model_name, tokenizer, word, add_space=False):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        return tokenize_word(tokenizer, word, False, add_space)
    elif model_name == "bert-base-uncased":
        return tokenize_word(tokenizer, word, True, False)
    else:
        raise ValueError("Incorrect model name")

def get_unique_word_list(wordlist_path, model_name, tokenizer, add_space):
    wl = pd.read_csv(wordlist_path, index_col=0)
    wl.drop_duplicates(inplace=True)
    wl["input_ids_word"] = wl["word"].apply(
        lambda x: tokenize_word_handler(model_name, tokenizer, x, add_space)
    )
    wl["ntokens"] = wl["input_ids_word"].apply(lambda x: len(x))
    wl_1tok = wl[wl["ntokens"]==1]
    wl_1tok["first_id_word"] = wl_1tok["input_ids_word"].apply(lambda x: int(x[0]))
    word_tok = wl_1tok["first_id_word"].to_numpy()
    word_tok_unq, word_tok_counts = np.unique(word_tok, return_counts=True)
    return word_tok_unq

def get_unique_verb_lists(verblist_path, model_name, tokenizer, add_space):
    with open(verblist_path, 'rb') as f:      
        vl = pickle.load(f)

    for col in ["sverb", "pverb"]:
        vl[f"{col}_input_ids"] = vl[col].apply(
            lambda x: tokenize_word_handler(model_name, tokenizer, x, add_space))
        vl[f"{col}_ntokens"] = vl[f"{col}_input_ids"].apply(
            lambda x: len(x)
        )

    vl["1tok"] = (vl["sverb_ntokens"] == 1) & (vl["pverb_ntokens"]==1)

    vl.drop(vl[vl["1tok"] != True].index, inplace=True)

    vl["sverb_tok"] = vl["sverb_input_ids"].apply(lambda x: x[0])
    vl["pverb_tok"] = vl["pverb_input_ids"].apply(lambda x: x[0])

    verb_p = vl[["p_sg", "p_pl"]].to_numpy()
    vl_sg_tok = vl["sverb_tok"].to_numpy()
    vl_pl_tok = vl["pverb_tok"].to_numpy()
    return verb_p, vl_sg_tok, vl_pl_tok

def embed_and_export_list(token_list, V, outfile):
    emb = V[token_list]
    np.save(outfile, emb)


#%%#################
# Main             #
####################
if __name__=="__main__":
    args = get_args()
    logging.info(args)

    dataset_name = args.dataset
    model_name = args.model
    #dataset_name = "gpt2"
    #model_name = "linzen"
    
    logging.info(f"Tokenizing and saving embeddings from word and verb lists for model {model_name}")
    wordlist_path, verblist_path = get_wordlist_paths(dataset_name, model_name)
    word_emb_outfile, verb_p_outfile, sg_emb_outfile, pl_emb_outfile = get_outfile_paths(dataset_name, model_name)

    tokenizer = get_tokenizer(model_name)
    V = get_V(model_name)
    add_space = define_add_space(model_name)

    wl = get_unique_word_list(wordlist_path, model_name, tokenizer, add_space)
    logging.info(f"Single token word list of length: {len(wl)}")
    embed_and_export_list(wl, V, word_emb_outfile)
    logging.info(f"Tokenized and exported word embeds to: {word_emb_outfile}")

    verb_p, sg_tok, pl_tok = get_unique_verb_lists(
        verblist_path, model_name, tokenizer, add_space)
    np.save(verb_p_outfile, verb_p)
    logging.info(f"Exported single token verb probs to: {verb_p_outfile}")

    logging.info(f"Single token sg verb list of length: {len(sg_tok)}")
    logging.info(f"Single token pl verb list of length: {len(pl_tok)}")
    embed_and_export_list(sg_tok, V, sg_emb_outfile)
    logging.info(f"Tokenized and exported sg verb embeds to: {sg_emb_outfile}")
    embed_and_export_list(pl_tok, V, pl_emb_outfile)
    logging.info(f"Tokenized and exported pl verb embeds to: {pl_emb_outfile}")

    
