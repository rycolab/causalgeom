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
from utils.lm_loaders import get_tokenizer, get_V, GPT2_LIST

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

LINZEN_WORD_LIST_PATH = os.path.join(DATASETS, f"processed/linzen/word_lists/wordlist.csv")
LINZEN_VERB_LIST_PATH = os.path.join(DATASETS, f"processed/linzen/word_lists/verb_list_final.pkl")

FR_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/fr/word_lists/adj_list.tsv")
FR_WORD_LIST_PATH = os.path.join(DATASETS, f"processed/fr/word_lists/other_list.tsv")

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
        choices=["bert-base-uncased"] + GPT2_LIST,
        help="MultiBERTs checkpoint for tokenizer and model"
    )
    return argparser.parse_args()

def get_wordlist_paths(dataset_name):
    if dataset_name == "linzen":
        return LINZEN_WORD_LIST_PATH, LINZEN_VERB_LIST_PATH
    elif dataset_name in ["ud_fr_gsd"]:
        return FR_WORD_LIST_PATH, FR_ADJ_LIST_PATH
    else:
        raise ValueError("Invalid dataset name")

def get_outfile_paths(dataset_name, model_name):
    if dataset_name == "linzen":
        word_emb_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_word_embeds.npy")
        verb_p_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_verb_p.npy")
        sg_emb_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_sg_embeds.npy")
        pl_emb_outfile = os.path.join(DATASETS, f"processed/{dataset_name}/word_lists/{model_name}_pl_embeds.npy")
        return word_emb_outfile, verb_p_outfile, sg_emb_outfile, pl_emb_outfile
    elif dataset_name in ["ud_fr_gsd"]:
        word_emb_outfile = os.path.join(DATASETS, f"processed/fr/word_lists/{model_name}_word_embeds.npy")
        adj_p_outfile = os.path.join(DATASETS, f"processed/fr/word_lists/{model_name}_adj_p.npy")
        masc_emb_outfile = os.path.join(DATASETS, f"processed/fr/word_lists/{model_name}_masc_embeds.npy")
        fem_emb_outfile = os.path.join(DATASETS, f"processed/fr/word_lists/{model_name}_fem_embeds.npy")
        return word_emb_outfile, adj_p_outfile, masc_emb_outfile, fem_emb_outfile
    else:
        raise ValueError("Invalid dataset name")

def define_add_space(model_name):
    if model_name in GPT2_LIST:
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
    if model_name in GPT2_LIST:
        return tokenize_word(tokenizer, word, False, add_space)
    elif model_name == "bert-base-uncased":
        return tokenize_word(tokenizer, word, True, False)
    else:
        raise ValueError("Incorrect model name")

def get_unique_word_list(wordlist_path, model_name, tokenizer, add_space):
    wl = pd.read_csv(wordlist_path, index_col=0, sep="\t")
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

def get_unique_lemma_lists(lemma_list_path, model_name, tokenizer, add_space):
    #with open(lemma_list_path, 'rb') as f:      
    #    ll = pickle.load(f)
    ll = pd.read_csv(lemma_list_path, index_col=0, sep="\t")


    for col in ["lemma_0", "lemma_1"]:
        ll[f"{col}_input_ids"] = ll[col].apply(
            lambda x: tokenize_word_handler(model_name, tokenizer, x, add_space))
        ll[f"{col}_ntokens"] = ll[f"{col}_input_ids"].apply(
            lambda x: len(x)
        )

    ll["1tok"] = (ll["lemma_0_ntokens"] == 1) & (ll["lemma_1_ntokens"]==1)

    ll.drop(ll[ll["1tok"] != True].index, inplace=True)

    ll["lemma_0_tok"] = ll["lemma_0_input_ids"].apply(lambda x: x[0])
    ll["lemma_1_tok"] = ll["lemma_1_input_ids"].apply(lambda x: x[0])

    lemma_p = ll[["p_0", "p_1"]].to_numpy()
    ll_sg_tok = ll["lemma_0_tok"].to_numpy()
    ll_pl_tok = ll["lemma_1_tok"].to_numpy()
    return lemma_p, ll_sg_tok, ll_pl_tok

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
    
    logging.info(f"Tokenizing and saving embeddings from word and lemma lists for model {model_name}")
    wordlist_path, lemmalist_path = get_wordlist_paths(dataset_name)
    word_emb_outfile, lemma_p_outfile, l0_emb_outfile, l1_emb_outfile = get_outfile_paths(dataset_name, model_name)

    tokenizer = get_tokenizer(model_name)
    V = get_V(model_name)
    add_space = define_add_space(model_name)

    wl = get_unique_word_list(wordlist_path, model_name, tokenizer, add_space)
    logging.info(f"Single token word list of length: {len(wl)}")
    embed_and_export_list(wl, V, word_emb_outfile)
    logging.info(f"Tokenized and exported word embeds to: {word_emb_outfile}")

    lemma_p, l0_tok, l1_tok = get_unique_lemma_lists(
        lemmalist_path, model_name, tokenizer, add_space)
    np.save(lemma_p_outfile, lemma_p)
    logging.info(f"Exported single token verb probs to: {lemma_p_outfile}")

    logging.info(f"Single token lemma 0 list of length: {len(l0_tok)}")
    logging.info(f"Single token lemma 1 list of length: {len(l1_tok)}")
    embed_and_export_list(l0_tok, V, l0_emb_outfile)
    logging.info(f"Tokenized and exported lemma 0 embeds to: {l0_emb_outfile}")
    embed_and_export_list(l1_tok, V, l1_emb_outfile)
    logging.info(f"Tokenized and exported lemma 1 embeds to: {l1_emb_outfile}")

    
