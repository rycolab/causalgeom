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

from paths import DATASETS, OUT, FR_DATASETS
from utils.lm_loaders import get_tokenizer, get_V, GPT2_LIST, BERT_LIST

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

EN_VERB_LIST_PATH = os.path.join(DATASETS, f"processed/en/word_lists/verb_pair_list.tsv")
EN_WORD_LIST_PATH = os.path.join(DATASETS, f"processed/en/word_lists/other_list.tsv")

FR_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/fr/word_lists/adj_pair_list.tsv")
FR_WORD_LIST_PATH = os.path.join(DATASETS, f"processed/fr/word_lists/other_list.tsv")

#%%#################
# Arguments        #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Embedding word and target lists')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["gender", "number"],
        help="Concept to create embedded word lists for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=BERT_LIST + GPT2_LIST,
        help="Models to create embedding files for"
    )
    return argparser.parse_args()

def get_wordlist_paths(concept):
    if concept == "number":
        return EN_WORD_LIST_PATH, EN_VERB_LIST_PATH
    elif concept == "gender":
        return FR_WORD_LIST_PATH, FR_ADJ_LIST_PATH
    else:
        raise ValueError("Invalid dataset name")

def get_emb_outfile_paths(concept, model_name):
    if concept == "number":
        word_emb_outfile = os.path.join(DATASETS, f"processed/en/embedded_word_lists/{model_name}_word_embeds.npy")
        verb_p_outfile = os.path.join(DATASETS, f"processed/en/embedded_word_lists/{model_name}_verb_p.npy")
        sg_emb_outfile = os.path.join(DATASETS, f"processed/en/embedded_word_lists/{model_name}_sg_embeds.npy")
        pl_emb_outfile = os.path.join(DATASETS, f"processed/en/embedded_word_lists/{model_name}_pl_embeds.npy")
        return word_emb_outfile, verb_p_outfile, sg_emb_outfile, pl_emb_outfile
    elif concept == "gender":
        word_emb_outfile = os.path.join(DATASETS, f"processed/fr/embedded_word_lists/{model_name}_word_embeds.npy")
        adj_p_outfile = os.path.join(DATASETS, f"processed/fr/embedded_word_lists/{model_name}_adj_p.npy")
        masc_emb_outfile = os.path.join(DATASETS, f"processed/fr/embedded_word_lists/{model_name}_masc_embeds.npy")
        fem_emb_outfile = os.path.join(DATASETS, f"processed/fr/embedded_word_lists/{model_name}_fem_embeds.npy")
        return word_emb_outfile, adj_p_outfile, masc_emb_outfile, fem_emb_outfile
    else:
        raise ValueError("Invalid dataset name")

def get_token_list_outfile_paths(concept, model_name):
    if concept == "number":
        other_outfile = os.path.join(DATASETS, f"processed/en/tokenized_lists/{model_name}_word_token_list.npy")
        sg_outfile = os.path.join(DATASETS, f"processed/en/tokenized_lists/{model_name}_sg_token_list.npy")
        pl_outfile = os.path.join(DATASETS, f"processed/en/tokenized_lists/{model_name}_pl_token_list.npy")
        return other_outfile, sg_outfile, pl_outfile
    elif concept == "gender":
        other_outfile = os.path.join(DATASETS, f"processed/fr/tokenized_lists/{model_name}_word_token_list.npy")
        masc_outfile = os.path.join(DATASETS, f"processed/fr/tokenized_lists/{model_name}_masc_token_list.npy")
        fem_outfile = os.path.join(DATASETS, f"processed/fr/tokenized_lists/{model_name}_fem_token_list.npy")
        return other_outfile, masc_outfile, fem_outfile
    else:
        raise ValueError("Invalid dataset name")

def load_concept_token_lists(concept, model_name):
    _, l0_tl_file, l1_tl_file = get_token_list_outfile_paths(
        concept, model_name)
    #other_tl = np.load(other_tl_file)
    l0_tl = np.load(l0_tl_file)
    l1_tl = np.load(l1_tl_file)
    return l0_tl, l1_tl

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
    elif model_name in BERT_LIST:
        return tokenize_word(tokenizer, word, True, False)
    else:
        raise ValueError("Incorrect model name")

def get_unique_word_list(wordlist_path, model_name, tokenizer, add_space):
    wl = pd.read_csv(wordlist_path, index_col=0, sep="\t")[["word"]]
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

    lemma_p = ll[["pair_p_0", "pair_p_1"]].to_numpy()
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
    concept = args.concept
    model_name = args.model
    #concept = "number"
    #model_name = "gpt2-large"
    
    logging.info(f"Tokenizing and saving embeddings from word and lemma lists for model {model_name}")
    wordlist_path, lemmalist_path = get_wordlist_paths(concept)
    word_emb_outfile, lemma_p_outfile, l0_emb_outfile, l1_emb_outfile = get_emb_outfile_paths(concept, model_name)
    other_tl_outfile, l0_tl_outfile, l1_tl_outfile = get_token_list_outfile_paths(concept, model_name)

    for filepath in [word_emb_outfile, other_tl_outfile]:
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            logging.info(f"Created directory: {dirpath}")

    tokenizer = get_tokenizer(model_name)
    V = get_V(model_name)
    add_space = define_add_space(model_name)

    wl = get_unique_word_list(wordlist_path, model_name, tokenizer, add_space)
    logging.info(f"Single token word list of length: {len(wl)}")
    np.save(other_tl_outfile, wl)
    embed_and_export_list(wl, V, word_emb_outfile)
    logging.info(f"Tokenized and exported word embeds to: {word_emb_outfile}")

    lemma_p, l0_tok, l1_tok = get_unique_lemma_lists(
        lemmalist_path, model_name, tokenizer, add_space)
    np.save(l0_tl_outfile, l0_tok)
    np.save(l1_tl_outfile, l1_tok)
    np.save(lemma_p_outfile, lemma_p)
    logging.info(f"Exported single token verb probs to: {lemma_p_outfile}")

    logging.info(f"Single token lemma 0 list of length: {len(l0_tok)}")
    logging.info(f"Single token lemma 1 list of length: {len(l1_tok)}")
    embed_and_export_list(l0_tok, V, l0_emb_outfile)
    logging.info(f"Tokenized and exported lemma 0 embeds to: {l0_emb_outfile}")
    embed_and_export_list(l1_tok, V, l1_emb_outfile)
    logging.info(f"Tokenized and exported lemma 1 embeds to: {l1_emb_outfile}")

    
