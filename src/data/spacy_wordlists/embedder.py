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
from utils.lm_loaders import get_tokenizer, get_V, GPT2_LIST, BERT_LIST, SUPPORTED_AR_MODELS

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

EN_NUMBER_VERB_LIST_PATH = os.path.join(DATASETS, f"processed/en/number_word_lists/verb_pair_list.tsv")
EN_NUMBER_WORD_LIST_PATH = os.path.join(DATASETS, f"processed/en/number_word_lists/number_other_list.tsv")

FR_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/fr/gender_word_lists/adj_pair_list.tsv")
FR_WORD_LIST_PATH = os.path.join(DATASETS, f"processed/fr/gender_word_lists/gender_other_list.tsv")

CEBAB_FOOD_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/food_adjs.tsv")
CEBAB_AMBIANCE_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/ambiance_adjs.tsv")
CEBAB_NOISE_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/noise_adjs.tsv")
CEBAB_SERVICE_ADJ_LIST_PATH = os.path.join(DATASETS, f"processed/CEBaB/word_lists_raw/service_adjs.tsv")

CEBAB_FOOD_OTHER_LIST_PATH = os.path.join(DATASETS, f"processed/en/CEBaB_word_lists/food_other_list.tsv")
CEBAB_AMBIANCE_OTHER_LIST_PATH = os.path.join(DATASETS, f"processed/en/CEBaB_word_lists/ambiance_other_list.tsv")
CEBAB_NOISE_OTHER_LIST_PATH = os.path.join(DATASETS, f"processed/en/CEBaB_word_lists/noise_other_list.tsv")
CEBAB_SERVICE_OTHER_LIST_PATH = os.path.join(DATASETS, f"processed/en/CEBaB_word_lists/service_other_list.tsv")

#%%#################
# Paths            #
####################
def get_wordlist_paths(concept):
    """ Returns other_list, concept_word_pair_list
    """
    if concept == "number":
        return EN_NUMBER_WORD_LIST_PATH, EN_NUMBER_VERB_LIST_PATH
    elif concept == "gender":
        return FR_WORD_LIST_PATH, FR_ADJ_LIST_PATH
    elif concept == "food":
        return CEBAB_FOOD_OTHER_LIST_PATH, CEBAB_FOOD_ADJ_LIST_PATH
    elif concept == "ambiance":
        return CEBAB_AMBIANCE_OTHER_LIST_PATH, CEBAB_AMBIANCE_ADJ_LIST_PATH
    elif concept == "noise":
        return CEBAB_NOISE_OTHER_LIST_PATH, CEBAB_NOISE_ADJ_LIST_PATH
    elif concept == "service":
        return CEBAB_SERVICE_OTHER_LIST_PATH, CEBAB_SERVICE_ADJ_LIST_PATH
    else:
        raise NotImplementedError("Invalid dataset name")

def get_emb_outfile_paths(concept, model_name):
    """ back in the single token days these were the embeddings of the single token word lists 
    """
    if concept == "number":
        word_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_word_embeds.npy")
        #verb_p_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_verb_p.npy")
        sg_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_sg_embeds.npy")
        pl_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_pl_embeds.npy")
        return word_emb_outfile, None, sg_emb_outfile, pl_emb_outfile
    elif concept == "gender":
        word_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_word_embeds.npy")
        #adj_p_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_adj_p.npy")
        masc_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_masc_embeds.npy")
        fem_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_fem_embeds.npy")
        return word_emb_outfile, None, masc_emb_outfile, fem_emb_outfile
    elif concept in ["food", "ambiance", "service", "noise"]:
        word_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_word_embeds.npy")
        l0_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_l0_embeds.npy")
        l1_emb_outfile = os.path.join(DATASETS, f"processed/embedded_word_lists/{concept}/{model_name}_l1_embeds.npy")
        return word_emb_outfile, None, l0_emb_outfile, l1_emb_outfile
    else:
        raise ValueError("Invalid dataset name")

def get_token_list_outfile_paths(concept, model_name, single_token):
    """ paths to tokenized word lists """
    if concept == "number":
        other_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_word_token_list.npy")
        l0_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_sg_token_list.npy")
        l1_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_pl_token_list.npy")
    elif concept == "gender":
        other_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_word_token_list.npy")
        l0_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_masc_token_list.npy")
        l1_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_fem_token_list.npy")
    elif concept in ["food", "ambiance", "service", "noise"]:
        other_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_word_token_list.npy")
        l0_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_l0_token_list.npy")
        l1_outfile = os.path.join(DATASETS, f"processed/tokenized_lists/{concept}/{model_name}_l1_token_list.npy")
    else:
        raise ValueError("Invalid concept")

    if not single_token:
        #if other_outfile is not None:
        other_outfile = other_outfile[:-4] + "_all_lengths.npy"
        l0_outfile = l0_outfile[:-4] + "_all_lengths.npy"
        l1_outfile = l1_outfile[:-4] + "_all_lengths.npy"
    return other_outfile, l0_outfile, l1_outfile

def load_concept_token_lists(concept, model_name, single_token):
    other_tl_file, l0_tl_file, l1_tl_file = get_token_list_outfile_paths(
        concept, model_name, single_token
    )
    l0_tl = np.load(l0_tl_file, allow_pickle=True)
    l1_tl = np.load(l1_tl_file, allow_pickle=True)
    
    if other_tl_file is not None:
        other_tl = np.load(other_tl_file, allow_pickle=True)
    else:
        other_tl = None
    return l0_tl, l1_tl, other_tl

def define_add_space(model_name):
    if model_name in GPT2_LIST:
        return True
    else: 
        return False

#%%#################
# Helpers         #
####################
def tokenize_word(model_name, tokenizer, word, masked, add_space):
    if type(word) != str:
        return []
    if add_space:
        word = " "+word
    if masked:
        return tokenizer(word)["input_ids"][1:-1]
    else:
        if model_name in GPT2_LIST:
            return tokenizer(word)["input_ids"]
        elif model_name == "llama2":
            return tokenizer(word)["input_ids"][1:]
        else:
            raise NotImplementedError(f"Model {model_name} not supported")

def tokenize_word_handler(model_name, tokenizer, word, add_space=False):
    if model_name in GPT2_LIST:
        return tokenize_word(model_name, tokenizer, word, masked=False, add_space=add_space)
    elif model_name in BERT_LIST:
        return tokenize_word(model_name, tokenizer, word, masked=True, add_space=False)
    elif model_name == "llama2":
        return tokenize_word(model_name, tokenizer, word, masked=False, add_space=add_space)
    else:
        raise ValueError("Incorrect model name")

def get_unique_word_list(wordlist_path, model_name, tokenizer, add_space, single_token):
    wl = pd.read_csv(wordlist_path, index_col=0, sep="\t")[["word"]]
    wl.drop_duplicates(inplace=True)
    wl["input_ids_word"] = wl["word"].apply(
        lambda x: tokenize_word_handler(model_name, tokenizer, x, add_space)
    )
    wl["ntokens"] = wl["input_ids_word"].apply(lambda x: len(x))
    if single_token:
        wl_1tok = wl[wl["ntokens"]==1]
        wl_1tok["first_id_word"] = wl_1tok["input_ids_word"].apply(lambda x: int(x[0]))
        word_tok = wl_1tok["first_id_word"].to_numpy()
        word_tok_unq, word_tok_counts = np.unique(word_tok, return_counts=True)
        return word_tok_unq
    else:
        wl_nonempty = wl[wl["ntokens"]!=0]
        return np.unique(wl_nonempty["input_ids_word"].to_numpy())

def get_unique_lemma_lists(lemma_list_path, model_name, tokenizer, add_space, single_token):
    #with open(lemma_list_path, 'rb') as f:      
    #    ll = pickle.load(f)
    ll = pd.read_csv(lemma_list_path, index_col=0, sep="\t")

    for col in ["lemma_0", "lemma_1"]:
        ll[f"{col}_input_ids"] = ll[col].apply(
            lambda x: tokenize_word_handler(model_name, tokenizer, x, add_space))
        ll[f"{col}_ntokens"] = ll[f"{col}_input_ids"].apply(
            lambda x: len(x)
        )

    if single_token:
        ll["1tok"] = (ll["lemma_0_ntokens"] == 1) & (ll["lemma_1_ntokens"]==1)

        ll.drop(ll[ll["1tok"] != True].index, inplace=True)

        ll["lemma_0_tok"] = ll["lemma_0_input_ids"].apply(lambda x: x[0])
        ll["lemma_1_tok"] = ll["lemma_1_input_ids"].apply(lambda x: x[0])

        lemma_p = ll[["pair_p_0", "pair_p_1"]].to_numpy()
        ll_l0_tok = ll["lemma_0_tok"].to_numpy()
        ll_l1_tok = ll["lemma_1_tok"].to_numpy()
        return lemma_p, ll_l0_tok, ll_l1_tok
    else:
        ll["0tok"] = (ll["lemma_0_ntokens"] == 0) & (ll["lemma_1_ntokens"]==0)
        assert ll[ll["0tok"] == True].shape[0] == 0, "Tokenization issue"
        #ll.drop(ll[ll["0tok"] == True].index, inplace=True)
            
        if all(i in ll.columns for i in ["pair_p_0", "pair_p_1"]):
            lemma_p = ll[["pair_p_0", "pair_p_1"]].to_numpy()
        else:
            lemma_p = None

        ll_l0_tok = ll["lemma_0_input_ids"].to_numpy()
        ll_l1_tok = ll["lemma_1_input_ids"].to_numpy()
        return lemma_p, ll_l0_tok, ll_l1_tok

def embed_and_export_list(token_list, V, outfile):
    emb = V[token_list]
    np.save(outfile, emb)

#%%#################
# Arguments        #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Embedding word and target lists')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["number", "gender", "food", "ambiance", "service", "noise"],
        help="Concept to create embedded word lists for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        help="Models to create embedding files for"
    )
    argparser.add_argument(
        "-single_token",
        action="store_true",
        default=False,
        help="Whether to filter word list to single token pairs"
    )
    return argparser.parse_args()

#%%#################
# Main             #
####################
if __name__=="__main__":

    args = get_args()
    logging.info(args)
    concept = args.concept
    model_name = args.model
    single_token = args.single_token
    #concept = "food"
    #model_name = "gpt2-large"
    #single_token = False
    
    if concept in ["food", "ambiance", "service", "noise"] and single_token:
        raise NotImplementedError("Single token version of unpaired word lists"
            "not implemented, either pair the lists or implement.")

    logging.info(f"Tokenizing and saving embeddings from word and lemma lists for model {model_name}")
    wordlist_path, lemmalist_path = get_wordlist_paths(concept)
    word_emb_outfile, lemma_p_outfile, l0_emb_outfile, l1_emb_outfile = get_emb_outfile_paths(
        concept, model_name)
    other_tl_outfile, l0_tl_outfile, l1_tl_outfile = get_token_list_outfile_paths(
        concept, model_name, single_token)

    for filepath in [l0_emb_outfile, l0_tl_outfile]:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)

    tokenizer = get_tokenizer(model_name)
    V = get_V(model_name)
    add_space = define_add_space(model_name)
    
    if wordlist_path is not None:
        wl = get_unique_word_list(
            wordlist_path, model_name, tokenizer, add_space, single_token
        )
        logging.info(f"Tokenized word list of length: {len(wl)}")
        np.save(other_tl_outfile, wl)
        if single_token:
            embed_and_export_list(wl, V, word_emb_outfile)
            logging.info(f"Tokenized and exported word embeds to: {word_emb_outfile}")

    lemma_p, l0_tok, l1_tok = get_unique_lemma_lists(
        lemmalist_path, model_name, tokenizer, add_space, single_token
    )
    logging.info(f"Lemma 0 list of length: {len(l0_tok)}")
    np.save(l0_tl_outfile, l0_tok)
    logging.info(f"Lemma 1 list of length: {len(l1_tok)}")
    np.save(l1_tl_outfile, l1_tok)
    if lemma_p is not None:
        np.save(lemma_p_outfile, lemma_p)
        logging.info(f"Exported token lemma probs to: {lemma_p_outfile}")

    if single_token:
        logging.info(f"Single token lemma 0 list of length: {len(l0_tok)}")
        logging.info(f"Single token lemma 1 list of length: {len(l1_tok)}")
        embed_and_export_list(l0_tok, V, l0_emb_outfile)
        logging.info(f"Tokenized and exported lemma 0 embeds to: {l0_emb_outfile}")
        embed_and_export_list(l1_tok, V, l1_emb_outfile)
        logging.info(f"Tokenized and exported lemma 1 embeds to: {l1_emb_outfile}")

    
