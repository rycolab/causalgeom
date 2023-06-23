import warnings
import logging
import os
import sys
import coloredlogs
import torch
import pickle
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import CamembertForMaskedLM, CamembertTokenizer

sys.path.append('./src/')

from paths import DATASETS

from data.embed_wordlists.embedder import get_emb_outfile_paths, load_concept_token_lists

#%%##############################
# Loading Model Word Lists.     #
#################################

def load_model_eval(concept_name, model_name):
    #TODO: adapt this to load token lists and V, and adapt downstream functions
    word_emb_path, lemma_p_path, lemma_0_path, lemma_1_path = get_emb_outfile_paths(concept_name, model_name)
    
    if concept_name == "number":
        concept_prob_path = os.path.join(DATASETS, "processed/en/word_lists/number_marginals.pkl")
    elif concept_name == "gender":
        concept_prob_path = os.path.join(DATASETS, "processed/fr/word_lists/gender_marginals.pkl")
    else:
        concept_prob_path = ""

    with open(concept_prob_path, 'rb') as f:
        #concept_prob = pickle.load(f).to_numpy().tolist()
        concept_marginals = pickle.load(f)
    
    other_emb = np.load(word_emb_path)
    pair_probs = np.load(lemma_p_path)
    l0_emb = np.load(lemma_0_path)
    l1_emb = np.load(lemma_1_path)
    return other_emb, l0_emb, l1_emb, pair_probs, concept_marginals

