import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import torch
import random 
from scipy.special import softmax, log_softmax

sys.path.append('..')

from data.filter_generations import load_filtered_generations
from utils.lm_loaders import GPT2_LIST, get_max_cxt_length
from paths import OUT
from evals.mi_distributor_utils import get_all_hs, sample_cxt_toks

#%%
def get_concept_cxt_toks(model_name, l0_gens, l1_gens, 
    max_n_cxts, cxt_max_length_pct=1):
    """ Process and filters generated context strings
    """
    l0_cxt_toks = [all_tok[:-len(fact)] for _,fact,_,all_tok in l0_gens]
    l1_cxt_toks = [all_tok[:-len(fact)] for _,fact,_,all_tok in l1_gens]

    max_cxt_length = get_max_cxt_length(model_name)
    cxt_size_limit = max_cxt_length * cxt_max_length_pct

    l0_cxt_toks = [x for x in l0_cxt_toks if len(x) < cxt_size_limit]
    l1_cxt_toks = [x for x in l1_cxt_toks if len(x) < cxt_size_limit]

    l0_cxt_toks = sample_cxt_toks(l0_cxt_toks, max_n_cxts)
    l1_cxt_toks = sample_cxt_toks(l0_cxt_toks, max_n_cxts)
    return l0_cxt_toks, l1_cxt_toks
    
def prep_int_generated_data(model_name, concept, nucleus, source, 
    torch_dtype, cxt_max_length_pct, max_n_cxts, max_n_all_hs):
    """ Loads generated text and outputs:
    - all_hs: generated hs by the model
    - cxt_toks: either concept or all context tokens, depending on source arg
    
    Args:
    - cxt_max_length_pct: keep only cxt strings with cxt_max_length_pct
        length of model context size
    - max_n_cxts: max number of context strings to return,
                  applied to both concept_cxt_toks and all_cxt_toks
    - max_n_all_hs: max number of all_hs to return
    """
    l0_gens, l1_gens, other_gens = load_filtered_generations(
        model_name, concept, nucleus=nucleus
    )

    all_hs = get_all_hs(
        l0_gens, l1_gens, other_gens, max_n_all_hs, torch_dtype
    )    

    #if source == "natural_concept":
    #    l0_cxt_toks = None
    #    l1_cxt_toks = None
    #    len_l0_cxt_toks = 0
    #    len_l1_cxt_toks = 0
    #else:
    #    l0_cxt_toks, l1_cxt_toks = get_concept_cxt_toks(
    #        model_name, l0_gens, l1_gens, max_n_cxts,
    #        cxt_max_length_pct
    #    )
    #    len_l0_cxt_toks = len(l0_cxt_toks)
    #    len_l1_cxt_toks = len(l1_cxt_toks)
    
    #logging.info(
    #    f"Loaded generated hs: model {model_name}, "
    #    f"concept {concept}, nucleus {nucleus}, "
    #    f"source {source}: \n"
    #    f"- all_hs: {all_hs.shape[0]} \n"
    #    f"- l0_cxt_toks: {len_l0_cxt_toks} \n"
    #    f"- l1_cxt_toks: {len_l1_cxt_toks}"
    #)
    return all_hs#, l0_cxt_toks, l1_cxt_toks


#%%##################
# Dataset Filtering #
#####################
def filter_hs_w_ys(X, facts, foils, y, value):
    idx = np.nonzero(y==value)
    sub_hs, sub_facts, sub_foils = X[idx], facts[idx], foils[idx]
    sub_hs_wff = [x for x in zip(sub_hs, sub_facts, sub_foils)]
    return sub_hs_wff

def filter_all_hs_w_ys(X, facts, foils, y):
    l0_hs_wff = filter_hs_w_ys(
        X, facts, foils, y, 0
    )
    l1_hs_wff = filter_hs_w_ys(
        X, facts, foils, y, 1
    )
    return l0_hs_wff, l1_hs_wff

def sample_filtered_hs(l0_hs, l1_hs, nsamples):
    np.random.shuffle(l0_hs)
    np.random.shuffle(l1_hs)
    ratio = len(l1_hs)/len(l0_hs)
    if ratio > 1:
        l0_hs = l0_hs[:nsamples]
        l1_hs = l1_hs[:int((nsamples*ratio))]
    else:
        ratio = len(l0_hs) / len(l1_hs)
        l0_hs = l0_hs[:int((nsamples*ratio))]
        l1_hs = l1_hs[:nsamples]
    return l0_hs, l1_hs