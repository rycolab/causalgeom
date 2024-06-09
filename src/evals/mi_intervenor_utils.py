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

from data.GenerationsFilter import load_filtered_generations
from utils.lm_loaders import GPT2_LIST, get_max_cxt_length
from paths import OUT
from evals.mi_distributor_utils import get_all_hs, sample_cxt_toks

#%%
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