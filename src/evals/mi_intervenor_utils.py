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
def prep_int_generated_data(model_name, concept, nucleus, 
    torch_dtype, max_n_all_hs):
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

    return all_hs