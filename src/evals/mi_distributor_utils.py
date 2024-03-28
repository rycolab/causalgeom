import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import torch
import random 
from scipy.special import softmax

from data.filter_generations import load_generated_hs_wff

#########################################
# Data Handling                         #
#########################################
def compute_p_c_bin(l0_hs, l1_hs):
    c_counts = np.array([len(l0_hs), len(l1_hs)])
    p_c = c_counts / np.sum(c_counts)
    return p_c

def prep_generated_data(model_name, concept, nucleus):
    l0_hs_wff, l1_hs_wff, other_hs = load_generated_hs_wff(
        model_name, concept, nucleus=nucleus
    )
    #TODO: will need to do a version of this that returns tokens not hs too
    all_concept_hs = [x for x,_,_,_ in l0_hs_wff + l1_hs_wff]
    #other_hs_no_x = [x for x in other_hs]
    all_hs = np.vstack(all_concept_hs + other_hs)

    p_c = compute_p_c_bin(l0_hs_wff, l1_hs_wff)
    return p_c, l0_hs_wff, l1_hs_wff, all_hs

#########################################
# Distribution Computation              #
#########################################
def compute_inner_loop_qxhs(mode, h, all_hs, P, I_P, V, msamples, processor=None):
    """ mode param determines whether averaging over hbot or hpar"""
    all_pxnewh = []
    idx = np.arange(0, all_hs.shape[0])
    np.random.shuffle(idx)
    for other_h in all_hs[idx[:msamples]]:
        if mode == "hbot":
            #TODO: check that this is the same as other_h.T @ I_P...
            newh = other_h @ I_P + h @ P
        elif mode == "hpar":
            newh = h @ I_P + other_h @ P
        else:
            raise ValueError(f"Incorrect mode {mode}")
        logits = V @ newh
        if processor is not None:
            logits = torch.FloatTensor(logits).unsqueeze(0)
            tokens = torch.LongTensor([0]).unsqueeze(0)
            logits = processor(tokens, logits).squeeze(0).numpy()
        pxnewh = softmax(logits)
        all_pxnewh.append(pxnewh)
    return np.vstack(all_pxnewh)