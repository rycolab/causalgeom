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
def compute_inner_loop_qxhs(mode, h, sampled_hs, P, I_P, V, processor=None):
    """ mode param determines whether averaging over hbot or hpar
    everything should be on GPU!
    """
    mh = h.repeat(sampled_hs.shape[0], 1)
    if mode == "hbot":
        newh = sampled_hs @ I_P + mh @ P
    elif mode == "hpar":
        newh = mh @ I_P + sampled_hs @ P
    else:
        raise ValueError(f"Incorrect mode {mode}")
    logits = newh @ V.T
    if processor is not None:
        raise NotImplementedError("haven't updated this")
        #logits = torch.FloatTensor(logits).unsqueeze(0)
        #tokens = torch.LongTensor([0]).unsqueeze(0)
        #logits = processor(tokens, logits).squeeze(0).numpy()
    pxnewh = softmax(logits.cpu(), axis=1)
    return pxnewh

def compute_batch_inner_loop_qxhs(mode, nmH, other_nmH, P, I_P, V, processor=None):
    """ n is the batch size of the original h's, 
    m is the number of other h's to average over. """
    assert nmH.shape == other_nmH.shape, "Incorrect inputs"
    #start = time.time()
    if mode == "hbot":
        newh = other_nmH @ I_P + nmH @ P
    elif mode == "hpar":
        newh = nmH @ I_P + other_nmH @ P
    else:
        raise ValueError(f"Incorrect mode {mode}")
    logits = newh @ V.T
    if processor is not None:
        raise NotImplementedError("haven't updated this")
        #logits = torch.FloatTensor(logits).unsqueeze(0)
        #tokens = torch.LongTensor([0]).unsqueeze(0)
        #logits = processor(tokens, logits).squeeze(0).numpy()
    pxnewh = softmax(logits.cpu(), axis=2)
    #end = time.time()
    #print(end - start)
    return pxnewh