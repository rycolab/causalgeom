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

def prep_generated_data(model_name, concept, nucleus, max_all_hs=500000):
    l0_hs_wff, l1_hs_wff, other_hs = load_generated_hs_wff(
        model_name, concept, nucleus=nucleus
    )
    p_c = compute_p_c_bin(l0_hs_wff, l1_hs_wff)

    #TODO: will need to do a version of this that returns tokens not hs too
    l0_hs_wff = [x for x,_,_,_ in l0_hs_wff]
    l1_hs_wff = [x for x,_,_,_ in l1_hs_wff]

    #other_hs_no_x = [x for x in other_hs]
    all_hs = np.vstack(l0_hs_wff + l1_hs_wff + other_hs)
    if all_hs.shape[0] > max_all_hs:
        idx = np.arange(all_hs.shape[0])
        np.random.shuffle(idx)
        all_hs = all_hs[idx[:max_all_hs]]

    l0_hs_wff = torch.tensor(l0_hs_wff, dtype=torch.float32)
    l1_hs_wff = torch.tensor(l1_hs_wff, dtype=torch.float32)
    all_hs = torch.tensor(all_hs, dtype=torch.float32)

    return p_c, l0_hs_wff, l1_hs_wff, all_hs

#########################################
# Distribution Computation              #
#########################################
def compute_inner_loop_qxhs(mode, h, sampled_hs, P, I_P, V, processor=None):
    #TODO: this version is unnecessary -- where this is used,
    # replace it with an mh input (doing the repeat before calling
    # the batched version)
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
    """ 
    Dimensions of nmH and other_nmH:
    - nwords x m_samples x d (single token)
    - nwords x max_n_tokens x m_samples x d

    returns: nwords x max_n_tokens x m_samples x |vocab| (distributions)
    """
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
    pxnewh = softmax(logits.cpu(), axis=-1)
    #end = time.time()
    #print(end - start)
    return pxnewh