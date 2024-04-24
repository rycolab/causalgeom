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

from data.filter_generations import load_filtered_generations


#########################################
# Arg Handling                         #
#########################################
def get_nucleus_arg(source):
    if source in ["natural", "gen_nucleus"]:
        nucleus = True
    elif source == "gen_normal":
        nucleus = False
    else:
        raise ValueError(f"Incorrect {source} argument")
    return nucleus

#########################################
# Data Handling                         #
#########################################
def compute_p_c_bin(l0_hs, l1_hs):
    c_counts = np.array([len(l0_hs), len(l1_hs)])
    p_c = c_counts / np.sum(c_counts)
    return p_c

def prep_generated_data(model_name, concept, nucleus, max_all_hs=500000):
    l0_gens, l1_gens, other_hs = load_filtered_generations(
        model_name, concept, nucleus=nucleus
    )
    p_c = compute_p_c_bin(l0_gens, l1_gens)

    l0_hs = [x for x,_,_,_ in l0_gens]
    l1_hs = [x for x,_,_,_ in l1_gens]

    all_hs = np.vstack(l0_hs + l1_hs + other_hs)
    del l0_hs
    del l1_hs

    if all_hs.shape[0] > max_all_hs:
        idx = np.arange(all_hs.shape[0])
        np.random.shuffle(idx)
        all_hs = all_hs[idx[:max_all_hs]]

    all_hs = torch.tensor(all_hs, dtype=torch.float32)

    l0_cxt_toks = [all_tok[:-len(fact)] for _,fact,_,all_tok in l0_gens]
    l1_cxt_toks = [all_tok[:-len(fact)] for _,fact,_,all_tok in l1_gens]
    logging.info(f"Loaded generated hs: model {model_name}, "
                 f"concept {concept}, nucleus {nucleus}"
                 f"l0 {len(l0_cxt_toks)}, l1 {len(l1_cxt_toks)}"
                 f"other {all_hs.shape[0]}")
    return p_c, l0_cxt_toks, l1_cxt_toks, all_hs


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

def compute_batch_inner_loop_qxhs(mode, nmH, other_nmH, 
    P, I_P, V, gpu_out, processor=None):
    """ 
    Dimensions of nmH and other_nmH:
    - nwords x m_samples x d (single token)
    - nwords x max_n_tokens x m_samples x d

    returns: nwords x max_n_tokens x m_samples x |vocab| (distributions)
    """
    #TODO: make this a log softmax
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
    if gpu_out:
        pxnewh = torch.nn.functional.softmax(logits, dim=-1)
    else:
        pxnewh = softmax(logits.cpu(), axis=-1)
    #end = time.time()
    #print(end - start)
    return pxnewh


def sample_gen_all_hs_batched(n_ntok_H, msamples, gen_all_hs, device):
    """ input dimensions: 
    - n_ntok_H: nwords x (max_ntokens + 1) x d
    output: nwords x (max_ntokens + 1) x msamples x d
    """
    n_ntok_m_H = n_ntok_H[:, :, None, :].repeat(1, 1, msamples, 1)
    idx = np.random.randint(
        0, gen_all_hs.shape[0], 
        n_ntok_H.shape[0]*n_ntok_H.shape[1]*msamples
    )
    other_hs = gen_all_hs[idx].to(device)
    other_hs_view = other_hs.view(
        (n_ntok_H.shape[0], n_ntok_H.shape[1], 
            msamples, n_ntok_H.shape[2])
    )
    return n_ntok_m_H, other_hs_view
    
    
def compute_pxh_batch_handler(method, nntokH, P, I_P, V, gpu_out):
    if method in ["hbot", "hpar"]:
        nntokmH, other_nntokmH = sample_gen_all_hs_batched(nntokH)
        qxhs = compute_batch_inner_loop_qxhs(
            method, nntokmH, other_nntokmH, 
            P, I_P, V, gpu_out
        )
        return qxhs.mean(axis=-2)
    elif method == "h":
        logits = nntokH @ V.T
        if gpu_out:
            pxh = torch.nn.functional.softmax(logits, dim=-1)
        else:
            pxh = softmax(logits.cpu(), axis=-1)
        return pxh
    else:
        raise ValueError(f"Incorrect method argument {method}")


def compute_p_words(batch_token_list, batch_pxh, pad_token_id, new_word_tokens):
    """ 
    TODO: would be nice to turn this into a vectorized operation
    just dont know how to do variable length indexing
    expected dimensions:
    - batch_token_list: n_words x max_n_tokens
    - batch_pxh: nwords x (max_n_tokens + 1) x |vocabulary|

    output: len nwords list of word probabilities
    """
    all_word_probs = []
    for word_tokens, word_probs in zip(batch_token_list, batch_pxh):
        counter=0
        p_word=1
        while (counter < len(word_tokens) and 
                word_tokens[counter] != pad_token_id):
            p_word = p_word * word_probs[counter, word_tokens[counter]]
            counter+=1
        new_word_prob = word_probs[counter, new_word_tokens].sum()
        p_word = p_word * new_word_prob
        all_word_probs.append(p_word)
    return all_word_probs


def fast_compute_p_words(batch_token_list, batch_pxh, 
                            pad_token_id, new_word_tokens, device):
    """ 
    expected dimensions:
    - batch_token_list: n_words x max_n_tokens
    - batch_pxh: nwords x (max_n_tokens + 1) x |vocabulary|
    - new_word_tokens: list of new word tokens for the model

    output: list of word probabilities
    """
    #start = time.time()
    #batch_pxh = torch.tensor(batch_pxh).to(self.device)
    batch_log_pxh = batch_pxh.log()
    seq_lens = (batch_token_list != pad_token_id).sum(1)
    n = len(batch_token_list)
    max_len = max(seq_lens)
    seq_idxs = torch.arange(n).to(device)
    log_p_x = torch.zeros(n).to(device)
    for i in range(max_len): # Could be vectorized further too if a bottleneck
        tok_idxs = batch_token_list[:, i]
        tok_pxhs = batch_log_pxh[seq_idxs, i, tok_idxs]
        log_p_x += tok_pxhs * (i < seq_lens)
        #p_x *= torch.where((seq_lens > i), tok_pxhs, ones)

    # pnewword
    #TODO: make this work with log
    v = batch_pxh.shape[2]
    v_mask = torch.isin(
        torch.arange(v), 
        torch.tensor(new_word_tokens), 
        assume_unique=True
    ).to(device)
    p_new = batch_pxh * v_mask
    p_new_given_x = p_new[seq_idxs, seq_lens].sum(1)
    final_log_p_x = log_p_x + p_new_given_x.log()
    #end = time.time()
    #print(end-start)
    return final_log_p_x.exp().cpu().tolist()