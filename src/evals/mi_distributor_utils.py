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
from paths import OUT

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
# Eval Directory Handling               #
#########################################
def get_mt_eval_directory(run_path, concept, model_name, 
    output_folder, source, iteration):
    rundir = os.path.dirname(run_path)
    rundir_name = os.path.basename(rundir)

    run_id = run_path[-27:-4]
    outdir = os.path.join(
        OUT, 
        f"mt_eval/{concept}/{model_name}/mt_eval_{rundir_name}/"
        f"{output_folder}/run_{run_id}/source_{source}/evaliter_{iteration}"
    )
    return outdir

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
def compute_batch_inner_loop_qxhs(mode, nmH, other_nmH, 
    P, I_P, V, gpu_out, processor=None):
    """ 
    Dimensions of nmH and other_nmH:
    - msamples x nwords x max_n_tokens x d (multi token)

    returns: msamples x nwords x max_n_tokens x |vocab| (distributions)
    """
    assert nmH.shape == other_nmH.shape, "Incorrect inputs"
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
    log_pxnewh = torch.nn.functional.log_softmax(logits, dim=-1)
    if gpu_out:
        return log_pxnewh
    else:
        return log_pxnewh.cpu()
    
#def OLD_sample_gen_all_hs_batched(n_ntok_H, msamples, gen_all_hs, device):
#    """ input dimensions: 
#    - n_ntok_H: nwords x (max_ntokens + 1) x d
#    output: nwords x (max_ntokens + 1) x msamples x d
#    """
#    n_ntok_m_H = n_ntok_H[:, :, None, :].repeat(1, 1, msamples, 1)
#    idx = np.random.randint(
#        0, gen_all_hs.shape[0], 
#        n_ntok_H.shape[0]*n_ntok_H.shape[1]*msamples
#    )
#    other_hs = gen_all_hs[idx].to(device)
#    other_hs_view = other_hs.view(
#        (n_ntok_H.shape[0], n_ntok_H.shape[1], 
#            msamples, n_ntok_H.shape[2])
#    )
#    return n_ntok_m_H, other_hs_view

def sample_gen_all_hs_batched(n_ntok_H, msamples, gen_all_hs, device):
    """ input dimensions: 
    - n_ntok_H: nwords x (max_ntokens + 1) x d
    output: msamples x nwords x (max_ntokens + 1) x d
    - where the other h sample is the same for the (max_ntokens + 1) dim
    """
    m_n_ntok_H = n_ntok_H[None, :, :, :].repeat(msamples, 1, 1, 1)
    idx = np.random.randint(
        0, gen_all_hs.shape[0], 
        n_ntok_H.shape[0]*msamples
    )
    other_hs = gen_all_hs[idx].to(device)
    other_hs_view = other_hs.view(
        (msamples, n_ntok_H.shape[0], n_ntok_H.shape[2])
    )
    other_hs_final = other_hs_view[:,:,None,:].repeat(1,1,n_ntok_H.shape[1],1)
    return m_n_ntok_H, other_hs_final

#def OLD_compute_qxh_batch(method, nntokH, msamples, all_hs, 
#        P, I_P, V, gpu_out, device):
#    nntokmH, other_nntokmH = sample_gen_all_hs_batched(
#        nntokH, msamples, all_hs, device)
#    qxhs = compute_batch_inner_loop_qxhs(
#        method, nntokmH, other_nntokmH, 
#        P, I_P, V, gpu_out
#    )
#    return qxhs

def compute_qxh_batch(method, nntokH, msamples, all_hs, 
        P, I_P, V, gpu_out, device):
    mnntokH, other_mnntokH = sample_gen_all_hs_batched(
        nntokH, msamples, all_hs, device)
    qxhs = compute_batch_inner_loop_qxhs(
        method, mnntokH, other_mnntokH, 
        P, I_P, V, gpu_out
    )
    return qxhs

def compute_log_pxh_batch(nntokH, V, gpu_out):
    logits = nntokH @ V.T
    log_pxnewh = torch.nn.functional.log_softmax(logits, dim=-1)
    if gpu_out:
        return log_pxnewh
    else:
        return log_pxnewh.cpu()


#def compute_pxh_batch_handler(method, nntokH, msamples, all_hs, 
#    P, I_P, V, gpu_out, device):
#    if method in ["hbot","hpar"]:
#        return compute_qxh_batch(method, nntokH, msamples, all_hs, 
#                    P, I_P, V, gpu_out, device)
#    elif method == "h":
#        return compute_pxh_batch(nntokH, V, gpu_out)
#    else:
#        raise ValueError(f"Incorrect method arg")

#def compute_p_words(batch_token_list, batch_pxh, pad_token_id, new_word_tokens):
#    """ 
#    TODO: would be nice to turn this into a vectorized operation
#    just dont know how to do variable length indexing
#    expected dimensions:
#    - batch_token_list: n_words x max_n_tokens
#    - batch_pxh: nwords x (max_n_tokens + 1) x |vocabulary|
#
#    output: len nwords list of word probabilities
#    """
#    all_word_probs = []
#    for word_tokens, word_probs in zip(batch_token_list, batch_pxh):
#        counter=0
#        p_word=1
#        while (counter < len(word_tokens) and 
#                word_tokens[counter] != pad_token_id):
#            p_word = p_word * word_probs[counter, word_tokens[counter]]
#            counter+=1
#        new_word_prob = word_probs[counter, new_word_tokens].sum()
#        p_word = p_word * new_word_prob
#        all_word_probs.append(p_word)
#    return all_word_probs

def fast_compute_p_words(batch_token_list, batch_log_pxh, 
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
    #batch_log_pxh = batch_pxh.log()
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
    batch_pxh = batch_log_pxh.exp()
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
    return final_log_p_x.exp().unsqueeze(0).cpu()

#def fast_compute_m_p_words_log(batch_token_list, batch_log_pxh, 
#                         pad_token_id, new_word_tokens, device):
#    """ 
#    expected dimensions:
#    - batch_token_list: n_words x max_n_tokens
#    - batch_pxh: nwords x (max_n_tokens + 1) x msamples x |vocabulary|
#    - new_word_tokens: list of new word tokens for the model
#
#    output: list of word probabilities
#    """
#    #start = time.time()
#    #batch_pxh = torch.tensor(batch_pxh).to(self.device)
#    #batch_log_pxh = batch_pxh.log()
#    seq_lens = (batch_token_list != pad_token_id).sum(1)
#    n = len(batch_token_list)
#    m = batch_log_pxh.shape[0]
#    max_len = max(seq_lens)
#    seq_idxs = torch.eye(batch_log_pxh.shape[1]).to(device)
#    log_p_x = torch.zeros((m, n)).to(device)
#    for i in range(max_len): # Could be vectorized further too if a bottleneck
#        #i = 1
#        tok_idxs = batch_token_list[:, i]
#        tok_pxhs = batch_log_pxh[:, :, i, tok_idxs]
#        next_pxhs = (tok_pxhs * seq_idxs).sum(-2)
#        log_p_x += next_pxhs * (i < seq_lens)
#        #p_x *= torch.where((seq_lens > i), tok_pxhs, ones)
#
#    # pnewword
#    v = batch_log_pxh.shape[-1]
#    v_mask = torch.isin(
#        torch.arange(v), 
#        torch.tensor(new_word_tokens), 
#        assume_unique=True
#    ).to(device)
#    masked_batch_log_pxh = batch_log_pxh * v_mask
#    log_p_new = masked_batch_log_pxh.logsumexp(-1)
#    log_p_new_given_x = log_p_new[:, :, seq_lens]
#    log_p_new_given_x_proj = (log_p_new_given_x * seq_idxs).sum(-2)
#    log_final_log_p_x = log_p_x + log_p_new_given_x_proj
#    #end = time.time()
#    #print(end-start)
#    return log_final_log_p_x.exp()

def fast_compute_m_p_words(batch_token_list, batch_log_pxh, 
                         pad_token_id, new_word_tokens, device):
    """ 
    expected dimensions:
    - batch_token_list: n_words x max_n_tokens
    - batch_pxh: nwords x (max_n_tokens + 1) x msamples x |vocabulary|
    - new_word_tokens: list of new word tokens for the model

    output: list of word probabilities
    """
    #start = time.time()
    #batch_pxh = torch.tensor(batch_pxh).to(self.device)
    #batch_log_pxh = batch_pxh.log()
    seq_lens = (batch_token_list != pad_token_id).sum(1)
    n = len(batch_token_list)
    m = batch_log_pxh.shape[0]
    max_len = max(seq_lens)
    seq_idxs = torch.eye(batch_log_pxh.shape[1]).to(device)
    log_p_x = torch.zeros((m, n)).to(device)
    for i in range(max_len): # Could be vectorized further too if a bottleneck
        #i = 1
        tok_idxs = batch_token_list[:, i]
        tok_pxhs = batch_log_pxh[:, :, i, tok_idxs]
        next_pxhs = (tok_pxhs * seq_idxs).sum(-2)
        log_p_x += next_pxhs * (i < seq_lens)
        #p_x *= torch.where((seq_lens > i), tok_pxhs, ones)

    # pnewword
    #TODO: make this work with log
    batch_pxh = batch_log_pxh.exp()
    v = batch_pxh.shape[-1]
    v_mask = torch.isin(
        torch.arange(v), 
        torch.tensor(new_word_tokens), 
        assume_unique=True
    ).to(device)
    p_new = (batch_pxh * v_mask).sum(-1)
    p_new_given_x = p_new[:, :, seq_lens]
    p_new_given_x_proj = (p_new_given_x * seq_idxs).sum(-2)
    final_log_p_x = log_p_x + p_new_given_x_proj.log()
    #end = time.time()
    #print(end-start)
    return final_log_p_x.exp()

#########################################
# New Single Intervention Functions     #
#########################################
def apply_projection(hs, other_hs, mode, P, I_P):
    assert hs.shape == other_hs.shape, "Incorrect inputs"
    if mode == "hbot":
        newh = other_hs @ I_P + hs @ P
    elif mode == "hpar":
        newh = hs @ I_P + other_hs @ P
    else:
        raise ValueError(f"Incorrect mode {mode}")
    return newh

def sample_other_hs(other_hs, nwords, msamples, device):
    idx = np.random.randint(
        0, other_hs.shape[0], 
        nwords*msamples #nwords x msamples
    )
    other_hs_sample = other_hs[idx].to(device)
    # msamples x nwords x d
    other_hs_view = other_hs_sample.view(
        (msamples, nwords, other_hs.shape[1])
    )
    return other_hs_view

def intervene_hs(n_ntok_H, method, msamples, gen_all_hs, P, I_P, device):
    """ input dimensions:
    - n_ntok_H: nwords x (max_ntokens + 1) x d
    output: msamples x nwords x (max_ntokens + 1) x d
    where the first of max_ntokens+1 hs has been intervened upon
    according to method
    """
    # repeating and splitting batch hs
    m_n_ntok_H = n_ntok_H[None, :, :, :].repeat(msamples, 1, 1, 1)
    first_hs = m_n_ntok_H[:, :, 0, :]
    next_hs = m_n_ntok_H[:, :, 1:, :]

    # sampling other hs
    sampled_hs = sample_other_hs(
       gen_all_hs, n_ntok_H.shape[0], msamples, device
    )

    # intervention on first hs
    first_hs_int = apply_projection(
        first_hs, sampled_hs, method, P, I_P
    )
    first_hs_int_view = first_hs_int[:, :, None, :]

    # combine first and next hs after intervention
    hs_int = torch.cat((first_hs_int_view, next_hs), dim=2)
    return hs_int
