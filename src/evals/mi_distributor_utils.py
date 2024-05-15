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

#########################################
# Arg Handling                         #
#########################################
def get_nucleus_arg(source):
    """ Use samples generated by model with nucleus sampling 
    for natural text (test set) samples.
    """
    if source in ["natural_concept", "gen_nucleus_concept", "gen_nucleus_all"]:
        nucleus = True
    elif source in ["gen_ancestral_concept", "gen_ancestral_all"]:
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
def get_all_hs(l0_gens, l1_gens, other_gens, max_all_hs, torch_dtype):
    """ Collects hs from filtered generations,
    stacks them and subsamples to max_all_hs samples.
    """
    l0_hs = [x for x,_,_,_ in l0_gens]
    l1_hs = [x for x,_,_,_ in l1_gens]
    other_hs = [x for x,_ in other_gens]

    all_hs = np.vstack(l0_hs + l1_hs + other_hs)
    del l0_hs
    del l1_hs
    del other_hs

    # subsamples all_hs to a max number to avoid memory issue
    if all_hs.shape[0] > max_all_hs:
        idx = np.arange(all_hs.shape[0])
        np.random.shuffle(idx)
        all_hs = all_hs[idx[:max_all_hs]]

    all_hs = torch.tensor(all_hs, dtype=torch_dtype)
    return all_hs

def sample_cxt_toks(cxt_toks, max_n_cxts):
    if len(cxt_toks) > max_n_cxts:
        cxt_toks = random.sample(
            cxt_toks, max_n_cxts)
    return cxt_toks

def get_cxt_toks(model_name, l0_gens, l1_gens, other_gens, 
    max_n_cxts, cxt_max_length_pct=1):
    l0_cxt_toks = [all_tok[:-len(fact)] for _,fact,_,all_tok in l0_gens]
    l1_cxt_toks = [all_tok[:-len(fact)] for _,fact,_,all_tok in l1_gens]
    other_cxt_toks = [all_tok for _,all_tok in other_gens]

    max_cxt_length = get_max_cxt_length(model_name)
    cxt_size_limit = max_cxt_length * cxt_max_length_pct

    concept_cxt_toks = [
        x for x in l0_cxt_toks + l1_cxt_toks if len(x) < cxt_size_limit
    ]
    del l0_cxt_toks
    del l1_cxt_toks
    other_cxt_toks = [
        x for x in other_cxt_toks if len(x) < cxt_size_limit
    ]
    all_cxt_toks = concept_cxt_toks + other_cxt_toks
    del other_cxt_toks

    concept_cxt_toks = sample_cxt_toks(concept_cxt_toks, max_n_cxts)
    all_cxt_toks = sample_cxt_toks(all_cxt_toks, max_n_cxts)
    return concept_cxt_toks, all_cxt_toks

def prep_generated_data(model_name, concept, nucleus, torch_dtype,
    cxt_max_length_pct=0.9, max_n_cxts=100000, max_n_all_hs=300000):
    """ Loads generated text and outputs:
    - all_hs: generated hs by the model
    - concept_cxt_toks: context strings that induced model to generate concept word
    - all_cxt_toks: generated strings by the model

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

    concept_cxt_toks, all_cxt_toks = get_cxt_toks(
        model_name, l0_gens, l1_gens, other_gens, max_n_cxts,
        cxt_max_length_pct
    )

    logging.info(
        f"Loaded generated hs: model {model_name}, "
        f"concept {concept}, nucleus {nucleus}: \n"
        f"- all_hs: {all_hs.shape[0]} \n"
        f"- concept_cxt_toks: {len(concept_cxt_toks)} \n"
        f"- all_cxt_toks: {len(all_cxt_toks)}"
    )
    return all_hs, concept_cxt_toks, all_cxt_toks

def pad_cxt_list(cxt_toks, max_nsamples, padding_value=-1):
    """ Input: list of tokenized strings
    Output: (nsamples, max_ntokens) dimensional tensor
    """
    if len(cxt_toks) > max_nsamples:
        cxt_toks = random.sample(cxt_toks, max_nsamples)
    torch_cxt_toks = [torch.tensor(x) for x in cxt_toks]
    padded_cxt_toks = torch.nn.utils.rnn.pad_sequence(
        torch_cxt_toks, padding_value=padding_value).T
    return padded_cxt_toks 

#########################################
# Past Key Values Handling              #
#########################################
def duplicate_pkv(pkv, num_repeats):
    return tuple(tuple(torch.cat([tensor] * num_repeats, dim=0) for tensor in layer) for layer in pkv)

#########################################
# Distribution Computation              #
#########################################
def compute_log_pxh_batch(nntokH, V, gpu_out):
    logits = nntokH @ V.T
    log_pxnewh = torch.nn.functional.log_softmax(logits, dim=-1)
    if gpu_out:
        return log_pxnewh
    else:
        return log_pxnewh.cpu()

def compute_log_p_new_word(batch_log_pxh, new_word_tokens, seq_idxs, seq_lens, device):
    v = batch_log_pxh.shape[-1]
    v_mask = torch.isin(
        torch.arange(v), 
        torch.tensor(new_word_tokens), 
        assume_unique=True
    ).to(device)

    log_p_new = (batch_log_pxh + v_mask.log()).logsumexp(dim=-1)
    log_p_new_given_x = log_p_new[seq_idxs, seq_lens]
    return log_p_new_given_x

def fast_compute_p_words(batch_token_list, batch_log_pxh, 
                         pad_token_id, new_word_tokens, device):
    """ 
    expected dimensions:
    - batch_token_list: bs x max_n_tokens
    - batch_pxh: bs x (max_n_tokens + 1) x |vocabulary|
    - new_word_tokens: list of new word tokens for the model

    output: list of word probabilities
    final_log_p_x: (bs)
    """
    seq_lens = (batch_token_list != pad_token_id).sum(1)
    n = len(batch_token_list)
    max_len = max(seq_lens)
    seq_idxs = torch.arange(n).to(device)
    log_p_x = torch.zeros(n).to(device)
    for i in range(max_len): # Could be vectorized further too if a bottleneck
        tok_idxs = batch_token_list[:, i]
        tok_pxhs = batch_log_pxh[seq_idxs, i, tok_idxs]
        log_p_x += tok_pxhs * (i < seq_lens)
    
    if new_word_tokens is not None:
        log_p_new_word = compute_log_p_new_word(
            batch_log_pxh, new_word_tokens, seq_idxs, seq_lens, device
        )
        final_log_p_x = log_p_x + log_p_new_word
        return final_log_p_x.exp().unsqueeze(0).cpu()
    else:
        return log_p_x.exp().unsqueeze(0).cpu()

def compute_m_log_p_new_word(batch_log_pxh, new_word_tokens, seq_idxs, seq_lens, device):
    v = batch_log_pxh.shape[-1] # vocab size
    v_mask = torch.isin(
        torch.arange(v),
        torch.tensor(new_word_tokens),
        assume_unique=True
    ).to(device) # shape: (vocab_size), automatically broadcasts to (m, nwords, Max_n_tokens + 1, vocab_size)
    
    log_p_new = (batch_log_pxh + v_mask.log()).logsumexp(dim=-1)
    log_p_new_given_x = log_p_new[:, :, seq_lens]
    log_p_new_given_x_proj = (log_p_new_given_x * seq_idxs).sum(-2)

    return log_p_new_given_x_proj


def fast_compute_m_p_words(batch_token_list, batch_log_pxh, 
                         pad_token_id, new_word_tokens, device):
    """ 
    expected dimensions:
    - batch_token_list: bs x max_n_tokens
    - batch_log_pxh: msamples x bs x (max_n_tokens + 1) x |vocabulary|
    - new_word_tokens: list of new word tokens for the model

    output: list of word probabilities
    - final_log_p_x: msamples x bs
    """

    seq_lens = (batch_token_list != pad_token_id).sum(1)
    n = len(batch_token_list)
    m = batch_log_pxh.shape[0]
    max_len = max(seq_lens)
    seq_idxs = torch.eye(batch_log_pxh.shape[1]).to(device)
    log_p_x = torch.zeros((m, n)).to(device)
    for i in range(max_len): # Could be vectorized further too if a bottleneck
        tok_idxs = batch_token_list[:, i]
        tok_pxhs = batch_log_pxh[:, :, i, tok_idxs]
        next_pxhs = (tok_pxhs * seq_idxs).sum(-2)
        log_p_x += next_pxhs * (i < seq_lens)

    if new_word_tokens is not None:
        log_p_new_word = compute_m_log_p_new_word(
            batch_log_pxh, new_word_tokens, seq_idxs, seq_lens, device
        )
        final_log_p_x = log_p_x + log_p_new_word
        return final_log_p_x.exp()
    else:
        return log_p_x.exp()

#########################################
# Single Intervention Functions         #
#########################################
def apply_projection(hs, other_hs, mode, P, I_P):
    """ 
    hs: (m x n x max_ntok + 1 x d)
    other_hs: (m x n x max_ntok + 1 x d)
    P: (d x d)
    I_P: (d x d)

    # rank(I-P)=d-1: project to H_bot
    # rank(P)=1: project to H_par
    """
    assert hs.shape == other_hs.shape, "Incorrect inputs"
    if mode == "hbot":
        newh = hs @ P + other_hs @ I_P
    elif mode == "hpar":
        newh = hs @ I_P + other_hs @ P
    else:
        raise ValueError(f"Incorrect mode {mode}")
    return newh


def sample_other_hs(other_hs, nwords, msamples, device):
    # TODO:
    # Use the sample msamples for each of the n words
    idx = np.random.randint(
        0, other_hs.shape[0], 
        nwords*msamples #nwords x msamples
    )
    other_hs_sample = other_hs[idx].to(device)
    # msamples x nwords x d
    other_hs_view = other_hs_sample.view(
        msamples, nwords, other_hs.shape[1]
    )
    return other_hs_view


def intervene_hs(n_ntok_H, method, msamples, gen_all_hs, P, I_P, device):
    """ input dimensions:
    - n_ntok_H: nwords x (max_ntokens + 1) x d
    output: msamples x nwords x (max_ntokens + 1) x d
    where the first of max_ntokens+1 hs has been intervened upon
    according to method
    return:
    - hs_int: msamples x bs x (max_ntokens + 1) x d
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
