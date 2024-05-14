#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

from scipy.stats import entropy
from scipy.special import kl_div
import numpy as np

sys.path.append('..')

from evals.eval_utils import renormalize


#%%######################################
# Aggregate over lemma hsamples         #
#########################################
def combine_lemma_contexts(l0_samples, l1_samples):
    all_l0_words = np.vstack((l0_samples[0], l1_samples[0]))
    all_l1_words = np.vstack((l0_samples[1], l1_samples[1]))
    all_na_words = np.vstack((l0_samples[2], l1_samples[2]))
    all_contexts = (all_l0_words, all_l1_words, all_na_words)
    return all_contexts

#%%######################################
# Z(h, c, x) distributions             #
#########################################
def compute_p_c(pxhs):
    """ works for both z and q distributions, i.e.
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x nwords, hdim x msamples x nwords)
    out p(c): (cdim)
    """
    # p(x, c, h), shape: (hdim x l0_n_words), (hdim x l1_n_words), (hdim x na_n_words)
    p_c_new = np.array([x.sum() for x in pxhs])
    print("p_c_new", p_c_new)
    assert np.allclose(p_c_new.sum(), 1), p_c_new.sum()
    return p_c_new


def compute_p_h(pxhs):
    """ works for both z and q distributions, i.e.
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x nwords, hdim x msamples x nwords)
    out p(h): (hdim)
    """
    # we are going to renormalize pxhs to p(x, c, h)
    # so they sum to one
    # p(x, c, h)
    # p(h, c), shape: (hdim x cdim)
    if len(pxhs[0].shape) == 2:
        p_h_c = np.array([x.sum(-1) for x in pxhs]).T
    else:
        p_h_c = np.array([x.sum(-1).sum(-1) for x in pxhs]).T

    p_h = p_h_c.sum(1)
    assert np.allclose(p_h.sum(), 1), p_h.sum()
    return p_h


def compute_p_h_c(pxhs):
    """ works for both z and q distributions, i.e.
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x nwords, hdim x msamples x nwords)
    out p(h, c): (hdim x cdim)
    """
    # p(h, c), shape: (hdim x cdim) OR (hdim x msamples x cdim)
    if len(pxhs[0].shape) == 2:
        p_h_c = np.array([x.sum(-1) for x in pxhs]).T
    else:
        p_h_c = np.array([x.sum(-1).sum(-1) for x in pxhs]).T

    assert np.allclose(p_h_c.sum(), 1), p_h_c.sum()
    return p_h_c


def compute_p_c_mid_h(pxhs):
    """ z distribution only
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)
    out p(c|h): (hdim x cdim)
    """
    all_pchs_l0 = pxhs[0].sum(1)
    all_pchs_l1 = pxhs[1].sum(1)
    all_pchs_unnorm = np.stack((all_pchs_l0, all_pchs_l1)) # shape: (2, hdim)
    all_pchs_na = (1 - all_pchs_unnorm.sum(0)).reshape(1,-1) # shape: (1, hdim)
    all_pchs = np.concatenate((all_pchs_unnorm, all_pchs_na), 0) # shape: (3, hdim)

    normalizer_z = sum([x.sum() for x in pxhs]) # shape: ()
    # we are going to renormalize pxhs to p(x, c, h)
    # so they sum to one
    # p(x, c, h), shape: (hdim x l0_n_words), (hdim x l1_n_words), (hdim x na_n_words)
    p_x_c_h_l0, p_x_c_h_l1, p_x_c_h_other = (x / normalizer_z for x in pxhs)
    p_c_new = np.array([p_x_c_h_l0.sum(), p_x_c_h_l1.sum(), p_x_c_h_other.sum()])
    # p(c, h), shape: (hdim x cdim)
    p_h_c_new = np.array([x.sum(1) for x in (p_x_c_h_l0, p_x_c_h_l1, p_x_c_h_other)]).T
    # p(c | h), shape: (hdim x cdim)
    p_c_mid_h_new = p_h_c_new / p_h_c_new.sum(1).reshape(-1,1)

    assert np.allclose(p_c_mid_h_new.sum(1), 1), p_c_mid_h_new.sum()
    return p_c_mid_h_new # all_pchs.T

def stack_p_x_mid_c(p_x_mid_c0, p_x_mid_c1, p_x_mid_na):
    """ Stacks three p(x|c) distributions
    Inputs: (l0_nwords,), (l1_nwords,), (na_nwords,)
    Output: (cdim x (l0_nwords + l1_nwords + na_nwords))
    """
    n_c0_words = p_x_mid_c0.shape[0]
    n_c1_words = p_x_mid_c1.shape[0]
    n_na_words = p_x_mid_na.shape[0]
    total_words = (
        n_c0_words + n_c1_words + n_na_words
    )
    p_x_mid_c = np.zeros((3, total_words))

    # 0:n_c0_words 
    c0_end_index = n_c0_words
    p_x_mid_c[0,:c0_end_index] = p_x_mid_c0
    # n_c0_words:(n_c0_words + n_c1_words)
    c1_end_index = (n_c0_words + n_c1_words)
    p_x_mid_c[1, n_c0_words:c1_end_index] = p_x_mid_c1
    # (n_c0_words + n_c1_words):
    p_x_mid_c[2, c1_end_index:] = p_x_mid_na
    return p_x_mid_c

def compute_p_x_mid_c(pxhs):
    """ z distribution only
    in: (nsamples x l0_nwords, nsamples x l1_nwords, nsamples x na_nwords)
    out p(x|c): (cdim x (l0_nwords + l1_nwords + na_nwords)) cdim=3, 
    """
    pxhs_l0, pxhs_l1, pxhs_other =  pxhs[0], pxhs[1], pxhs[2]
    z_x_mid_c0 = renormalize(pxhs_l0.mean(0)) # shape: (l0_nwords, ) distribution
    z_x_mid_c1 = renormalize(pxhs_l1.mean(0))
    z_x_mid_na = renormalize(pxhs_other.mean(0))
    z_x_mid_c = stack_p_x_mid_c(z_x_mid_c0, z_x_mid_c1, z_x_mid_na)

    normalizer_z = sum([x.sum() for x in pxhs]) # shape: ()
    # we are going to renormalize pxhs to p(x, c, h)
    # so they sum to one
    # p(x, c, h), shape: (hdim x l0_n_words), (hdim x l1_n_words), (hdim x na_n_words)
    p_x_c_h_l0, p_x_c_h_l1, p_x_c_h_other = (x / normalizer_z for x in pxhs)
    p_x_c_l0, p_x_c_l1, p_x_c_other = (
        p_x_c_h_l0.sum(0) / p_x_c_h_l0.sum(),
        p_x_c_h_l1.sum(0) / p_x_c_h_l1.sum(),
        p_x_c_h_other.sum(0) / p_x_c_h_other.sum()
    )
    # p(x|c), shape: (cdim x (l0_nwords + l1_nwords + na_nwords))
    p_x_mid_c_new = stack_p_x_mid_c(p_x_c_l0, p_x_c_l1, p_x_c_other)
    # assert np.allclose(z_x_mid_c, p_x_mid_c_new)
    return p_x_mid_c_new # z_x_mid_c

def stack_p_x_mid_h_c(p_x_mid_h_c0, p_x_mid_h_c1, p_x_mid_h_na):
    n_c0_words = p_x_mid_h_c0.shape[1]
    n_c1_words = p_x_mid_h_c1.shape[1]
    n_na_words = p_x_mid_h_na.shape[1]
    n_hs = p_x_mid_h_c0.shape[0]
    total_words = (
        n_c0_words + n_c1_words + n_na_words
    )
    p_x_mid_h_c = np.zeros((3, n_hs, total_words))

    c0_end_index = n_c0_words
    p_x_mid_h_c[0,:,:c0_end_index] = p_x_mid_h_c0

    c1_end_index = (n_c0_words + n_c1_words)
    p_x_mid_h_c[1, :, n_c0_words:c1_end_index] = p_x_mid_h_c1
    p_x_mid_h_c[2, :, c1_end_index:] = p_x_mid_h_na

    # ISSUE 1 FIXED: reshape -> transpose
    old = p_x_mid_h_c.reshape(p_x_mid_h_c.shape[1], p_x_mid_h_c.shape[0], -1)
    new = np.transpose(p_x_mid_h_c, (1, 0, 2))
    return new

def compute_p_x_mid_h_c(pxhs):
    """ z distribution only
    in: (hdim x l0_nwords, hdim x l1_nwords, hdim x na_nwords)
    out p(x | c, h): (hdim x cdim x (l0_nwords + l1_nwords + na_nwords))
    """
    pxhs_l0, pxhs_l1, pxhs_other =  pxhs[0], pxhs[1], pxhs[2]
    z_x_mid_h_c0 = (pxhs_l0.T / pxhs_l0.sum(1)).T
    z_x_mid_h_c1 = (pxhs_l1.T / pxhs_l1.sum(1)).T
    z_x_mid_h_na = (pxhs_other.T / pxhs_other.sum(1)).T

    z_x_mid_h_c = stack_p_x_mid_h_c(z_x_mid_h_c0, z_x_mid_h_c1, z_x_mid_h_na)

    normalizer_z = sum([x.sum() for x in pxhs]) # shape: ()
    # we are going to renormalize pxhs to p(x, c, h)
    # so they sum to one
    # p(x, c, h), shape: (hdim x l0_n_words), (hdim x l1_n_words), (hdim x na_n_words)
    p_x_c_h_l0, p_x_c_h_l1, p_x_c_h_other = (x / normalizer_z for x in pxhs)
    p_x_mid_c_l0_h, p_x_mid_c_l1_h, p_x_mid_c_other_h = (
        p_x_c_h_l0 / p_x_c_h_l0.sum(1).reshape(-1,1), 
        p_x_c_h_l1 / p_x_c_h_l1.sum(1).reshape(-1,1),
        p_x_c_h_other / p_x_c_h_other.sum(1).reshape(-1,1)
    )
    # p(x|c), shape: (cdim x (l0_nwords + l1_nwords + na_nwords))
    p_x_mid_c_h_new = stack_p_x_mid_h_c(p_x_mid_c_l0_h, p_x_mid_c_l1_h, p_x_mid_c_other_h)
    # assert np.allclose(z_x_mid_h_c, p_x_mid_c_h_new)
    return p_x_mid_c_h_new # z_x_mid_h_c

#%%#####################################################
# qbot/par(hbot, hpar, c, x) distributions             #
########################################################
def compute_q_c_mid_h(qxhs):
    """ 
    in: (hdim x msamples x l0_nwords, 
         hdim x msamples x l1_nwords, 
         hdim x msamples x na_nwords)
    out q(c|hbot\par): hdim x cdim
    """
    qchs_l0 = qxhs[0].sum(-1).mean(-1)
    qchs_l1 = qxhs[1].sum(-1).mean(-1)
    qchs_unnorm = np.stack((qchs_l0, qchs_l1))
    #qchs = qchs_unnorm / qchs_unnorm.sum(0)

    qchs_na = (1 - qchs_unnorm.sum(0)).reshape(1,-1)
    qchs = np.concatenate((qchs_unnorm, qchs_na), 0)

    normalizer_z = sum([x.sum() for x in qxhs]) # shape: ()
    # we are going to renormalize pxhs to p(x, c, h)
    # so they sum to one
    # p(x, c, h), shape: (hdim x m x l0_n_words), (hdim x m x l1_n_words), (hdim x m x na_n_words)
    p_x_c_h_l0, p_x_c_h_l1, p_x_c_h_other = (x / normalizer_z for x in qxhs)
    p_c_new = np.array([p_x_c_h_l0.sum(), p_x_c_h_l1.sum(), p_x_c_h_other.sum()])
    # p(c, h), shape: (hdim x cdim)
    p_h_c_new = np.array([x.sum(-1).sum(-1) for x in (p_x_c_h_l0, p_x_c_h_l1, p_x_c_h_other)]).T
    # p(c | h), shape: (hdim x cdim)
    p_c_mid_h_new = p_h_c_new / p_h_c_new.sum(1).reshape(-1,1)

    assert np.allclose(p_c_mid_h_new.sum(1), 1), p_c_mid_h_new.sum()
    return p_c_mid_h_new # p_c_mid_h_new # qchs.T

def compute_q_x_mid_c(qxhs):
    """ 
    in: (hdim x msamples x l0_nwords, 
         hdim x msamples x l1_nwords, 
         hdim x msamples x na_nwords)
    out q(x|c): cdim x (l0_nwords + l1_nwords + na_nwords)
    """
    q_x_mid_c0_unnorm = qxhs[0].mean(0).mean(0)
    q_x_mid_c0 = q_x_mid_c0_unnorm / q_x_mid_c0_unnorm.sum()
    q_x_mid_c1_unnorm = qxhs[1].mean(0).mean(0)
    q_x_mid_c1 = q_x_mid_c1_unnorm / q_x_mid_c1_unnorm.sum()
    q_x_mid_na_unnorm = qxhs[2].mean(0).mean(0)
    q_x_mid_na = q_x_mid_na_unnorm / q_x_mid_na_unnorm.sum()

    q_x_mid_c = stack_p_x_mid_c(q_x_mid_c0, q_x_mid_c1, q_x_mid_na)
    return q_x_mid_c

def compute_q_x_mid_h_c(qxhs):
    """ 
    in: (hdim x msamples x l0_nwords, 
         hdim x msamples x l1_nwords, 
         hdim x msamples x na_nwords)
    out q(x|hbot/hpar,c): hdim x cdim x (l0_nwords + l1_nwords + na_nwords)
    """
    q_x_mid_h_c0_unnorm = qxhs[0].mean(1)
    q_x_mid_h_c0 = (q_x_mid_h_c0_unnorm.T / q_x_mid_h_c0_unnorm.sum(1)).T
    q_x_mid_h_c1_unnorm = qxhs[1].mean(1)
    q_x_mid_h_c1 = (q_x_mid_h_c1_unnorm.T / q_x_mid_h_c1_unnorm.sum(1)).T
    q_x_mid_h_na_unnorm = qxhs[2].mean(1)
    q_x_mid_h_na = (q_x_mid_h_na_unnorm.T / q_x_mid_h_na_unnorm.sum(1)).T

    q_x_mid_h_c = stack_p_x_mid_h_c(q_x_mid_h_c0, q_x_mid_h_c1, q_x_mid_h_na)
    return q_x_mid_h_c

def pxhs_to_p_x_c_mid_h(all_pxhs, weight_na=1.):
    # all_pxhs: tuple(hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)

    # renormalize all_pxhs to:
    # all_pxhs: p(x, c | h)
    p_c0_mid_h = all_pxhs[0].sum(-1) # shape: (hdim)
    p_c1_mid_h = all_pxhs[1].sum(-1) # shape: (hdim)
    p_na_mid_h = 1 - (p_c0_mid_h + p_c1_mid_h)

    # p_c0_mid_h, p_c1_mid_h are computed from full list of c0, c1 words
    # p_na_mid_h is the correct p_na_mid_h. all_pxhs[2].sum(-1) is smaller than p_na_mid_h
    # since it's computed from the subsampled na word list.
    # The sampled na words in all_pxhs should be scaled up to match p_na
    
    na_scale = (p_na_mid_h / all_pxhs[2].sum(-1)).reshape(-1, 1) * weight_na # shape: (hdim, 1)
    all_pxhs = [all_pxhs[0].clip(min=0), all_pxhs[1].clip(min=0), (all_pxhs[2] * na_scale).clip(min=0)]

    h_normalizer = sum([x.sum(-1) for x in all_pxhs]) # shape: (hdim, 1)
    # assert np.allclose(h_normalizer, 1), f"{h_normalizer} should be all 1"
    all_pxhs = [(x / h_normalizer.sum()) for x in all_pxhs]
    assert np.allclose(sum([x.sum() for x in all_pxhs]), 1), f"{sum([x.sum() for x in all_pxhs])} should be 1"
    
    return all_pxhs


def pxhs_to_q_x_c_mid_h(all_qxhs, weight_na=1.):
    # all_qxhs: tuple(hdim x m x l0_xdim, hdim x m x l1_xdim, hdim x m x other_xdim)
    # renormalize all_pxhs to:
    # all_qxhs: p(x, c | h, m)
    p_c0_mid_h = all_qxhs[0].sum(-1) # shape: (hdim, m)
    p_c1_mid_h = all_qxhs[1].sum(-1) # shape: (hdim, m)
    p_na_mid_h = 1 - (p_c0_mid_h + p_c1_mid_h)

    # p_c0_mid_h, p_c1_mid_h are computed from full list of c0, c1 words
    # p_na_mid_h is the correct p_na_mid_h. all_pxhs[2].sum(-1) is smaller than p_na_mid_h
    # since it's computed from the subsampled na word list.
    # The sampled na words in all_pxhs should be scaled up to match p_na
    
    na_scale = np.expand_dims(p_na_mid_h / all_qxhs[2].sum(-1), 2) * weight_na # shape: (hdim, m, 1)
    all_qxhs = [all_qxhs[0].clip(min=0), all_qxhs[1].clip(min=0), (all_qxhs[2] * na_scale).clip(min=0)]

    h_normalizer = sum([x.sum(-1) for x in all_qxhs]) # shape: (hdim, 1)
    # assert np.allclose(h_normalizer, 1), f"{h_normalizer} should be all 1"
    all_qxhs = [(x / h_normalizer.sum()) for x in all_qxhs]
    assert np.allclose(sum([x.sum() for x in all_qxhs]), 1), f"{sum([x.sum() for x in all_qxhs])} should be 1"
    
    return all_qxhs


#%%#####################################################
# Compute all distributions                            #
########################################################
def compute_all_z_distributions(all_pxhs):
    # all_pxhs: tuple(hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)
    all_pxhs = pxhs_to_p_x_c_mid_h(all_pxhs, weight_na=1)

    z_c = compute_p_c(all_pxhs)
    z_h = compute_p_h(all_pxhs)
    z_h_c = compute_p_h_c(all_pxhs)
    z_c_mid_h = compute_p_c_mid_h(all_pxhs)
    z_x_mid_c = compute_p_x_mid_c(all_pxhs)
    z_x_mid_h_c = compute_p_x_mid_h_c(all_pxhs)
    return z_c, z_h, z_h_c, z_c_mid_h, z_x_mid_c, z_x_mid_h_c

def compute_all_q_distributions(all_qxhs):
    # all_qxhs: tuple(hdim x m x l0_xdim, hdim x m x l1_xdim, hdim x m x other_xdim)
    all_qxhs = pxhs_to_q_x_c_mid_h(all_qxhs, weight_na=1)

    q_c = compute_p_c(all_qxhs)
    q_h = compute_p_h(all_qxhs)
    
    q_h_c = compute_p_h_c(all_qxhs)

    q_c_mid_h = compute_q_c_mid_h(all_qxhs)
    q_x_mid_c = compute_q_x_mid_c(all_qxhs)
    q_x_mid_h_c = compute_q_x_mid_h_c(all_qxhs)
    return q_c, q_h, q_h_c, q_c_mid_h, q_x_mid_c, q_x_mid_h_c

#%%#####################################################
# Compute MIs                                          #
########################################################
def compute_I_C_H(p_c, p_c_mid_h, p_h=None):
    """
    in: 
    - p_c: (cdim)
    - p_c_mid_h: (hdim x cdim)
    out: H(C), H(C|H), I(C;H)
    """
    # H(C)
    H_c = entropy(p_c)

    # H(C | H)
    if p_h is None:
        H_c_mid_h = entropy(p_c_mid_h,axis=1).mean()
    else:
        H_c_mid_h = (entropy(p_c_mid_h, axis=1) * p_h).sum()
        # assert np.allclose(p_h, 1/p_h.shape[0]), p_h

    MI_c_h = H_c - H_c_mid_h
    assert MI_c_h >= 0, (H_c, H_c_mid_h)
    print("MI_c_h", MI_c_h, H_c, H_c_mid_h)
    return H_c, H_c_mid_h, MI_c_h

def compute_H_x_mid_h_c(p_x_mid_h_c, p_c_mid_h):
    """ H(X | H, C)
    Inputs:
    p(x | h, c): n_samples x c_dim x x_dim
    p(c | h): n_samples x c_dim

    H(X | H, C) = E_{h,c} [H(X | h, c)]
    """
    H_X_mid_h_c = entropy(p_x_mid_h_c, axis=2) # n_samples x c_dim
    H_X_mid_h_C = (H_X_mid_h_c * p_c_mid_h).sum(-1) # n_samples
    H_X_mid_H_C = H_X_mid_h_C.mean()

    H_X_mid_H_C_old = (H_X_mid_h_c * p_c_mid_h).mean()

    return H_X_mid_H_C


def compute_I_X_H_mid_C(p_c, p_x_mid_c, p_c_mid_h, p_x_mid_h_c, p_h_c):
    """ I(X ; H | C)
    in: 
    - p(c): (cdim)
    - p(x | c): (cdim x xdim)
    - p(c | h): (hdim x cdim)
    - p(x | h, c): (hdim x cdim x xdim)
    - p(h, c): (hdim x cdim)
    out: H(X|C), H(X|H, C), I(X;H|C)
    ISSUE: I(X;H|C) too small: MIz_x_h_mid_c
    """

    # H(X | H, C)
    H_x_mid_h_c = (entropy(p_x_mid_h_c, axis=2) * p_h_c).sum()
    # p(c,h)
    # assert np.allclose((p_x_mid_h_c*np.expand_dims(p_h_c, 2)).sum(2), p_h_c)
    # assert np.allclose((p_x_mid_c*np.expand_dims(p_c, 1)).sum(1), p_c), (p_x_mid_c.sum(1), p_c)

    a = (p_x_mid_c * np.expand_dims(p_c, 1)).sum(0)
    b = (p_x_mid_h_c * np.expand_dims(p_h_c, 2)).sum(0).sum(0)
    # assert np.allclose(a, b), np.linalg.norm(a-b)

    # H(X | C)
    H_x_mid_c = p_c @ entropy(p_x_mid_c, axis=1)
    MI_x_h_mid_c = H_x_mid_c - H_x_mid_h_c
    print("MI_x_h_mid_c", MI_x_h_mid_c, H_x_mid_c, H_x_mid_h_c)

    # MI_x_h_mid_c  I(X, H_bot | C) = E_C [KL(p(x,h | c) || p(x | c) p(h | c))]
    p_h_mid_c = p_h_c / p_h_c.sum(0).reshape(1, -1) # shape: (hdim x cdim)
    p_x_h_mid_c = p_x_mid_h_c * np.expand_dims(p_h_mid_c, 2) # shape: (hdim x cdim x xdim)
    
    # kl = (p_x_h_mid_c * np.log((p_x_h_mid_c+1e-8) / (np.expand_dims(p_x_mid_c, 0) * np.expand_dims(p_h_mid_c, 2)+1e-8))).sum(2).sum(0) # shape: (cdim)
    # print("MI_x_h_mid_c, p_c @ kl", MI_x_h_mid_c, p_c @ kl)
    # print(kl)

    # ISSUE 1: Most kl mass is in na concept
    # ISSUE 2: I(X, H_bot | C) and I(X, H_par | C)  too large
    # MIqpar_x_hpar_mid_c kl too large!

    return H_x_mid_c, H_x_mid_h_c, MI_x_h_mid_c

def compute_all_MIs(all_pxhs, all_qxhbots, all_qxhpars):
    z_c, z_h, z_h_c, z_c_mid_h, z_x_mid_c, z_x_mid_h_c = compute_all_z_distributions(all_pxhs)
    qbot_c, qbot_h, qbot_c_hbot, qbot_c_mid_hbot, qbot_x_mid_c, qbot_x_mid_hbot_c = compute_all_q_distributions(all_qxhbots)
    qpar_c, qpar_h, qpar_c_hpar, qpar_c_mid_hpar, qpar_x_mid_c, qpar_x_mid_hpar_c = compute_all_q_distributions(all_qxhpars)

    # I(C;H)
    print("estimating I(C, H) MIz_c_h")
    Hz_c, Hz_c_mid_h, MIz_c_h = compute_I_C_H(
        z_c, z_c_mid_h, p_h=z_h
    )
    print("estimating I(C, H_bot) MIqbot_c_hbot")
    Hqbot_c, Hqbot_c_mid_hbot, MIqbot_c_hbot = compute_I_C_H(
        qbot_c, qbot_c_mid_hbot, p_h=qbot_h
    )
    print("estimating I(C, H_par) MIqpar_c_hpar")
    Hqpar_c, Hqpar_c_mid_hpar, MIqpar_c_hpar = compute_I_C_H(
        qpar_c, qpar_c_mid_hpar, p_h=qpar_h
    )

    # I(X;H|C)
    print("estimating I(X, H | C) MIz_x_h_mid_c")
    Hz_x_c, Hz_x_mid_h_c, MIz_x_h_mid_c = compute_I_X_H_mid_C(
        z_c, z_x_mid_c, z_c_mid_h, z_x_mid_h_c, z_h_c
    )
    print("estimating I(X, H_bot | C) MIqbot_x_hbot_mid_c")
    Hqbot_x_c, Hqbot_x_mid_hbot_c, MIqbot_x_hbot_mid_c = compute_I_X_H_mid_C(
        qbot_c, qbot_x_mid_c, qbot_c_mid_hbot, qbot_x_mid_hbot_c, qbot_c_hbot
    )
    print("estimating I(X, H_par | C) MIqpar_x_hpar_mid_c")
    Hqpar_x_c, Hqpar_x_mid_hpar_c, MIqpar_x_hpar_mid_c = compute_I_X_H_mid_C(
        qpar_c, qpar_x_mid_c, qpar_c_mid_hpar, qpar_x_mid_hpar_c, qpar_c_hpar
    )

    assert MIz_x_h_mid_c >= 0, MIz_x_h_mid_c
    assert MIqbot_x_hbot_mid_c >= 0, MIqbot_x_hbot_mid_c
    assert MIqpar_x_hpar_mid_c >= 0, MIqpar_x_hpar_mid_c

    res = {
        "Hz_c":Hz_c, 
        "Hz_c_mid_h":Hz_c_mid_h, 
        "MIz_c_h":MIz_c_h, 
        "Hqbot_c":Hqbot_c, 
        "Hqbot_c_mid_hbot":Hqbot_c_mid_hbot, 
        "MIqbot_c_hbot":MIqbot_c_hbot, 
        "Hqpar_c":Hqpar_c, 
        "Hqpar_c_mid_hpar":Hqpar_c_mid_hpar, 
        "MIqpar_c_hpar":MIqpar_c_hpar, 
        "Hz_x_c":Hz_x_c, 
        "Hz_x_mid_h_c":Hz_x_mid_h_c, 
        "MIz_x_h_mid_c":MIz_x_h_mid_c, 
        "Hqbot_x_c":Hqbot_x_c, 
        "Hqbot_x_mid_hbot_c":Hqbot_x_mid_hbot_c, 
        "MIqbot_x_hbot_mid_c":MIqbot_x_hbot_mid_c, 
        "Hqpar_x_c":Hqpar_x_c, 
        "Hqpar_x_mid_hpar_c":Hqpar_x_mid_hpar_c, 
        "MIqpar_x_hpar_mid_c":MIqpar_x_hpar_mid_c,
    }
    return res