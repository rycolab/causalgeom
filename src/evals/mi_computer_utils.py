#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

from scipy.stats import entropy
import numpy as np

sys.path.append('..')


#%%######################################
# Aggregate over lemma hsamples         #
#########################################
#def combine_lemma_contexts(l0_samples, l1_samples):
#    all_l0_words = np.vstack((l0_samples[0], l1_samples[0]))
#    all_l1_words = np.vstack((l0_samples[1], l1_samples[1]))
#    all_na_words = np.vstack((l0_samples[2], l1_samples[2]))
#    all_contexts = (all_l0_words, all_l1_words, all_na_words)
#    return all_contexts

#%%######################################
# Z(h, c, x) distributions             #
#########################################
def compute_p_c(pxhs):
    """ works for both z and q distributions, i.e.
    in: p(c, h, x)
        (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x l0_xdim, hdim x msamples x l1_xdim, hdim x msamples x other_xdim)
    out p(c): (cdim)
    """
    p_c = np.array([x.sum() for x in pxhs])
    assert np.allclose(p_c.sum(), 1), p_c.sum()
    return p_c

def compute_p_h_c(pxhs):
    """ works for both z and q distributions, i.e.
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x l0_xdim, hdim x msamples x l1_xdim, hdim x msamples x other_xdim)
    out p(h, c): (hdim x cdim)
    """
    if len(pxhs[0].shape) == 2:
        p_h_c = np.array([x.sum(-1) for x in pxhs]).T
    else:
        p_h_c = np.array([x.sum(-1).sum(-1) for x in pxhs]).T

    assert np.allclose(p_h_c.sum(), 1), p_h_c.sum()
    return p_h_c


def compute_p_h(pxhs):
    """ works for both z and q distributions, i.e.
    in: p(c, h, x)
        (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x l0_xdim, hdim x msamples x l1_xdim, hdim x msamples x other_xdim)
    out p(h): (hdim)
    """
    p_h_c = compute_p_h_c(pxhs)
    p_h = p_h_c.sum(1)
    assert np.allclose(p_h.sum(), 1), p_h.sum()
    return p_h


def compute_p_c_mid_h(pxhs):
    """ z distribution only
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)
    out p(c|h): (hdim x cdim)
    """
    # p(c), shape: (cdim)
    p_c = np.array([x.sum() for x in pxhs])
    # p(c, h), shape: (hdim x cdim)
    p_h_c = np.array([x.sum(1) for x in pxhs]).T
    # p(c | h), shape: (hdim x cdim)
    p_c_mid_h = p_h_c / p_h_c.sum(1).reshape(-1,1)

    assert np.allclose(p_c_mid_h.sum(1), 1), p_c_mid_h.sum()
    return p_c_mid_h

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
    # marginalizing over h dim
    p_x_c_l0, p_x_c_l1, p_x_c_other = [x.sum(0) for x in pxhs]
    # p(x|c), shape: (cdim x (l0_nwords + l1_nwords + na_nwords))
    p_x_mid_c = stack_p_x_mid_c(p_x_c_l0, p_x_c_l1, p_x_c_other)
    return p_x_mid_c


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
    p_x_mid_h_c = np.transpose(p_x_mid_h_c, (1, 0, 2))

    return p_x_mid_h_c

def compute_p_x_mid_h_c(pxhs):
    """ z distribution only
    in: (hdim x l0_nwords, hdim x l1_nwords, hdim x na_nwords)
    out p(x | c, h): (hdim x cdim x (l0_nwords + l1_nwords + na_nwords))
    """
    # x.sum(1) = p(h, c)
    p_x_mid_c_l0_h, p_x_mid_c_l1_h, p_x_mid_c_other_h = (
        x / x.sum(1).reshape(-1, 1) for x in pxhs)

    # p(x|c, h), shape: (cdim x hdim x (l0_nwords + l1_nwords + na_nwords))
    p_x_mid_c_h = stack_p_x_mid_h_c(p_x_mid_c_l0_h, p_x_mid_c_l1_h, p_x_mid_c_other_h)
    return p_x_mid_c_h

#%%######################################
# Normalize distributions               #
#########################################
def pxhs_to_p_x_c_h(all_pxhs, weight_na=1.):
    # all_pxhs: tuple(hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)
    # renormalize all_pxhs to:
    # all_pxhs: p(x, c | h), p(h) that is uniform
    p_c0_mid_h = all_pxhs[0].sum(-1) # shape: (hdim)
    p_c1_mid_h = all_pxhs[1].sum(-1) # shape: (hdim)
    p_na_mid_h = 1 - (p_c0_mid_h + p_c1_mid_h)

    # p_c0_mid_h, p_c1_mid_h are computed from full list of c0, c1 words
    # p_na_mid_h is the correct p_na_mid_h. all_pxhs[2].sum(-1) is smaller than p_na_mid_h
    # since it's computed from the subsampled na word list.
    # The sampled na words in all_pxhs should be scaled up to match p_na
    
    na_scale = (p_na_mid_h / all_pxhs[2].sum(-1)).reshape(-1, 1) * weight_na # shape: (hdim, 1)
    # negative probs might appear, clipping them to zero
    # all_pxhs is p(x, c | h)
    p_x_c_mid_h = [all_pxhs[0].clip(min=0), all_pxhs[1].clip(min=0), (all_pxhs[2] * na_scale).clip(min=0)]

    # p_x_c_mid_h[0] stores p(x, c=c0 | h), p_x_c_mid_h[1] stores p(x, c=c1 | h), ...
    h_normalizer = sum([x.sum(-1) for x in p_x_c_mid_h])  # shape: (hdim, 1)

    # Now, we compute p(x, c, h) = p(x, c | h) * p(h),
    # where p(h) = 1 / hdim is uniform, h_normalizer.sum() = hdim
    p_x_c_h = [(x / h_normalizer.sum()) for x in p_x_c_mid_h]
    assert np.allclose(sum([x.sum() for x in p_x_c_h]),
                       1), f"{sum([x.sum() for x in all_pxhs])} should be 1"

    return p_x_c_h


def qxhs_to_q_x_c_h(all_qxhs, weight_na=1.):
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
    p_x_c_mid_h = [all_qxhs[0].clip(min=0), all_qxhs[1].clip(min=0), (all_qxhs[2] * na_scale).clip(min=0)]

    # Now, we compute p(x, c, h) = p(x, c | h) * p(h),
    # where p(h) = 1 / (hdim * m) is uniform, h_normalizer.sum() = hdim * m
    h_normalizer = sum([x.sum(-1) for x in p_x_c_mid_h]) # shape: (hdim, m, 1)

    p_x_c_h = [(x / h_normalizer.sum()) for x in p_x_c_mid_h]
    assert np.allclose(sum([x.sum() for x in p_x_c_h]), 1), f"{sum([x.sum() for x in p_x_c_h])} should be 1"
    
    return p_x_c_h


#%%#####################################################
# Compute all distributions                            #
########################################################
def compute_all_z_distributions(all_pxhs):
    # all_pxhs: tuple(hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)
    all_pxhs = pxhs_to_p_x_c_h(all_pxhs, weight_na=1)

    z_c = compute_p_c(all_pxhs)
    z_h = compute_p_h(all_pxhs)
    z_h_c = compute_p_h_c(all_pxhs)
    z_c_mid_h = compute_p_c_mid_h(all_pxhs)
    z_x_mid_c = compute_p_x_mid_c(all_pxhs)
    z_x_mid_h_c = compute_p_x_mid_h_c(all_pxhs)
    return z_c, z_h, z_h_c, z_c_mid_h, z_x_mid_c, z_x_mid_h_c

def compute_all_q_distributions(all_qxhs):
    # all_qxhs: tuple(hdim x m x l0_xdim, hdim x m x l1_xdim, hdim x m x other_xdim)
    all_qxhs = qxhs_to_q_x_c_h(all_qxhs, weight_na=1)
    all_qxhs = [x.sum(1) for x in all_qxhs]

    q_c = compute_p_c(all_qxhs)
    q_h = compute_p_h(all_qxhs)
    
    q_h_c = compute_p_h_c(all_qxhs)

    q_c_mid_h = compute_p_c_mid_h(all_qxhs)
    q_x_mid_c = compute_p_x_mid_c(all_qxhs)
    q_x_mid_h_c = compute_p_x_mid_h_c(all_qxhs)

    return q_c, q_h, q_h_c, q_c_mid_h, q_x_mid_c, q_x_mid_h_c

#%%#####################################################
# Compute MIs                                          #
########################################################
def compute_I_C_H(p_c, p_c_mid_h, p_h):
    """
    in: 
    - p_c: (cdim)
    - p_c_mid_h: (hdim x cdim)
    out: H(C), H(C|H), I(C;H)
    """
    # H(C)
    H_c = entropy(p_c)
    
    # H(C | H)
    H_c_mid_h = (entropy(p_c_mid_h, axis=1) * p_h).sum()

    MI_c_h = H_c - H_c_mid_h
    return H_c, H_c_mid_h, MI_c_h


def compute_I_X_H_mid_C(p_c, p_x_mid_c, p_c_mid_h, p_x_mid_h_c, p_h_c):
    """ I(X ; H | C)
    in: 
    - p(c): (cdim)
    - p(x | c): (cdim x xdim)
    - p(c | h): (hdim x cdim)
    - p(x | h, c): (hdim x cdim x xdim)
    - p(h, c): (hdim x cdim)
    out: H(X|C), H(X|H, C), I(X;H|C)
    """

    # H(X | H, C)
    H_x_mid_h_c = (entropy(p_x_mid_h_c, axis=2) * p_h_c).sum()

    # H(X | C)
    H_x_mid_c =  (entropy(p_x_mid_c, axis=1) * p_c).sum()
    MI_x_h_mid_c = H_x_mid_c - H_x_mid_h_c

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