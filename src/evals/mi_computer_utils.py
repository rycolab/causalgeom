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

from evals.eval_utils import renormalize


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
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim) OR 
        (hdim x msamples x nwords, hdim x msamples x nwords)
    out p(c): (cdim)
    """
    pchs_l0 = pxhs[0].sum(-1)
    pchs_l1 = pxhs[1].sum(-1)
    p_c0 = pchs_l0.mean()
    p_c1 = pchs_l1.mean()
    p_na = 1 - (p_c0 + p_c1)
    p_c = renormalize([p_c0, p_c1, p_na])
    return p_c

def compute_p_c_mid_h(pxhs):
    """ z distribution only
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x other_xdim)
    out p(c|h): (hdim x cdim)
    """
    all_pchs_l0 = pxhs[0].sum(1)
    all_pchs_l1 = pxhs[1].sum(1)
    all_pchs_unnorm = np.stack((all_pchs_l0, all_pchs_l1))
    all_pchs_na = (1 - all_pchs_unnorm.sum(0)).reshape(1,-1)
    all_pchs = np.concatenate((all_pchs_unnorm, all_pchs_na), 0)
    return all_pchs.T

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
    in: (hdim x l0_xdim, hdim x l1_xdim, hdim x na_xdim)
    out p(x|c): (cdim x (l0_xdim + l1_xdim + na_xdim))
    """
    pxhs_l0, pxhs_l1, pxhs_other =  pxhs[0], pxhs[1], pxhs[2]
    z_x_mid_c0 = renormalize(pxhs_l0.mean(0))
    z_x_mid_c1 = renormalize(pxhs_l1.mean(0))
    z_x_mid_na = renormalize(pxhs_other.mean(0))

    z_x_mid_c = stack_p_x_mid_c(z_x_mid_c0, z_x_mid_c1, z_x_mid_na)
    return z_x_mid_c

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

    #return p_x_mid_h_c.reshape(p_x_mid_h_c.shape[1], p_x_mid_h_c.shape[0], -1)
    return np.transpose(p_x_mid_h_c, (1, 0, 2))

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
    return z_x_mid_h_c

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
    return qchs.T

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

#%%#####################################################
# Compute all distributions                            #
########################################################
def compute_all_z_distributions(all_pxhs):
    z_c = compute_p_c(all_pxhs)
    z_c_mid_h = compute_p_c_mid_h(all_pxhs)
    z_x_mid_c = compute_p_x_mid_c(all_pxhs)
    z_x_mid_h_c = compute_p_x_mid_h_c(all_pxhs)
    return z_c, z_c_mid_h, z_x_mid_c, z_x_mid_h_c

def compute_all_q_distributions(all_qxhs):
    q_c = compute_p_c(all_qxhs)
    q_c_mid_h = compute_q_c_mid_h(all_qxhs)
    q_x_mid_c = compute_q_x_mid_c(all_qxhs)
    q_x_mid_h_c = compute_q_x_mid_h_c(all_qxhs)
    return q_c, q_c_mid_h, q_x_mid_c, q_x_mid_h_c

#%%#####################################################
# Compute MIs                                          #
########################################################
def compute_I_C_H(p_c, p_c_mid_h):
    """
    in: 
    - p_c: (cdim)
    - p_c_mid_h: (hdim x cdim)
    out: H(C), H(C|H), I(C;H)
    """
    # H(C)
    H_c = entropy(p_c)

    # H(C | H)
    H_c_mid_h = entropy(p_c_mid_h,axis=1).mean()

    MI_c_h = H_c - H_c_mid_h
    return H_c, H_c_mid_h, MI_c_h

def compute_H_x_mid_h_c(p_x_mid_h_c, p_c_mid_h):
    """ H(X | H, C)
    Inputs:
    p(x | h, c): h_dim x c_dim x x_dim
    p(c | h): hdim x c_dim
    """
    inner_ent = entropy(p_x_mid_h_c, axis=2)
    outer_ent = (inner_ent * p_c_mid_h).sum(-1)
    H_x_mid_h_c = outer_ent.mean()
    return H_x_mid_h_c

def compute_I_X_H_mid_C(p_c, p_x_mid_c, p_c_mid_h, p_x_mid_h_c):
    """ I(X ; H | C)
    in: 
    - p(c): (cdim)
    - p(x | c): (cdim x xdim)
    - p(c | h): (hdim x c_dim)
    - p(x | h, c): (h_dim x c_dim x x_dim)
    out: H(X|C), H(X|H, C), I(X;H|C)
    """
    # H(X | C)
    H_x_c = p_c @ entropy(p_x_mid_c, axis=1)

    # H(X | H, C)
    H_x_mid_h_c = compute_H_x_mid_h_c(p_x_mid_h_c, p_c_mid_h)

    MI_x_h_mid_c = H_x_c - H_x_mid_h_c
    return H_x_c, H_x_mid_h_c, MI_x_h_mid_c

def compute_all_MIs(all_pxhs, all_qxhbots, all_qxhpars):
    z_c, z_c_mid_h, z_x_mid_c, z_x_mid_h_c = compute_all_z_distributions(all_pxhs)
    qbot_c, qbot_c_mid_hbot, qbot_x_mid_c, qbot_x_mid_hbot_c = compute_all_q_distributions(all_qxhbots)
    qpar_c, qpar_c_mid_hpar, qpar_x_mid_c, qpar_x_mid_hpar_c = compute_all_q_distributions(all_qxhpars)

    # I(C;H)
    Hz_c, Hz_c_mid_h, MIz_c_h = compute_I_C_H(z_c, z_c_mid_h)
    Hqbot_c, Hqbot_c_mid_hbot, MIqbot_c_hbot = compute_I_C_H(
        qbot_c, qbot_c_mid_hbot
    )
    Hqpar_c, Hqpar_c_mid_hpar, MIqpar_c_hpar = compute_I_C_H(
        qpar_c, qpar_c_mid_hpar
    )

    # I(X;H|C)
    Hz_x_c, Hz_x_mid_h_c, MIz_x_h_mid_c = compute_I_X_H_mid_C(
        z_c, z_x_mid_c, z_c_mid_h, z_x_mid_h_c
    )
    Hqbot_x_c, Hqbot_x_mid_hbot_c, MIqbot_x_hbot_mid_c = compute_I_X_H_mid_C(
        qbot_c, qbot_x_mid_c, qbot_c_mid_hbot, qbot_x_mid_hbot_c
    )
    Hqpar_x_c, Hqpar_x_mid_hpar_c, MIqpar_x_hpar_mid_c = compute_I_X_H_mid_C(
        qpar_c, qpar_x_mid_c, qpar_c_mid_hpar, qpar_x_mid_hpar_c
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