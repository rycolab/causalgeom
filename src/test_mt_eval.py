#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

import re
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
#import torch
import random 
from scipy.special import softmax
#from scipy.stats import entropy
from tqdm import trange
from transformers import TopPLogitsWarper, LogitsProcessorList
import torch 
from torch.utils.data import DataLoader, Dataset
from abc import ABC
from itertools import zip_longest
from scipy.special import softmax
from scipy.stats import entropy
import math

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS


# from evals.mi_distributor_utils import prep_generated_data, \
#    compute_batch_inner_loop_qxhs, get_nucleus_arg, \
#        sample_gen_all_hs_batched, compute_pxh_batch_handler,\
#            fast_compute_p_words
from utils.lm_loaders import SUPPORTED_AR_MODELS
from evals.eval_utils import load_run_Ps, load_run_output, renormalize
from evals.mi_computer_utils import combine_lemma_contexts, compute_all_MIs
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

seed = 10
random.seed(seed)
np.random.seed(seed)



#%%
from evals.MultiTokenDistributor import MultiTokenDistributor, CustomDataset

model_name="gpt2-large"
concept="number"
source="natural" #"gen_nucleus", "gen_ancestral"
nsamples=1
msamples=25
nwords=None # ISSUE: larger nwords might cause low I(X, H_par | C) encapsulation ratio!
n_other_words=500 # 1500
batch_size=128
run_path=os.path.join(
    OUT, 
    "/cluster/work/cotterell/cguerner/usagebasedprobing/out/run_output/number/gpt2-large/leacefinal/run_leace_number_gpt2-large_2024-05-04-12:07:29_0_3.pkl"
)
output_folder = "test"
iteration = 0
#%%
distributor = MultiTokenDistributor(
    model_name,
    concept,
    source,
    nsamples,
    msamples,
    nwords,
    n_other_words,
    run_path,
    output_folder,
    iteration,
    batch_size
)
distributor.compute_all_pxs(exist_ok=True)

all_pxhs = combine_lemma_contexts(
    [np.vstack(x) for x in distributor.l0_cxt_pxhs], 
    [np.vstack(x) for x in distributor.l1_cxt_pxhs]
)
all_qxhbots = combine_lemma_contexts(
    [np.stack(x) for x in distributor.l0_cxt_qxhs_bot], 
    [np.stack(x) for x in distributor.l1_cxt_qxhs_bot]
)
all_qxhpars = combine_lemma_contexts(
    [np.stack(x) for x in distributor.l0_cxt_qxhs_par], 
    [np.stack(x) for x in distributor.l1_cxt_qxhs_par]
)

all_pxhs = [all_pxhs[0], all_pxhs[1], all_pxhs[2]]
all_qxhbots = [all_qxhbots[0], all_qxhbots[1], all_qxhbots[2]]
all_qxhpars = [all_qxhpars[0], all_qxhpars[1], all_qxhpars[2]]

MIs = compute_all_MIs(all_pxhs, all_qxhbots, all_qxhpars)
print(MIs)

ratios = {}
ratios["reconstructed"] = MIs["MIqbot_c_hbot"] + MIs["MIqpar_c_hpar"]
ratios["new_ratio_erasure"] = 1 - (MIs["MIqbot_c_hbot"] / MIs["MIz_c_h"]) # larger is better
ratios["new_ratio_encapsulation"] = MIs["MIqpar_c_hpar"] / MIs["MIz_c_h"] # larger is better
ratios["new_ratio_reconstructed"] = ratios["reconstructed"] / MIs["MIz_c_h"] # larger is better
ratios["new_ratio_containment"] = 1 - (MIs["MIqpar_x_hpar_mid_c"]/MIs["MIz_x_h_mid_c"]) # larger is better
ratios["new_ratio_stability"] = MIs["MIqbot_x_hbot_mid_c"]/MIs["MIz_x_h_mid_c"] # larger is better

print(ratios)

# {'Hz_c': 0.80017054, 'Hz_c_mid_h': 0.4788106, 'MIz_c_h': 0.32135993, 'Hqbot_c': 0.62074685, 'Hqbot_c_mid_hbot': 0.51606864, 'MIqbot_c_hbot': 0.10467821, 'Hqpar_c': 0.52026063, 'Hqpar_c_mid_hpar': 0.45346534, 'MIqpar_c_hpar': 0.06679529, 'Hz_x_c': 3.89076371594107, 'Hz_x_mid_h_c': 2.8216689432102564, 'MIz_x_h_mid_c': 1.0690947727308138, 'Hqbot_x_c': 3.9340193615386227, 'Hqbot_x_mid_hbot_c': 2.9639016797513555, 'MIqbot_x_hbot_mid_c': 0.9701176817872672, 'Hqpar_x_c': 4.586516182751735, 'Hqpar_x_mid_hpar_c': 3.046429980254612, 'MIqpar_x_hpar_mid_c': 1.5400862024971231}
# {'reconstructed': 0.1714735, 'new_ratio_erasure': 0.6742648780345917, 'new_ratio_encapsulation': 0.20785195, 'new_ratio_reconstructed': 0.53358704, 'new_ratio_containment': -0.4405516159837215, 'new_ratio_stability': 0.907419722303265}

# p(EOT | word) effect ~ 0.01 in ratios
# {'Hz_c': 0.8080159, 'Hz_c_mid_h': 0.33986193, 'MIz_c_h': 0.46815395, 'Hqbot_c': 0.94638, 'Hqbot_c_mid_hbot': 0.6941526, 'MIqbot_c_hbot': 0.25222743, 'Hqpar_c': 1.0118899, 'Hqpar_c_mid_hpar': 0.2464175, 'MIqpar_c_hpar': 0.7654724, 'Hz_x_c': 2.3949640456273773, 'Hz_x_mid_h_c': 1.8627211448778556, 'MIz_x_h_mid_c': 0.5322429007495217, 'Hqbot_x_c': 2.970287463706631, 'Hqbot_x_mid_hbot_c': 2.3102124318359962, 'MIqbot_x_hbot_mid_c': 0.660075031870635, 'Hqpar_x_c': 3.601559015936033, 'Hqpar_x_mid_hpar_c': 1.7481376009975726, 'MIqpar_x_hpar_mid_c': 1.8534214149384602}
# {'reconstructed': 1.0176998, 'new_ratio_erasure': 0.46122974157333374, 'new_ratio_encapsulation': 1.6350869, 'new_ratio_reconstructed': 2.1738572, 'new_ratio_containment': -2.4822848972309672, 'new_ratio_stability': 1.2401763009729128}

# {'Hz_c': 0.8080159, 'Hz_c_mid_h': 0.33986193, 'MIz_c_h': 0.46815395, 'Hqbot_c': 0.9515517, 'Hqbot_c_mid_hbot': 0.6943474, 'MIqbot_c_hbot': 0.2572043, 'Hqpar_c': 1.006391, 'Hqpar_c_mid_hpar': 0.24998009, 'MIqpar_c_hpar': 0.75641096, 'Hz_x_c': 2.3949640456273773, 'Hz_x_mid_h_c': 1.8627211448778556, 'MIz_x_h_mid_c': 0.5322429007495217, 'Hqbot_x_c': 2.999229636643495, 'Hqbot_x_mid_hbot_c': 2.337001501541772, 'MIqbot_x_hbot_mid_c': 0.662228135101723, 'Hqpar_x_c': 3.687408873318506, 'Hqpar_x_mid_hpar_c': 1.8103455117207292, 'MIqpar_x_hpar_mid_c': 1.8770633615977768}
# {'reconstructed': 1.0136153, 'new_ratio_erasure': 0.45059889554977417, 'new_ratio_encapsulation': 1.6157312, 'new_ratio_reconstructed': 2.1651323, 'new_ratio_containment': -2.5267043655339227, 'new_ratio_stability': 1.244221640474963}

# l0_inputs, l1_inputs = distributor.sample_filtered_contexts()

# #%%
# #def compute_lemma_probs(self, lemma_samples, method, outdir, pad_token=-1):
# lemma_samples = l0_inputs
# method = "hbot"
# outdir = distributor.outdir
# pad_token = -1 
# # actual function:

# #%%
# l0_probs, l1_probs, other_probs = [], [], []
# #for i, cxt_pad in enumerate(tqdm(lemma_samples)):
# i=0
# cxt_pad = lemma_samples[i]
# #
# cxt = cxt_pad[cxt_pad != pad_token]

# if distributor.prompt_set is not None:
#     cxt = distributor.add_concept_suffix(cxt)

# logging.info(f"---New eval context: {distributor.tokenizer.decode(cxt)}---")

# l0_word_probs = distributor.compute_token_list_word_probs(
#     distributor.l0_tl, cxt, method)
# logging.info("finish l0")
# torch.cuda.empty_cache()
# l1_word_probs = distributor.compute_token_list_word_probs(
#     distributor.l1_tl, cxt, method)
# logging.info("finish l1")
# torch.cuda.empty_cache()
# other_word_probs = distributor.compute_token_list_word_probs(
#     distributor.other_tl, cxt, method)
# logging.info("finish other")
# torch.cuda.empty_cache()

# export_path = os.path.join(
#     outdir, 
#     f"word_probs_sample_{i}.pkl"
# )            
# with open(export_path, 'wb') as f:
#     pickle.dump(
#         (l0_word_probs, l1_word_probs, other_word_probs), f, 
#         protocol=pickle.HIGHEST_PROTOCOL
#     )
            
# l0_probs.append(l0_word_probs)
# l1_probs.append(l1_word_probs)
# other_probs.append(other_word_probs)

# np.stack(l0_probs), np.stack(l1_probs), np.stack(other_probs)

# #%%

# l0_probs, l1_probs = [], []
# #l0_entropies, l1_entropies = [], []
# #for i, cxt_pad in enumerate(tqdm(lemma_samples)):
# i=0
# cxt_pad = lemma_samples[i]

# cxt = cxt_pad[cxt_pad != pad_token]

# if distributor.prompt_set is not None:
#     cxt = distributor.add_concept_suffix(cxt)

# logging.info(f"---New eval context: {distributor.tokenizer.decode(cxt)}---")

# token_list = distributor.l0_tl

# cxt_last_index = cxt.shape[0] - 1
# cxt_np = cxt.clone().cpu().numpy()
# cxt_plus_tl = [np.append(cxt_np, x).tolist() for x in token_list]
# cxt_plus_tl_batched = torch.tensor(
#     list(zip_longest(*cxt_plus_tl, fillvalue=distributor.tokenizer.pad_token_id))
# ).T

# ds = CustomDataset(cxt_plus_tl_batched)
# dl = DataLoader(dataset = ds, batch_size=distributor.batch_size)

# tl_word_probs_hbot = []
# tl_word_probs_hpar = []
# tl_word_probs_h = []
# i, batch_tokens = next(enumerate(dl))

# from evals.mi_distributor_utils import intervene_hs, compute_log_pxh_batch,\
#     fast_compute_m_p_words, fast_compute_p_words

# def compute_word_probs_batch_handler(distributor, method, batch_tok_ids, 
#    batch_hidden_states, gpu_out):
#     if method in ["hbot","hpar"]:
#         hs_int = intervene_hs(
#             batch_hidden_states, method, 
#             distributor.msamples, distributor.gen_all_hs, 
#             distributor.P, distributor.I_P, distributor.device
#         )
#         batch_log_qxhs = compute_log_pxh_batch(
#             hs_int, distributor.V, gpu_out
#         )
#         batch_word_probs = fast_compute_m_p_words(
#             batch_tok_ids, batch_log_qxhs, distributor.tokenizer.pad_token_id, 
#             distributor.new_word_tokens, distributor.device
#         )
#     elif method == "h":
#         batch_log_pxhs = compute_log_pxh_batch(
#             batch_hidden_states, distributor.V, gpu_out
#         )
#         batch_word_probs = fast_compute_p_words(
#             batch_tok_ids, batch_log_pxhs, distributor.tokenizer.pad_token_id, 
#             distributor.new_word_tokens, distributor.device
#         )
#     else:
#         raise ValueError(f"Incorrect method arg")
#     return batch_word_probs

# batch_tokens = batch_tokens.to(distributor.device)
# with torch.no_grad():
#     output = distributor.model(
#         input_ids=batch_tokens, 
#         #attention_mask=attention_mask, 
#         labels=batch_tokens,
#         output_hidden_states=True,
#         # past_key_values=past_key_values
#     )
#     past_key_values = output["past_key_values"]

# batch_tok_ids = batch_tokens[:,(cxt_last_index+1):]
# batch_hidden_states = output["hidden_states"][-1][:,cxt_last_index:,:].type(torch.float32)
# batch_word_probs = compute_word_probs_batch_handler(
#     distributor, method, batch_tok_ids, batch_hidden_states, gpu_out=True
# )

# batch_log_pxhs = distributor.compute_pxh_batch_handler(
#     method,
#     batch_tok_ids,
#     batch_hidden_states,
#     True
# )


# batch_token_list = batch_tok_ids
# batch_log_pxh = batch_log_pxhs
# pad_token_id = distributor.tokenizer.pad_token_id
# new_word_tokens = distributor.new_word_tokens
# device=distributor.device

# seq_lens = (batch_token_list != pad_token_id).sum(1)
# n = len(batch_token_list)
# m = batch_log_pxh.shape[0]
# max_len = max(seq_lens)
# seq_idxs = torch.eye(batch_log_pxh.shape[1]).to(device)
# log_p_x = torch.zeros((m, n)).to(device)
# for i in range(max_len): # Could be vectorized further too if a bottleneck
#     #i = 1
#     print(batch_log_pxh.shape)
#     tok_idxs = batch_token_list[:, i]
#     tok_pxhs = batch_log_pxh[:, :, i, tok_idxs]
#     next_pxhs = (tok_pxhs * seq_idxs).sum(-2)
#     log_p_x += next_pxhs * (i < seq_lens)
#     #p_x *= torch.where((seq_lens > i), tok_pxhs, ones)

# # pnewword
# batch_pxh = batch_log_pxh.exp()
# v = batch_pxh.shape[-1]
# v_mask = torch.isin(
#     torch.arange(v), 
#     torch.tensor(new_word_tokens), 
#     assume_unique=True
# ).to(device)
# p_new = (batch_pxh * v_mask).sum(-1)
# p_new_given_x = p_new[:, :, seq_lens]
# p_new_given_x_proj = (p_new_given_x * seq_idxs).sum(-2)
# final_log_p_x = log_p_x + p_new_given_x_proj.log()


# masked_batch_log_pxh = batch_log_pxh * v_mask
# log_p_new = torch.logsumexp(masked_batch_log_pxh, dim=-1)
# log_p_new_given_x = log_p_new[:, :, seq_lens]
# log_p_new_given_x_proj = (log_p_new_given_x * seq_idxs).sum(-2)
# log_final_log_p_x = log_p_x + log_p_new_given_x_proj
