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
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
#import torch
import time
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


from evals.mi_distributor_utils import prep_generated_data, \
    get_nucleus_arg, get_mt_eval_directory,\
        duplicate_pkv, pad_cxt_list, \
        intervene_first_h, compute_log_pxh_batch, compute_m_p_words,\
            compute_p_words

from utils.lm_loaders import SUPPORTED_AR_MODELS, GPT2_LIST
from evals.eval_utils import load_run_Ps, load_run_output, renormalize
from data.spacy_wordlists.embedder import load_concept_token_lists
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
# Default args not worth putting into command line args
CXT_MAX_LENGTH_PCT = 0.9 # keep only context strings of length less than % of context window
MAX_N_CXTS = 100000 # max number of context strings to store in memory for eval
MAX_N_ALL_HS = 300000 # max number of generated hs to store in memory for eval

#%%
class CustomDataset(Dataset, ABC):
    def __init__(self, token_tensor):
        self.data = token_tensor
        if isinstance(token_tensor, torch.Tensor):
            self.n_instances = token_tensor.shape[0]
        else:
            self.n_instances = len(token_tensor)
        
    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return self.data[index]


#%%
class MultiTokenDistributor:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 source, # ["natural_concept", "gen_ancestral_concept", "gen_nucleus_concept", "gen_ancestral_all", "gen_nucleus_all"]
                 nsamples, # number of test set samples
                 msamples, # number of dev set samples for each q computation
                 nwords, # DEBUG ONLY: number of words to use from token lists
                 n_other_words, # number of other words
                 run_path, # path of LEACE training run output
                 output_folder_name, # directory for exporting individual distributions
                 iteration, # iteration of the eval for this run
                 batch_size, #batch size for the word list probability computation
                 p_new_word=True, # whether to multiply the p(word | h) by p(new_word | h)
                 exist_ok=False, # DEBUG ONLY: export directory exist_ok
                 torch_dtype=torch.float32, # torch data type to use for eval
                ):

        self.model_name = model_name
        self.nsamples = nsamples
        self.msamples = msamples
        self.nwords = nwords
        self.batch_size = batch_size
        self.n_other_words = n_other_words
        self.torch_dtype = torch_dtype

        # Exist ok
        self.exist_ok = exist_ok
        if self.exist_ok:
            logging.warn("DEBUG ONLY: RUNNING WITH EXIST_OK=TRUE")

        nucleus = get_nucleus_arg(source)

        # directory handling
        self.outdir = get_mt_eval_directory(run_path, concept, model_name, 
            output_folder_name, source, iteration)
        os.makedirs(self.outdir, exist_ok=self.exist_ok)
        logging.info(f"Created outdir: {self.outdir}")
        
        # Load run data
        run = load_run_output(run_path)

        # Load model
        self.device = get_device()
        self.model = get_model(model_name, device=self.device)
        self.model.eval()
        self.tokenizer = get_tokenizer(model_name)
        if p_new_word:
            self.new_word_tokens = self.get_new_word_tokens(model_name)
        else:
            self.new_word_tokens = None
            logging.warn("Not computing p(new_word | h)")

        # Load with device
        P, I_P, _ = load_run_Ps(run_path)
        self.P = torch.tensor(P, dtype=self.torch_dtype).to(self.device) 
        self.I_P = torch.tensor(I_P, dtype=self.torch_dtype).to(self.device)

        # Load model eval components
        self.V = get_V(
            model_name, model=self.model, numpy_cpu=False
        ).clone().type(self.torch_dtype).to(self.device)
        self.l0_tl, self.l1_tl, other_tl_full = load_concept_token_lists(
            concept, model_name, single_token=False
        )

        # subsample other words
        random.shuffle(other_tl_full)
        self.other_tl = other_tl_full[:self.n_other_words]

        if self.nwords is not None:
            logging.warn(f"Applied nwords={self.nwords}, intended for DEBUGGING ONLY")
            random_start = random.randint(0, len(self.l0_tl)-self.nwords)
            self.l0_tl = self.l0_tl[random_start:random_start+self.nwords]
            self.l1_tl = self.l1_tl[random_start:random_start+self.nwords]

        # Load generated samples
        
        self.gen_all_hs, self.gen_cxt_toks = prep_generated_data(
            model_name, concept, nucleus, source, self.torch_dtype,
            CXT_MAX_LENGTH_PCT, MAX_N_CXTS, MAX_N_ALL_HS
        )

        # Load test set samples
        self.cxt_toks_test = run["cxt_toks_test"]

        # Select samples to use to compute distributions based on source
        self.cxt_toks = self.get_eval_contexts(source)

        # Delete cxts for memory
        self.gen_cxt_toks = None
        self.cxt_toks_test = None

    #########################################
    # Tokenizer specific new word tokens    #
    #########################################
    def get_gpt2_new_word_tokens(self):
        pattern = re.compile("^[\W][^a-zA-Z]*")

        new_word_tokens = []
        new_word_token_pairs = []
        other_tokens = []
        other_token_pairs = []
        for token, token_id in self.tokenizer.vocab.items():
            if token.startswith("Ġ"):
                new_word_tokens.append(token_id)
                new_word_token_pairs.append((token, token_id))
            elif pattern.match(token):
                new_word_tokens.append(token_id)
                new_word_token_pairs.append((token, token_id))
            else:
                other_tokens.append(token_id)
                other_token_pairs.append((token, token_id))
        return new_word_tokens

    def get_llama2_new_word_tokens(self):
        pattern = re.compile("^[\W][^a-zA-Z]*")

        new_word_tokens = []
        new_word_token_pairs = []
        other_tokens = []
        other_token_pairs = []
        for token, token_id in self.tokenizer.vocab.items():
            if token.startswith("▁"):
                new_word_tokens.append(token_id)
                new_word_token_pairs.append((token, token_id))
            elif pattern.match(token):
                new_word_tokens.append(token_id)
                new_word_token_pairs.append((token, token_id))
            else:
                other_tokens.append(token_id)
                other_token_pairs.append((token, token_id))
        return new_word_tokens

    def get_new_word_tokens(self, model_name):
        if model_name in GPT2_LIST:
            return self.get_gpt2_new_word_tokens()
        elif model_name == "llama2":
            return self.get_llama2_new_word_tokens()
        else:
            return NotImplementedError(f"Model not yet implemented")
    
    #########################################
    # Data handling                         #
    #########################################
    def get_eval_contexts(self, source, max_nsamples=MAX_N_CXTS):
        if source in ["gen_ancestral_concept", "gen_nucleus_concept", 
                      "gen_ancestral_all", "gen_nucleus_all"]:
            padded_cxt_toks = pad_cxt_list(self.gen_cxt_toks, max_nsamples)
            return padded_cxt_toks
        elif source == "natural_concept":
            return torch.from_numpy(self.cxt_toks_test)
        else: 
            raise ValueError(f"Evaluation context source {source} invalid")

    def sample_filtered_contexts(self):
        idx = torch.randperm(self.cxt_toks.shape[0])
        sample_cxt_toks = self.cxt_toks[idx[:self.nsamples]]
        return sample_cxt_toks

    #########################################
    # Probability computations              #
    #########################################
    def compute_qxhs(self,
        cxt_hidden_state, n_ntok_H, method, batch_tokens):
        """ input dimensions:
        - cxt_hidden_state: 1 x d 
        - n_ntok_H: bs x max_ntokens x d
        - batch_tokens: bs x max_n_tokens 
        - new_word_tokens: list of new word tokens for the model

        output: tensor word probabilities
        - batch_word_probs: msamples x bs
        """
        # intervention on first hs
        # shape: (msamples x d)
        first_hs_int = intervene_first_h(
            cxt_hidden_state, method, self.msamples, 
            self.gen_all_hs, self.P, self.I_P, self.device
        )
        
        # shape: msamples x V
        first_log_pxh = compute_log_pxh_batch(first_hs_int, self.V)
        # shape: bs x max_ntokens x V
        next_log_pxh = compute_log_pxh_batch(n_ntok_H, self.V)

        # batch_word_probs: msamples x bs
        batch_word_probs = compute_m_p_words(
            batch_tokens, first_log_pxh, next_log_pxh,
            self.tokenizer.pad_token_id, 
            self.new_word_tokens, self.device
        )
        return batch_word_probs
    
    def compute_pxhs(self, cxt_hidden_state, batch_hidden_states, batch_tokens):
        """ In:
        - batch_tokens: bs x max_ntok
        - cxt_hidden_state: 1 x d
        - batch_hidden_states: bs x max_ntok x d

        Out: word probabilities dim (1, bs)
        """
        # batch_cxt_hidden_state: bs x 1 x d
        batch_cxt_hidden_state = cxt_hidden_state[None, :, :].repeat(
            batch_tokens.shape[0], 1, 1
        )
        # batch_hidden_states: bs x (max_ntok + 1) x d
        concat_batch_hidden_states = torch.cat(
            (batch_cxt_hidden_state, batch_hidden_states), 1
        ).type(self.torch_dtype)
        # bs x (max_ntok + 1) x |V|
        batch_log_pxhs = compute_log_pxh_batch(
            concat_batch_hidden_states, self.V
        )
        # (1, bs)
        batch_word_probs = compute_p_words(
            batch_tokens, batch_log_pxhs, self.tokenizer.pad_token_id, 
            self.new_word_tokens, self.device
        )
        return batch_word_probs

    def compute_pxh_batch_handler(self, method, batch_tokens, 
        cxt_hidden_state, batch_hidden_states):
        """ In:
        - batch_tokens: bs x max_ntok
        - cxt_hidden_state: 1 x d
        - batch_hidden_states: bs x max_ntok x d
        """
        if method in ["hbot", "hpar"]:
            batch_word_probs = self.compute_qxhs(
                cxt_hidden_state, batch_hidden_states, method, 
                batch_tokens
            )
        elif method == "h":
            batch_word_probs = self.compute_pxhs(
                cxt_hidden_state, batch_hidden_states, batch_tokens
            )
        else:
            raise ValueError(f"Incorrect method arg")
        return batch_word_probs
            
    def create_batch_pkv(self, batch_nwords, cxt_pkv, batch_size_pkv):
        if (batch_nwords < self.batch_size or 
            batch_size_pkv is None):
            return duplicate_pkv(cxt_pkv, batch_nwords)
        else:
            return batch_size_pkv

    def compute_batch_p_words(self, batch_tokens, cxt_pkv, 
        batch_size_pkv, cxt_hidden_state, method): 
        batch_tokens = batch_tokens.to(self.device)
        
        dup_pkv = self.create_batch_pkv(
            batch_tokens.shape[0], cxt_pkv, batch_size_pkv
        )

        #with torch.no_grad():
        pkv_output = self.model(
            input_ids=batch_tokens, 
            #attention_mask=attention_mask, 
            labels=batch_tokens,
            output_hidden_states=True,
            past_key_values=dup_pkv
        )

        pkv_batch_hs = pkv_output["hidden_states"][-1]
        batch_word_probs = self.compute_pxh_batch_handler(
            method, batch_tokens, cxt_hidden_state, pkv_batch_hs
        )
        return batch_word_probs

    def compute_token_list_word_probs(self, token_list, cxt_hidden_state, 
        cxt_pkv, method):
        tl_batched = torch.tensor(
            list(zip_longest(*token_list, fillvalue=self.tokenizer.pad_token_id))
        ).T

        ds = CustomDataset(tl_batched)
        dl = DataLoader(dataset=ds, batch_size=self.batch_size)

        if tl_batched.shape[0] > self.batch_size:
            batch_size_pkv = duplicate_pkv(cxt_pkv, self.batch_size)
        else:
            batch_size_pkv = None

        tl_word_probs = []
        for i, batch_tokens in enumerate(dl):

            batch_word_probs = self.compute_batch_p_words(
                batch_tokens, cxt_pkv, batch_size_pkv, 
                cxt_hidden_state, method
            )
            tl_word_probs.append(batch_word_probs)
        
        return torch.hstack(tl_word_probs).cpu().numpy()

    @staticmethod
    def get_cxt_hidden_state(model_name, output_hidden_states):
        """ Process last cxt hidden state depending on model
        - Output dims: (1, d)
        """
        if model_name == "llama2":
            return output_hidden_states[-1][:, -1, :]
        elif model_name in GPT2_LIST:
            return output_hidden_states[-1][-1].unsqueeze(0)
        else: 
            raise NotImplementedError(
                f"Model name {model_name} not implemented"
            )

    def compute_cxt_pkv_h(self, cxt):
        """ Takes a context string and outputs past key values
        and the hidden state corresponding to last token of 
        context string.
        Outputs: 
        - cxt_pkv: nlayers tuple, (([1, d1, d2, d3], []), )
        - cxt_hidden_state: (1, d)
        """
        cxt_tok = cxt.to(self.device)
        cxt_output = self.model(
            input_ids=cxt_tok, 
            #attention_mask=attention_mask, 
            labels=cxt_tok,
            output_hidden_states=True,
            #past_key_values= cxt_pkv
        )
        cxt_pkv = cxt_output.past_key_values

        cxt_hidden_state = self.get_cxt_hidden_state(
            self.model_name, cxt_output["hidden_states"]
        )
        return cxt_pkv, cxt_hidden_state

    def compute_lemma_probs(self, lemma_samples, method, outdir, pad_token=-1):
        l0_probs, l1_probs, other_probs = [], [], []
        for i, cxt_pad in enumerate(tqdm(lemma_samples)):
            cxt = cxt_pad[cxt_pad != pad_token]

            logging.info(f"---New eval context: {self.tokenizer.decode(cxt)}---")

            #start = time.time()
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.torch_dtype, enabled=True):
                    cxt_pkv, cxt_hidden_state = self.compute_cxt_pkv_h(cxt)

                    l0_word_probs = self.compute_token_list_word_probs(self.l0_tl, 
                        cxt_hidden_state, cxt_pkv, method)
                    torch.cuda.empty_cache()
                    l1_word_probs = self.compute_token_list_word_probs(self.l1_tl, 
                        cxt_hidden_state, cxt_pkv, method)
                    torch.cuda.empty_cache()
                    other_word_probs = self.compute_token_list_word_probs(self.other_tl, 
                        cxt_hidden_state, cxt_pkv, method)
                    torch.cuda.empty_cache()
            #end = time.time()
            #pkv_time = end - start

            export_path = os.path.join(
                outdir, 
                f"word_probs_sample_{i}.pkl"
            )            
            with open(export_path, 'wb') as f:
                pickle.dump(
                    (l0_word_probs, l1_word_probs, other_word_probs), f, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
                        
            l0_probs.append(l0_word_probs)
            l1_probs.append(l1_word_probs)
            other_probs.append(other_word_probs)

        return l0_probs, l1_probs, other_probs

    def compute_pxs(self, htype, cxts):
        """ Computes three possible distributions:
        q(x | hbot), q(x | hpar), p(x | h)
        """
        htype_outdir = os.path.join(
            self.outdir, 
            f"h_distribs/{htype}"
        )
        os.makedirs(htype_outdir, exist_ok=self.exist_ok)   

        if htype == "q_x_mid_hpar": 
            return self.compute_lemma_probs(cxts, "hbot", htype_outdir)
        elif htype == "q_x_mid_hbot":
            return self.compute_lemma_probs(cxts, "hpar", htype_outdir)
        elif htype == "p_x_mid_h":
            return self.compute_lemma_probs(cxts, "h", htype_outdir)
        else:
            raise ValueError(f"Incorrect htype: {htype}")

    def compute_all_pxs(self):
        assert self.nsamples is not None
        n_cxts = self.sample_filtered_contexts()

        q_x_mid_hpar = self.compute_pxs("q_x_mid_hpar", n_cxts)
        q_x_mid_hbot = self.compute_pxs("q_x_mid_hbot", n_cxts)
        p_x_mid_h = self.compute_pxs("p_x_mid_h", n_cxts)
        return q_x_mid_hpar, q_x_mid_hbot, p_x_mid_h

