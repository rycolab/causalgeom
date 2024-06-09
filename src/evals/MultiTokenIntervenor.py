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


from evals.mi_distributor_utils import get_nucleus_arg, \
    get_eval_directory, duplicate_pkv, pad_cxt_list, \
        intervene_first_h, compute_log_pxh_batch, compute_m_p_words,\
            compute_p_words, filter_cxt_toks_by_length

from utils.lm_loaders import SUPPORTED_AR_MODELS, GPT2_LIST
from evals.mi_intervenor_utils import prep_int_generated_data
from evals.eval_utils import load_run_Ps, load_run_output, renormalize
from data.spacy_wordlists.embedder import load_concept_token_lists
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
# Default args not worth putting into command line args
CXT_MAX_LENGTH_PCT = 0.90 # keep only context strings of length less than % of context window
MAX_N_CXTS = 1000 # max number of context strings to store in memory for eval
MAX_N_ALL_HS = 400000 # max number of generated hs to store in memory for eval

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
class MultiTokenIntervenor:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 int_source, # ["train_all", "train_concept", "test_all", "test_concept", "gen_ancestral_concept", "gen_nucleus_concept", "gen_ancestral_all", "gen_nucleus_all"]
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

        # Load run data
        run = load_run_output(run_path)
        self.proj_source = run["proj_source"]
        assert run["config"]['model_name'] == model_name, "Run model doesn't match"
        assert run["config"]['concept'] == concept, "Run concept doesn't match"

        # directory handling
        self.outdir = get_eval_directory(
            "int_eval", run_path, concept, model_name, 
            self.proj_source, output_folder_name, 
            int_source, iteration
        )
        os.makedirs(self.outdir, exist_ok=self.exist_ok)
        logging.info(f"Created outdir: {self.outdir}")

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
        #TODO: this is a bit messy, figure it out
        if int_source in ["train_all", "train_concept", "test_all", "test_concept"]:
            nucleus = get_nucleus_arg(self.proj_source)
        elif int_source in ["gen_ancestral_all", "gen_ancestral_concept", "gen_nucleus_all", "gen_nucleus_concept"]:
            nucleus = get_nucleus_arg(int_source)
        else:
            raise NotImplementedError(f"int_source {int_source} not supported")
        self.gen_all_hs = prep_int_generated_data(
            model_name, concept, nucleus, int_source, self.torch_dtype,
            CXT_MAX_LENGTH_PCT, MAX_N_CXTS, MAX_N_ALL_HS
        )

        # Load test set samples
        #self.cxt_toks_train, self.y_train = run["cxt_toks_train"], run["y_train"]
        self.cxt_toks_test, self.y_test = run["cxt_toks_test"], run["y_test"]
        self.facts_test, self.foils_test = run["facts_test"], run["foils_test"]

        # Load val set hs
        hs_val, y_val = run["X_val"], run["y_val"]
        self.l0_hs = torch.tensor(hs_val[y_val == 0], dtype=self.torch_dtype)
        self.l1_hs = torch.tensor(hs_val[y_val == 1], dtype=self.torch_dtype)

        # Select samples to use to compute distributions based on eval_source
        #self.cxt_toks = self.get_eval_contexts(model_name, eval_source)

        # Delete cxts for memory
        #self.gen_cxt_toks = None
        #self.cxt_toks_train = None
        #self.cxt_toks_test = None
        #self.y_train = None
        #self.y_test = None

    #########################################
    # Tokenizer specific new word tokens    #
    #########################################
    #TODO: ALL OF THIS SHOULD BE MOVED TO UTILS SAME AS MT DISTRIB
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
    def get_eval_contexts(self, model_name, eval_source, max_nsamples=MAX_N_CXTS):
        if eval_source in ["gen_ancestral_concept", "gen_nucleus_concept", 
                      "gen_ancestral_all", "gen_nucleus_all"]:
            padded_cxt_toks = pad_cxt_list(self.gen_cxt_toks, max_nsamples)
        elif eval_source == "train_all":
            sub_cxt_toks_train = filter_cxt_toks_by_length(
                model_name, self.cxt_toks_train, 
                cxt_max_length_pct=CXT_MAX_LENGTH_PCT
            )
            padded_cxt_toks = torch.from_numpy(sub_cxt_toks_train)
        elif eval_source == "test_all":
            sub_cxt_toks_test = filter_cxt_toks_by_length(
                model_name, self.cxt_toks_test, 
                cxt_max_length_pct=CXT_MAX_LENGTH_PCT
            )
            padded_cxt_toks = torch.from_numpy(sub_cxt_toks_test)
        elif eval_source == "train_concept":
            concept_cxt_toks = self.cxt_toks_train[np.isin(self.y_train, [0,1])]
            sub_concept_cxt_toks = filter_cxt_toks_by_length(
                model_name, concept_cxt_toks, 
                cxt_max_length_pct=CXT_MAX_LENGTH_PCT
            )
            padded_cxt_toks = torch.from_numpy(sub_concept_cxt_toks)
        elif eval_source == "test_concept":
            concept_cxt_toks = self.cxt_toks_test[np.isin(self.y_test, [0,1])]
            sub_concept_cxt_toks = filter_cxt_toks_by_length(
                model_name, concept_cxt_toks, 
                cxt_max_length_pct=CXT_MAX_LENGTH_PCT
            )
            padded_cxt_toks = torch.from_numpy(sub_concept_cxt_toks)
        else: 
            raise ValueError(f"Evaluation context eval_source {eval_source} invalid")
        logging.info(
            f"Total contexts to sample from: {padded_cxt_toks.shape[0]}"
        )
        return padded_cxt_toks

    def sample_filtered_contexts(self):
        idx = torch.randperm(self.cxt_toks.shape[0])
        sample_cxt_toks = self.cxt_toks[idx[:self.nsamples]]
        return sample_cxt_toks

    #########################################
    # Probability computations              #
    #########################################
    def compute_do_qxhs(self,
        do_type, cxt_hidden_state, n_ntok_H, batch_tokens):
        """ input dimensions:
        - do_type: ["c=0", "c=1"]
        - cxt_hidden_state: 1 x d 
        - n_ntok_H: bs x max_ntokens x d
        - batch_tokens: bs x max_n_tokens 

        output: tensor word probabilities
        - batch_word_probs: msamples x bs
        """
        if do_type == "c=0":
            hpars = self.l0_hs
        elif do_type == "c=1":
            hpars = self.l1_hs
        else:
            raise ValueError(f"Incorrect do_type: {do_type}")

        # intervention on first hs
        # shape: (msamples x d)
        first_hs_int = intervene_first_h(
            cxt_hidden_state, "hpar", self.msamples, 
            hpars, self.P, self.I_P, self.device
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

    #TODO: SAME AS IN DISTRIBUTOR, SHOULD MOVE TO SHARED UTILS
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
    
    #TODO: SAME AS IN DISTRIBUTOR, SHOULD MOVE TO SHARED UTILS
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

    def compute_all_intervention_distribs(
        self, cxt_hidden_state, pkv_batch_hs, batch_tokens
    ):
        """ Computes all interventional distributions and 
        outputs each in a 2-dimensional tensor
        """
        # original p(x|h): 1 x 2
        p_x_mid_h = self.compute_pxhs(
            cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        # qbot(x|hbot): msamples x 2
        q_x_mid_hbot = self.compute_qxhs(
            cxt_hidden_state, pkv_batch_hs, "hpar", 
            batch_tokens
        )
        # p(x | hbot, do(C=0)): msamples x 2
        do_c0_word_probs = self.compute_do_qxhs(
            "c=0", cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        # p(x | hbot, do(C=1)): msamples x 2 
        do_c1_word_probs = self.compute_do_qxhs(
            "c=1", cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        return (
            p_x_mid_h.squeeze(0),
            q_x_mid_hbot.mean(0),
            do_c0_word_probs.mean(0),
            do_c1_word_probs.mean(0)
        )
            
    @staticmethod
    def score_interventions(y: int, 
        orig: torch.tensor, erased: torch.tensor, 
        do_c0: torch.tensor, do_c1: torch.tensor
    ) -> dict:
        """ Scores word probabilities. 
        orig, erased, do_c0 and d0_c1 have shape (2)
        where position 0 is the probability of the fact word
        and position 1 is the probability of the foil word
        """
        # fact > foil
        base_correct = orig[0] > orig[1]
        # fact > foil
        erased_correct = erased[0] > erased[1]
        if y == 0:
            # fact is c=0, foil is c=1
            do_c0_correct = do_c0[0] > do_c0[1] # c=0 > c=1
            do_c1_correct = do_c0[1] > do_c0[0] # c=1 > c=0
        elif y == 1:
            # fact is c=1, foil is c=0
            do_c0_correct = do_c0[1] > do_c0[0] # c=0 > c=1
            do_c1_correct = do_c0[0] > do_c0[1] # c=1 > c=0
        else:
            raise ValueError(f"Incorrect label y: {y}")
        return dict(
            y = y,
            base_correct = base_correct.item(),
            erased_correct = erased_correct.item(),
            do_c0_correct = do_c0_correct.item(),
            do_c1_correct = do_c1_correct.item()
        )

    def intervene_and_score(self, y, fact, foil, cxt_hidden_state, cxt_pkv):
        batch_tokens = torch.tensor(
            list(zip_longest(*[fact, foil], 
                fillvalue=self.tokenizer.pad_token_id))
        ).T.to(self.device)

        batch_size_pkv = duplicate_pkv(cxt_pkv, batch_tokens.shape[0])

        pkv_output = self.model(
            input_ids=batch_tokens, 
            #attention_mask=attention_mask, 
            labels=batch_tokens,
            output_hidden_states=True,
            past_key_values=batch_size_pkv
        )

        pkv_batch_hs = pkv_output["hidden_states"][-1]
        orig, erased, do_c0, do_c1 = self.compute_all_intervention_distribs(
            cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        scores = self.score_interventions(y, orig, erased, do_c0, do_c1)
        return scores

    #TODO: SAME AS MultiTokenDistribor move to utils
    def compute_cxt_pkv_h(self, cxt):
        """ Takes a context string and outputs past key values
        and the hidden state corresponding to last token of 
        context string.
        Outputs: 
        - cxt_pkv: nlayers tuple, (([1, d1, d2, d3], []), )
        - cxt_hidden_state: (1, d)
        """
        cxt_tok = cxt.unsqueeze(0).to(self.device)
    
        cxt_output = self.model(
            input_ids=cxt_tok, 
            #attention_mask=attention_mask, 
            labels=cxt_tok,
            output_hidden_states=True,
            #past_key_values= cxt_pkv
        )
        cxt_pkv = cxt_output.past_key_values

        # (1 , d)
        cxt_hidden_state = cxt_output["hidden_states"][-1][:, -1, :]
        return cxt_pkv, cxt_hidden_state

    def compute_intervention_eval(self, samples, pad_token=-1):
        #TODO: add application of nsamples
        scores = []
        for i, (cxt_pad, y, fact, foil) in enumerate(tqdm(samples)):

            cxt = cxt_pad[cxt_pad != pad_token]

            logging.info(
                f"---New eval context: {self.tokenizer.decode(cxt)}---"
            )

            with torch.no_grad():
                with torch.autocast(
                    device_type="cuda", dtype=self.torch_dtype, enabled=True
                ):
                    cxt_pkv, cxt_hidden_state = self.compute_cxt_pkv_h(cxt)

                    sample_score = self.intervene_and_score(
                        y, fact, foil, cxt_hidden_state, cxt_pkv
                    )
                    torch.cuda.empty_cache()
                    
            scores.append(sample_score)

        return scores

