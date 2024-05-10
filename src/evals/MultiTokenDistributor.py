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
        intervene_hs, compute_log_pxh_batch,\
            fast_compute_m_p_words, fast_compute_p_words,\
                duplicate_pkv
from utils.lm_loaders import SUPPORTED_AR_MODELS, GPT2_LIST
from evals.eval_utils import load_run_Ps, load_run_output, renormalize
from data.spacy_wordlists.embedder import load_concept_token_lists
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer, get_V
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
FOOD_PROMPTS = [
    "The cuisine was",
    "The dishes were",
    "The meal was",
    "The food tasted",
    "The flavors were",
]

NOISE_PROMPTS = [
    "The ambient noise level was",
    "The background noise was",
    "The surrounding sounds were",
    "The auditory atmosphere was",
    "The ambient soundscape was",
]

SERVICE_PROMPTS = [
    "The service was", 
    "The staff was", 
    "The hospitality extended by the staff was", 
    "The waiter was", 
    "The host was", 
]

AMBIANCE_PROMPTS = [
    "The ambiance was",
    "The atmosphere was",
    "The restaurant was",
    "The vibe was",
    "The setting was",
]

#%%
class CustomDataset(Dataset, ABC):
    def __init__(self, token_tensor):
        self.data = token_tensor
        self.n_instances = token_tensor.shape[0]
        
    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return self.data[index]


#%%
class MultiTokenDistributor:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 source, # ["natural", "gen_ancestral", "gen_nucleus"]
                 nsamples, # number of test set samples
                 msamples, # number of dev set samples for each q computation
                 nwords, # number of words to use from token lists -- GET RID OF THIS
                 n_other_words, # number of other words
                 run_path, # path of LEACE training run output
                 output_folder_name, # directory for exporting individual distributions
                 iteration, # iteration of the eval for this run
                 batch_size, #batch size for the word list probability computation
                ):

        self.nsamples = nsamples
        self.msamples = msamples
        self.nwords = nwords
        self.batch_size = batch_size
        self.n_other_words = n_other_words

        nucleus = get_nucleus_arg(source)

        # directory handling
        self.outdir = get_mt_eval_directory(run_path, concept, model_name, 
            output_folder_name, source, iteration)
        os.makedirs(self.outdir, exist_ok=output_folder_name=="test")
        logging.info(f"Created outdir: {self.outdir}, exist_ok = {output_folder_name=='test'}")
        
        # Load run data
        run = load_run_output(run_path)

        # Load model
        self.device = get_device()
        self.model = get_model(model_name, device=self.device)
        self.tokenizer = get_tokenizer(model_name)
        self.new_word_tokens = self.get_new_word_tokens(model_name) 

        # Load with device
        P, I_P, _ = load_run_Ps(run_path)
        self.P = torch.tensor(P, dtype=torch.float32).to(self.device) 
        self.I_P = torch.tensor(I_P, dtype=torch.float32).to(self.device)

        # Load model eval components
        self.V = get_V(model_name, model=self.model, numpy_cpu=False).clone().type(torch.float32).to(self.device)
        self.l0_tl, self.l1_tl, other_tl_full = load_concept_token_lists(concept, model_name, single_token=False)

        # subsample other words
        random.shuffle(other_tl_full)
        self.other_tl = other_tl_full[:self.n_other_words]

        if self.nwords is not None:
            logging.warn(f"Applied nwords={self.nwords}, intended for DEBUGGING ONLY")
            self.l0_tl = self.l0_tl[:self.nwords]
            self.l1_tl = self.l1_tl[:self.nwords]

        # CEBaB prompts
        #TODO: clean this up once tested
        #self.prompt_set = self.load_suffixes(concept, source)
        self.prompt_set = None # added suffixes to CEBaB data in preprocessing
        logging.info(f"Prompt set for samples: {self.prompt_set}")
        #self.prompt_end_space = model_name == "llama2"

        # Load generated samples
        _, self.gen_l0_cxt_toks, self.gen_l1_cxt_toks, self.gen_all_hs = prep_generated_data(
            model_name, concept, nucleus
        )

        # Load test set samples
        self.y_dev, self.cxt_toks_dev = \
            run["y_val"], run["cxt_toks_val"]
        self.y_test, self.cxt_toks_test = \
            run["y_test"], run["cxt_toks_test"]

        # Select L0 and L1 samples to use to compute distributions based on source
        # TODO: need to try to do this with all_hs samples -- need past_key_values!
        self.l0_cxt_toks, self.l1_cxt_toks = self.get_concept_eval_contexts(source)
        

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
    def get_concept_eval_contexts(self, source):
        if source in ["gen_nucleus", "gen_ancestral"]:
            torch_gen_l0_cxt_toks = [torch.tensor(x) for x in self.gen_l0_cxt_toks]
            torch_gen_l1_cxt_toks = [torch.tensor(x) for x in self.gen_l1_cxt_toks]
            l0_cxt_toks = torch.nn.utils.rnn.pad_sequence(torch_gen_l0_cxt_toks, padding_value=-1).T
            l1_cxt_toks = torch.nn.utils.rnn.pad_sequence(torch_gen_l1_cxt_toks, padding_value=-1).T 
            return l0_cxt_toks, l1_cxt_toks
        elif source == "natural":
            l0_cxt_toks = torch.tensor(self.cxt_toks_test[self.y_test==0])
            l1_cxt_toks = torch.tensor(self.cxt_toks_test[self.y_test==1])
            # deleting the generated ones for memory
            self.gen_l0_cxt_toks, self.gen_l1_cxt_toks = None, None
            return l0_cxt_toks, l1_cxt_toks
        else: 
            raise ValueError(f"Evaluation context source {source} invalid")

    def sample_filtered_contexts(self):
        l0_len = self.l0_cxt_toks.shape[0]
        l1_len = self.l1_cxt_toks.shape[0]
        if (l0_len < self.nsamples or l1_len < self.nsamples):
            logging.info(f"Sample contexts: no sampling applied, l0 {l0_len}, l1 {l1_len}")
            return self.l0_cxt_toks, self.l1_cxt_toks
        else:    
            ratio = l1_len / l0_len
            l0_ind = np.arange(l0_len)
            l1_ind = np.arange(l1_len)
            np.random.shuffle(l0_ind)
            np.random.shuffle(l1_ind)
            if ratio > 1:
                l0_cxt_toks_sample = self.l0_cxt_toks[l0_ind[:self.nsamples]].to(self.device)
                l1_cxt_toks_sample = self.l1_cxt_toks[l1_ind[:int((self.nsamples*ratio))]].to(self.device)
            else:
                ratio = l0_len / l1_len
                l0_cxt_toks_sample = self.l0_cxt_toks[l0_ind[:int((self.nsamples*ratio))]].to(self.device)
                l1_cxt_toks_sample = self.l1_cxt_toks[l1_ind[:self.nsamples]].to(self.device)
            logging.info(f"Sample contexts: sampling applied,"
                         f" l0 {l0_cxt_toks_sample.shape[0]},"
                         f" l1 {l1_cxt_toks_sample.shape[0]}")
            return l0_cxt_toks_sample, l1_cxt_toks_sample

    #########################################
    # Concept prompt adding for CEBaB.      #
    #########################################
    #TODO: once tested, delete the three functions in this section
    @staticmethod
    def load_suffixes(concept, source):
        if source in ["gen_normal", "gen_nucleus"]:
            return None
        else:
            if concept == "ambiance":
                return AMBIANCE_PROMPTS
            elif concept == "food":
                return FOOD_PROMPTS
            elif concept == "noise":
                return NOISE_PROMPTS
            elif concept == "service":
                return SERVICE_PROMPTS
            elif concept in ["number", "gender"]:
                return None
            else:
                raise ValueError("Incorrect concept")

    @staticmethod
    def add_suffix(text, suffix):
        stripped_text = text.strip()
        if stripped_text.endswith("."):
            return stripped_text + " " + suffix
        else:
            return stripped_text + ". " + suffix
        
    def add_concept_suffix(self, cxt_tokens):
        text = self.tokenizer.decode(cxt_tokens)
        suffix = np.random.choice(self.prompt_set)
        text_w_suffix = self.add_suffix(text, suffix)
        suffixed_sentence = self.tokenizer(
            text_w_suffix, return_tensors="pt"
        )["input_ids"][0]
        double_bos = (
            suffixed_sentence[:2] == torch.tensor(
                [self.tokenizer.bos_token_id, self.tokenizer.bos_token_id]
            )
        ).all()
        if double_bos:
            return suffixed_sentence[1:]
        else:
            return suffixed_sentence

    #########################################
    # Probability computations              #
    #########################################
    def compute_pxh_batch_handler(self, method, batch_tok_ids, 
        batch_hidden_states, gpu_out):
        if method in ["hbot", "hpar"]:
            hs_int = intervene_hs(
                batch_hidden_states, method, 
                self.msamples, self.gen_all_hs, 
                self.P, self.I_P, self.device
            )
            batch_log_qxhs = compute_log_pxh_batch(
                hs_int, self.V, gpu_out
            )
            batch_word_probs = fast_compute_m_p_words(
                batch_tok_ids, batch_log_qxhs, self.tokenizer.pad_token_id, 
                self.new_word_tokens, self.device
            )
        elif method == "h":
            batch_log_pxhs = compute_log_pxh_batch(
                batch_hidden_states, self.V, gpu_out
            )
            batch_word_probs = fast_compute_p_words(
                batch_tok_ids, batch_log_pxhs, self.tokenizer.pad_token_id, 
                self.new_word_tokens, self.device
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

        with torch.no_grad():
            pkv_output = self.model(
                input_ids=batch_tokens, 
                #attention_mask=attention_mask, 
                labels=batch_tokens,
                output_hidden_states=True,
                past_key_values=dup_pkv
            )

        pkv_batch_hs = pkv_output["hidden_states"][-1]
        batch_cxt_hidden_state = cxt_hidden_state[None, :, :].repeat(
            batch_tokens.shape[0], 1, 1
        )
        batch_hidden_states = torch.cat(
            (batch_cxt_hidden_state, pkv_batch_hs), 1
        ).type(torch.float32)
        batch_word_probs = self.compute_pxh_batch_handler(
            method, batch_tokens, batch_hidden_states, gpu_out=True
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
            #i, batch_tokens = next(enumerate(dl))

            batch_word_probs = self.compute_batch_p_words(
                batch_tokens, cxt_pkv, batch_size_pkv, 
                cxt_hidden_state, method
            )

            tl_word_probs.append(batch_word_probs)

        return torch.hstack(tl_word_probs).cpu().numpy()

    def compute_cxt_pkv_h(self, cxt):
        cxt_tok = cxt.to(self.device)
        with torch.no_grad():
            cxt_output = self.model(
                input_ids=cxt_tok, 
                #attention_mask=attention_mask, 
                labels=cxt_tok,
                output_hidden_states=True,
                #past_key_values= cxt_pkv
            )
            cxt_pkv = cxt_output.past_key_values
        cxt_hidden_state = cxt_output["hidden_states"][-1][-1].unsqueeze(0)
        return cxt_pkv, cxt_hidden_state

    def compute_lemma_probs(self, lemma_samples, method, outdir, pad_token=-1):
        l0_probs, l1_probs, other_probs = [], [], []
        for i, cxt_pad in enumerate(tqdm(lemma_samples)):
            cxt = cxt_pad[cxt_pad != pad_token]

            if self.prompt_set is not None:
                cxt = self.add_concept_suffix(cxt)

            logging.info(f"---New eval context: {self.tokenizer.decode(cxt)}---")

            #start = time.time()
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

        return np.stack(l0_probs), np.stack(l1_probs), np.stack(other_probs)

    def compute_pxs(self, htype):
        htype_outdir = os.path.join(
            self.outdir, 
            f"h_distribs/{htype}"
        )
        os.makedirs(htype_outdir, exist_ok=False)   

        assert self.nsamples is not None
        l0_inputs, l1_inputs = self.sample_filtered_contexts()
        if htype == "l0_cxt_qxhs_par": 
            l0_cxt_qxhs_par = self.compute_lemma_probs(l0_inputs, "hbot", htype_outdir)
        elif htype == "l1_cxt_qxhs_par":
            l1_cxt_qxhs_par = self.compute_lemma_probs(l1_inputs, "hbot", htype_outdir)
        elif htype == "l0_cxt_qxhs_bot":
            l0_cxt_qxhs_bot = self.compute_lemma_probs(l0_inputs, "hpar", htype_outdir)
        elif htype == "l1_cxt_qxhs_bot":
            l1_cxt_qxhs_bot = self.compute_lemma_probs(l1_inputs, "hpar", htype_outdir)
        elif htype == "l0_cxt_pxhs":
            l0_cxt_pxhs = self.compute_lemma_probs(l0_inputs, "h", htype_outdir)
        elif htype == "l1_cxt_pxhs":
            l1_cxt_pxhs = self.compute_lemma_probs(l1_inputs, "h", htype_outdir)
        else:
            raise ValueError(f"Incorrect htype: {htype}")

    def compute_all_pxs(self):
        self.compute_pxs("l0_cxt_qxhs_par")
        self.compute_pxs("l1_cxt_qxhs_par")
        self.compute_pxs("l0_cxt_qxhs_bot")
        self.compute_pxs("l1_cxt_qxhs_bot")
        self.compute_pxs("l0_cxt_pxhs")
        self.compute_pxs("l1_cxt_pxhs")

