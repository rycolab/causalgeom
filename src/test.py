#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
from datetime import datetime
import csv

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
from itertools import zip_longest

#sys.path.append('..')
#sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS


from evals.mi_eval import prep_generated_data, compute_inner_loop_qxhs
#from utils.lm_loaders import get_V, GPT2_LIST, BERT_LIST
from evals.eval_utils import load_run_Ps, load_run_output, load_model_eval
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
class MultiTokenEvaluator:

    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 nucleus, # whether to use samples generated with nucleus sampling
                 nsamples, # number of test set samples
                 msamples, # number of dev set samples for each q computation
                 nwords, # number of words to use from token lists -- GET RID OF THIS
                 run_path, # path of LEACE training run output
                ):

        self.nsamples = nsamples
        self.msamples = msamples
        self.nwords = nwords

        run = load_run_output(run_path)
        self.P, self.I_P, _ = load_run_Ps(run_path)

        # test set version of the eval
        self.V, self.l0_tl, self.l1_tl = load_model_eval(
            model_name, concept, single_token=False
        )

        #p_x = load_p_x(MODEL_NAME, NUCLEUS)
        self.p_c, self.l0_hs_wff, self.l1_hs_wff, self.all_hs = prep_generated_data(
            model_name, nucleus
        )

        #TODO: get rid of unnecessary ones
        self.X_dev, self.y_dev, self.facts_dev, self.foils_dev, self.cxt_toks_dev = \
            run["X_val"], run["y_val"], run["facts_val"], run["foils_val"], run["cxt_toks_val"]
        self.X_test, self.y_test, self.facts_test, self.foils_test, self.cxt_toks_test = \
            run["X_test"], run["y_test"], run["facts_test"], run["foils_test"], run["cxt_toks_test"]

        #TODO: RIGHT NOW THIS IS THE ONLY WAY, NEED TO ENABLE GENERATED
        source = "natural"
        self.l0_cxt_toks, self.l1_cxt_toks = self.get_concept_eval_contexts(source)

        #%%
        self.model = get_model(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.device = get_device()

        if model_name == "gpt2-large":
            self.model = self.model.to(self.device)

        self.new_word_tokens = self.get_new_word_tokens(model_name) 

    #########################################
    # Tokenizer specific new word tokens    #
    #########################################
    def get_gpt2_large_new_word_tokens(self):
        #TODO: need to refine this list cuz other tokens includes punctuation for example
        new_word_tokens = []
        new_word_token_pairs = []
        other_tokens = []
        other_token_pairs = []
        for token, token_id in self.tokenizer.vocab.items():
            if token.startswith("Ġ"):
                new_word_tokens.append(token_id)
                new_word_token_pairs.append((token, token_id))
            else:
                other_tokens.append(token_id)
                other_token_pairs.append((token, token_id))
        return new_word_tokens

    def get_new_word_tokens(self, model_name):
        if model_name == "gpt2-large":
            return self.get_gpt2_large_new_word_tokens()
        #elif model_name == "llama2":
        else:
            return NotImplementedError(f"Model not yet implemented")
    
    #########################################
    # Data handling                         #
    #########################################
    def get_concept_eval_contexts(self, source):
        if source == "generated":
            raise NotImplementedError("This is currently not working cuz I save h's not tokens")
            #l0_hs_n, l1_hs_n = sample_filtered_hs(evaluator.l0_hs_wff, evaluator.l0_hs_wff, evaluator.nsamples)
        elif source == "natural":
            l0_cxt_toks = self.cxt_toks_test[self.y_test==0]
            l1_cxt_toks = self.cxt_toks_test[self.y_test==1]
            return l0_cxt_toks, l1_cxt_toks
        else: 
            raise ValueError(f"Evaluation context source {source} invalid")

    @staticmethod
    def sample_filtered_contexts(l0_cxts, l1_cxts, nsamples):
        np.random.shuffle(l0_cxts)
        np.random.shuffle(l1_cxts)
        ratio = len(l1_cxts)/len(l0_cxts)
        if ratio > 1:
            l0_cxts = l0_cxts[:nsamples]
            l1_cxts = l1_cxts[:int((nsamples*ratio))]
        else:
            ratio = len(l0_cxts) / len(l1_cxts)
            l0_cxts = l0_cxts[:int((nsamples*ratio))]
            l1_cxts = l1_cxts[:nsamples]
        return l0_cxts, l1_cxts 

    #########################################
    # Probability computations              #
    #########################################
    def compute_p_word_method_handler(self, method, h):
        if method in ["hbot", "hpar"]:
            return compute_inner_loop_qxhs(
                method, h, self.all_hs, self.P, self.I_P, 
                self.V, self.msamples, processor=None
            ).mean(axis=0)
        elif method == "h":
            logits = self.V @ h
            pxh = softmax(logits)
            return pxh
        else:
            raise ValueError(f"Incorrect method argument {method}")

    def compute_p_word(self, word_tok, cxt_last_index, hidden_states, method):
        i = 0 
        p_word = 1
        while i < len(word_tok):
            logging.info(f"Loop iteration: {i}")
            h = hidden_states[cxt_last_index + i]
            logging.info(f"Hs index: {cxt_last_index + i} / {cxt_last_index + len(word_tok)}")
            pxh = self.compute_p_word_method_handler(method, h)
            #qxh = compute_inner_loop_qxhs(
            #    "hbot", h, self.all_hs, self.P, self.I_P, 
            #    self.V, self.msamples, processor=None
            #).mean(axis=0)
            token_prob = pxh[word_tok[i]]
            logging.info(f"Token prob: {token_prob}")
            p_word = p_word * token_prob
            logging.info(f"Word prob: {p_word}")
            i += 1
        h = hidden_states[cxt_last_index + i]
        logging.info(f"Last h index: {cxt_last_index + i} / {cxt_last_index + len(word_tok)}")
        #qxh = compute_inner_loop_qxhs(
        #    "hbot", h, self.all_hs, self.P, self.I_P, self.V, self.msamples, processor=None
        #).mean(axis=0)
        pxh = self.compute_p_word_method_handler(method, h)
        p_new_word = pxh[self.new_word_tokens].sum()
        logging.info(f"New word prob: {p_new_word}")
        p_word = p_word * p_new_word
        logging.info(f"Final word prob: {p_word}")
        return p_word

    def compute_batch_p_words(self, token_list, cxt_plus_tl_batched, batch_hidden_states, cxt_last_index, method):
        p_words = []
        for word_index, word_tok in enumerate(token_list):
            logging.info(f"------Computing probability of word {word_index}-------")
            cxt_plus_word_tok = cxt_plus_tl_batched[word_index]
            hidden_states = batch_hidden_states[word_index]
            p_word = self.compute_p_word(word_tok, cxt_last_index, hidden_states, method)
            p_words.append(p_word)
        return p_words

    def compute_p_words(self, word_tl, cxt, cxt_last_index, method):
        cxt_plus_tl = [np.append(cxt, x).tolist() for x in word_tl]
        cxt_plus_tl_batched = torch.tensor(
            list(zip_longest(*cxt_plus_tl, fillvalue=self.tokenizer.pad_token_id))
        ).T

        with torch.no_grad():
            cxt_plus_tl_input = cxt_plus_tl_batched.to(self.device)
            output = self.model(
                input_ids=cxt_plus_tl_input, 
                #attention_mask=attention_mask, 
                labels=cxt_plus_tl_input,
                output_hidden_states=True,
                #past_key_values= cxt_pkv
            )
        
        batch_hidden_states = output["hidden_states"][-1].cpu().numpy()
        tl_word_probs = self.compute_batch_p_words(
            word_tl, cxt_plus_tl_batched, batch_hidden_states, cxt_last_index, method
        )
        return tl_word_probs

    def compute_all_word_probs(self, cxt, method):
        cxt_last_index = cxt.shape[0] - 1
        #TODO: get rid of this after debugging
        if self.nwords:
            l0_word_probs = self.compute_p_words(self.l0_tl[:self.nwords], cxt, cxt_last_index, method)
            l1_word_probs = self.compute_p_words(self.l1_tl[:self.nwords], cxt, cxt_last_index, method)
        else:
            l0_word_probs = self.compute_p_words(self.l0_tl, cxt, cxt_last_index, method)
            l1_word_probs = self.compute_p_words(self.l1_tl, cxt, cxt_last_index, method)
        return l0_word_probs, l1_word_probs

    def compute_lemma_probs(self, padded_contexts, method, pad_token=-1):
        """ method: ["h", "hbot", "hpar"] """
        concept_prob_pairs = []
        for cxt_pad in padded_contexts:
            #TODO: will want to replace this pad token with the tokenizer pad token
            cxt = cxt_pad[cxt_pad != pad_token]
            logging.info(f"------New eval context: {cxt}------")
            l0_word_probs, l1_word_probs = self.compute_all_word_probs(cxt, method)
            concept_prob_pairs.append((l0_word_probs, l1_word_probs))
        return concept_prob_pairs


#%%
# To do list:
# - adapt this to compute the quantities from single token mi_eval
# - introduce option to use test set samples AND generated samples


#%%
evaluator = MultiTokenEvaluator(
    "gpt2-large", 
    "number", 
    True, # nucleus 
    3, #nsamples
    10, #msamples
    10, #nwords
    os.path.join(OUT, 
        "run_output/number/gpt2-large/leace29022024/run_leace_number_gpt2-large_2024-02-29-18:30:00_0_3.pkl"
    ), #run_path
)

#%%
def sample_filtered_contexts(l0_cxts, l1_cxts, nsamples):
    np.random.shuffle(l0_cxts)
    np.random.shuffle(l1_cxts)
    ratio = len(l1_cxts)/len(l0_cxts)
    if ratio > 1:
        l0_cxts = l0_cxts[:nsamples]
        l1_cxts = l1_cxts[:int((nsamples*ratio))]
    else:
        ratio = len(l0_cxts) / len(l1_cxts)
        l0_cxts = l0_cxts[:int((nsamples*ratio))]
        l1_cxts = l1_cxts[:nsamples]
    return l0_cxts, l1_cxts 

#%%
#y_dev, cxt_toks_dev = evaluator.y_dev, evaluator.cxt_toks_dev
#y_test, cxt_toks_test = evaluator.y_test, evaluator.cxt_toks_test

#TODO: put this into the if else below -- maybe only the second part idk        

assert evaluator.nsamples is not None
l0_cxt_toks_n, l1_cxt_toks_n = sample_filtered_contexts(
    evaluator.l0_cxt_toks, evaluator.l1_cxt_toks, evaluator.nsamples
)
l0_cxt_qxhs_par = evaluator.compute_lemma_probs(l0_cxt_toks_n, "hbot")
l1_cxt_qxhs_par = evaluator.compute_lemma_probs(l1_cxt_toks_n, "hbot")

l0_cxt_toks_n, l1_cxt_toks_n = sample_filtered_contexts(
    evaluator.l0_cxt_toks, evaluator.l1_cxt_toks, evaluator.nsamples
)
l0_cxt_qxhs_bot = evaluator.compute_lemma_probs(l0_cxt_toks_n, "hpar")
l1_cxt_qxhs_bot = evaluator.compute_lemma_probs(l1_cxt_toks_n, "hpar")

l0_cxt_toks_n, l1_cxt_toks_n = sample_filtered_contexts(
    evaluator.l0_cxt_toks, evaluator.l1_cxt_toks, evaluator.nsamples
)
l0_cxt_pxhs = evaluator.compute_lemma_probs(l0_cxt_toks_n, "h")
l1_cxt_qxhs = evaluator.compute_lemma_probs(l1_cxt_toks_n, "h")

#%%


#%% new h
#with torch.no_grad():
#    cxt_input = torch.tensor(cxt).to(device)
#    output = model(
#        input_ids=cxt_input, 
#        #attention_mask=attention_mask, 
#        labels=cxt_input,
#        output_hidden_states=True
#    )
#    cxt_pkv = output["past_key_values"]


#%%




#%%
#p_w = first_qxh[word_tok[0]]
#i = 1
#cxt_loop = torch.tensor(np.append(cxt.copy(), word_tok[0]))#.to(device)
#while i < len(word_tok):
#    with torch.no_grad():
#        output = model(
#            input_ids=cxt_loop, 
##            #attention_mask=attention_mask, 
#            labels=cxt_loop,
#            output_hidden_states=True
#        ) 



# %%

