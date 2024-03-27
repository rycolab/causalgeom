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
import random 
from scipy.special import softmax
#from scipy.stats import entropy
from tqdm import trange
from transformers import TopPLogitsWarper, LogitsProcessorList
import torch 
from itertools import zip_longest
from scipy.special import softmax
from scipy.stats import entropy

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT, RESULTS, MODELS


from evals.mi_eval import prep_generated_data, compute_inner_loop_qxhs
from utils.lm_loaders import SUPPORTED_AR_MODELS
from evals.eval_utils import load_run_Ps, load_run_output, load_model_eval,\
    renormalize
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer
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
# TODO:
# - incorporate the two functions imported from mi_eval into the class
#
class MultiTokenEvaluator:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 nucleus, # whether to use samples generated with nucleus sampling
                 nsamples, # number of test set samples
                 msamples, # number of dev set samples for each q computation
                 nwords, # number of words to use from token lists -- GET RID OF THIS
                 run_path, # path of LEACE training run output
                 outdir, # directory for exporting individual distributions
                ):

        self.nsamples = nsamples
        self.msamples = msamples
        self.nwords = nwords
        self.outdir = outdir

        run = load_run_output(run_path)
        self.P, self.I_P, _ = load_run_Ps(run_path)

        # test set version of the eval
        self.V, self.l0_tl, self.l1_tl = load_model_eval(
            model_name, concept, single_token=False
        )

        # CEBaB prompts
        self.prompt_set = self.load_suffixes(concept)
        self.prompt_end_space = model_name == "llama2"

        #p_x = load_p_x(MODEL_NAME, NUCLEUS)
        self.p_c, self.l0_hs_wff, self.l1_hs_wff, self.all_hs = prep_generated_data(
            model_name, nucleus
        )

        #self.X_dev, self.y_dev, self.facts_dev, self.foils_dev, self.cxt_toks_dev = \
        #    run["X_val"], run["y_val"], run["facts_val"], run["foils_val"], run["cxt_toks_val"]
        #self.X_test, self.y_test, self.facts_test, self.foils_test, self.cxt_toks_test = \
        #    run["X_test"], run["y_test"], run["facts_test"], run["foils_test"], run["cxt_toks_test"]
        self.y_dev, self.cxt_toks_dev = run["y_val"], run["cxt_toks_val"]
        self.y_test, self.cxt_toks_test = run["y_test"], run["cxt_toks_test"]

        #TODO: RIGHT NOW THIS IS THE ONLY WAY, NEED TO ENABLE GENERATED
        source = "natural"
        self.l0_cxt_toks, self.l1_cxt_toks = self.get_concept_eval_contexts(source)

        #%%
        self.device = get_device()
        self.model = get_model(model_name, device=self.device)
        self.tokenizer = get_tokenizer(model_name)

        self.new_word_tokens = self.get_new_word_tokens(model_name) 

    #########################################
    # Tokenizer specific new word tokens    #
    #########################################
    def get_gpt2_large_new_word_tokens(self):
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
        if model_name == "gpt2-large":
            return self.get_gpt2_large_new_word_tokens()
        elif model_name == "llama2":
            return self.get_llama2_new_word_tokens()
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
    # Concept prompt adding for CEBaB.      #
    #########################################
    @staticmethod
    def load_suffixes(concept):
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
    def add_suffix(text, suffix, end_space):
        if end_space:
            suffix = suffix + " "

        if text.strip().endswith("."):
            return text + " " + suffix
        else:
            return text + ". " + suffix
        
    def add_concept_suffix(self, cxt_tokens):
        text = self.tokenizer.decode(cxt_tokens)
        suffix = np.random.choice(self.prompt_set)
        text_w_suffix = self.add_suffix(text, suffix, self.prompt_end_space)
        suffixed_sentence = self.tokenizer(
            text_w_suffix, return_tensors="pt"
        )["input_ids"][0]
        return suffixed_sentence

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
            #logging.info(f"Loop iteration: {i}")
            h = hidden_states[cxt_last_index + i]
            #logging.info(f"Hs index: {cxt_last_index + i} / {cxt_last_index + len(word_tok)}")
            pxh = self.compute_p_word_method_handler(method, h)
            #qxh = compute_inner_loop_qxhs(
            #    "hbot", h, self.all_hs, self.P, self.I_P, 
            #    self.V, self.msamples, processor=None
            #).mean(axis=0)
            token_prob = pxh[word_tok[i]]
            #logging.info(f"Token prob: {token_prob}")
            p_word = p_word * token_prob
            #logging.info(f"Word prob: {p_word}")
            i += 1
        h = hidden_states[cxt_last_index + i]
        #logging.info(f"Last h index: {cxt_last_index + i} / {cxt_last_index + len(word_tok)}")
        #qxh = compute_inner_loop_qxhs(
        #    "hbot", h, self.all_hs, self.P, self.I_P, self.V, self.msamples, processor=None
        #).mean(axis=0)
        pxh = self.compute_p_word_method_handler(method, h)
        p_new_word = pxh[self.new_word_tokens].sum()
        #logging.info(f"New word prob: {p_new_word}")
        p_word = p_word * p_new_word
        #logging.info(f"Final word prob: {p_word}")
        return p_word

    def compute_batch_p_words(self, token_list, cxt_plus_tl_batched, batch_hidden_states, cxt_last_index, method):
        p_words = []
        #for word_index, word_tok in enumerate(tqdm(token_list)):
        for word_index, word_tok in enumerate(token_list):
            #logging.info(f"------Computing probability of word {word_index}-------")
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
        if self.nwords is not None:
            l0_word_probs = self.compute_p_words(self.l0_tl[:self.nwords], cxt, cxt_last_index, method)
            l1_word_probs = self.compute_p_words(self.l1_tl[:self.nwords], cxt, cxt_last_index, method)
        else:
            l0_word_probs = self.compute_p_words(self.l0_tl, cxt, cxt_last_index, method)
            l1_word_probs = self.compute_p_words(self.l1_tl, cxt, cxt_last_index, method)
        return l0_word_probs, l1_word_probs

    def compute_lemma_probs(self, padded_contexts, method, pad_token=-1):
        """ method: ["h", "hbot", "hpar"] """
        l0_probs, l1_probs = [], []
        for i, cxt_pad in enumerate(tqdm(padded_contexts)):
            #TODO: will want to replace this pad token with the tokenizer pad token
            cxt = cxt_pad[cxt_pad != pad_token]
            
            if self.prompt_set:
                cxt = self.add_concept_suffix(cxt)
            
            logging.info(f"------New eval context: {self.tokenizer.decode(cxt)}------")
            l0_word_probs, l1_word_probs = self.compute_all_word_probs(cxt, method)

            export_path = os.path.join(self.outdir, f"word_probs_sample_{i}.pkl")
    
            with open(export_path, 'wb') as f:
                pickle.dump((l0_word_probs, l1_word_probs), f, protocol=pickle.HIGHEST_PROTOCOL)
            
            l0_probs.append(l0_word_probs)
            l1_probs.append(l1_word_probs)
            #concept_prob_pairs.append((l0_word_probs, l1_word_probs))
        return np.vstack(l0_probs), np.vstack(l1_probs)

    def compute_pxs(self, htype):
        assert self.nsamples is not None
        l0_cxt_toks_n, l1_cxt_toks_n = self.sample_filtered_contexts(
            self.l0_cxt_toks, self.l1_cxt_toks, self.nsamples
        )
        if htype == "l0_cxt_qxhs_par": 
            l0_cxt_qxhs_par = self.compute_lemma_probs(l0_cxt_toks_n, "hbot")
            return l0_cxt_qxhs_par
        elif htype == "l1_cxt_qxhs_par":
            l1_cxt_qxhs_par = self.compute_lemma_probs(l1_cxt_toks_n, "hbot")
            return l1_cxt_qxhs_par
        elif htype == "l0_cxt_qxhs_bot":
            l0_cxt_qxhs_bot = self.compute_lemma_probs(l0_cxt_toks_n, "hpar")
            return l0_cxt_qxhs_bot
        elif htype == "l1_cxt_qxhs_bot":
            l1_cxt_qxhs_bot = self.compute_lemma_probs(l1_cxt_toks_n, "hpar")
            return l1_cxt_qxhs_bot
        elif htype == "l0_cxt_pxhs":
            l0_cxt_pxhs = self.compute_lemma_probs(l0_cxt_toks_n, "h")
            return l0_cxt_pxhs
        elif htype == "l1_cxt_pxhs":
            l1_cxt_pxhs = self.compute_lemma_probs(l1_cxt_toks_n, "h")
            return l1_cxt_pxhs
        else:
            raise ValueError(f"Incorrect htype: {htype}")

# %%
def compute_eval(model_name, concept, run_path,
    nsamples, msamples, nwords, nucleus, output_folder, htype, iteration):
    #rundir = os.path.join(
    #    OUT, f"run_output/{concept}/{model_name}/{run_output_folder}"
    #)

    rundir = os.path.dirname(run_path)
    rundir_name = os.path.basename(rundir)

    run_id = run_path[-27:-4]
    outdir = os.path.join(
        os.path.dirname(rundir), 
        f"{output_folder}_{rundir_name}/run_{run_id}/nuc_{nucleus}/evaliter_{iteration}/h_distribs/{htype}"
    )
    os.makedirs(outdir, exist_ok=False)

    #run_files = [x for x in os.listdir(rundir) if x.endswith(".pkl")]
    #random.shuffle(run_files)

    #for run_file in run_files:
    #    run_path = os.path.join(rundir, run_file)
    #outpath = os.path.join(
    #    outdir, 
    #    f"{concept}_{model_name}_nuc_{nucleus}_{iteration}_{run_file[:-4]}.pkl"
    #)

    #run = load_run_output(run_path)
    #if run["config"]["k"] != k:
    #    continue
    #elif os.path.exists(outpath):
        #logging.info(f"Run already evaluated: {run_path}")
        #continue
    #else:
    evaluator = MultiTokenEvaluator(
        model_name, 
        concept, 
        nucleus, # nucleus 
        nsamples, #nsamples
        msamples, #msamples
        nwords, #nwords
        run_path, #run_path
        outdir
    )
    run_eval_output = evaluator.compute_pxs(htype)
    logging.info(f"Done")
    #logging.info(f"Finished computing evals for pair {model_name}, {concept}, folder {run_output_folder}, k:{k}")

#%%#################
# Main             #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Formatting Results Tables')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["number", "gender", "food", "ambiance", "service", "noise"],
        help="Concept to create embedded word lists for"
    )
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        help="Models to create embedding files for"
    )
    argparser.add_argument(
        "-k",
        type=int,
        default=1,
        help="K value for the runs"
    )
    argparser.add_argument(
        "-nsamples",
        type=int,
        help="Number of samples for outer loops"
    )
    argparser.add_argument(
        "-msamples",
        type=int,
        help="Number of samples for inner loops"
    )
    argparser.add_argument(
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    argparser.add_argument(
        "-run_path",
        type=str,
        default=None,
        help="Run to evaluate"
    )
    argparser.add_argument(
        "-out_folder",
        type=str,
        default="test",
        help="Directory for exporting run eval"
    )
    argparser.add_argument(
        "-htype",
        type=str,
        choices=["l0_cxt_qxhs_par", "l1_cxt_qxhs_par", 
                 "l0_cxt_qxhs_bot", "l1_cxt_qxhs_bot", 
                 "l0_cxt_pxhs", "l1_cxt_pxhs"],
        help="Type of test set contexts to compute eval distrib for"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    concept = args.concept
    nucleus = args.nucleus
    k = args.k
    nsamples=args.nsamples
    msamples=args.msamples
    nwords = None
    output_folder = args.out_folder
    run_path = args.run_path
    nruns = 3
    htype=args.htype
    #model_name = "gpt2-large"
    #concept = "food"
    #nucleus = True
    #k=1
    #nsamples=3
    #msamples=3
    #nwords = None
    #output_folder = "test_multitokeneval"
    #nruns = 1
    #htype = "l1_cxt_qxhs_par"
    #run_path="out/run_output/food/gpt2-large/leace29022024/run_leace_food_gpt2-large_2024-02-29-17:25:50_0_3.pkl"
    

    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        compute_eval(
            model_name, concept, run_path, nsamples, msamples, 
            nwords, nucleus, output_folder, htype, i
        )
    logging.info("Finished exporting all results.")

