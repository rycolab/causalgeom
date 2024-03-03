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
        for word_index, word_tok in enumerate(tqdm(token_list)):
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
        if self.nwords:
            l0_word_probs = self.compute_p_words(self.l0_tl[:self.nwords], cxt, cxt_last_index, method)
            l1_word_probs = self.compute_p_words(self.l1_tl[:self.nwords], cxt, cxt_last_index, method)
        else:
            l0_word_probs = self.compute_p_words(self.l0_tl, cxt, cxt_last_index, method)
            l1_word_probs = self.compute_p_words(self.l1_tl, cxt, cxt_last_index, method)
        return l0_word_probs, l1_word_probs

    def compute_lemma_probs(self, padded_contexts, method, pad_token=-1):
        """ method: ["h", "hbot", "hpar"] """
        l0_probs, l1_probs = [], []
        for cxt_pad in tqdm(padded_contexts):
            #TODO: will want to replace this pad token with the tokenizer pad token
            cxt = cxt_pad[cxt_pad != pad_token]
            logging.info(f"------New eval context: {cxt}------")
            l0_word_probs, l1_word_probs = self.compute_all_word_probs(cxt, method)
            l0_probs.append(l0_word_probs)
            l1_probs.append(l1_word_probs)
            #concept_prob_pairs.append((l0_word_probs, l1_word_probs))
        return np.vstack(l0_probs), np.vstack(l1_probs)

    def compute_all_pxs(self):
        assert self.nsamples is not None
        l0_cxt_toks_n, l1_cxt_toks_n = self.sample_filtered_contexts(
            self.l0_cxt_toks, self.l1_cxt_toks, self.nsamples
        )
        l0_cxt_qxhs_par = self.compute_lemma_probs(l0_cxt_toks_n, "hbot")
        l1_cxt_qxhs_par = self.compute_lemma_probs(l1_cxt_toks_n, "hbot")

        l0_cxt_toks_n, l1_cxt_toks_n = self.sample_filtered_contexts(
            self.l0_cxt_toks, self.l1_cxt_toks, self.nsamples
        )
        l0_cxt_qxhs_bot = self.compute_lemma_probs(l0_cxt_toks_n, "hpar")
        l1_cxt_qxhs_bot = self.compute_lemma_probs(l1_cxt_toks_n, "hpar")

        l0_cxt_toks_n, l1_cxt_toks_n = self.sample_filtered_contexts(
            self.l0_cxt_toks, self.l1_cxt_toks, self.nsamples
        )
        l0_cxt_pxhs = self.compute_lemma_probs(l0_cxt_toks_n, "h")
        l1_cxt_qxhs = self.compute_lemma_probs(l1_cxt_toks_n, "h")
        return l0_cxt_qxhs_par, l1_cxt_qxhs_par, l0_cxt_qxhs_bot, \
            l1_cxt_qxhs_bot, l0_cxt_pxhs, l1_cxt_qxhs

    #########################################
    # Containment and Stability             #
    #########################################
    @staticmethod
    def compute_avg_of_cond_ents(pxs):
        ents = []
        #assert case in [0,1], "Incorrect case"
        #case_pxs = pxs[case]
        for p in pxs:
            p_x_c = renormalize(p)
            ents.append(entropy(p_x_c))
        return np.mean(ents)

    @staticmethod
    def compute_ent_of_avg(pxs):
        #assert case in [0,1], "Incorrect case"
        #case_pxs = pxs[case]
        mean_px = pxs.mean(axis=0)
        p_x_c = renormalize(mean_px)
        return entropy(p_x_c)

    def compute_containment(self, l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs):
        # H(X|H_par, C)
        cont_l0_ent_qxhcs = self.compute_avg_of_cond_ents(l0_qxhs_par[0])
        cont_l1_ent_qxhcs = self.compute_avg_of_cond_ents(l1_qxhs_par[1])
        cont_ent_qxcs = (self.p_c * np.array([cont_l0_ent_qxhcs, cont_l1_ent_qxhcs])).sum()

        # H(X|C)
        l0_ent_pxc = self.compute_ent_of_avg(l0_pxhs[0])
        l1_ent_pxc = self.compute_ent_of_avg(l1_pxhs[1])
        ent_pxc = (self.p_c * np.array([l0_ent_pxc, l1_ent_pxc])).sum()

        cont_l0_mi = l0_ent_pxc - cont_l0_ent_qxhcs
        cont_l1_mi = l1_ent_pxc - cont_l1_ent_qxhcs
        cont_mi = ent_pxc - cont_ent_qxcs

        logging.info(f"Containment metrics: {cont_l0_mi}, {cont_l1_mi}, {cont_mi}")
        return dict(
            cont_l0_ent_qxhcs=cont_l0_ent_qxhcs,
            cont_l1_ent_qxhcs=cont_l1_ent_qxhcs,
            cont_ent_qxcs=cont_ent_qxcs,
            l0_ent_pxc=l0_ent_pxc,
            l1_ent_pxc=l1_ent_pxc,
            ent_pxc=ent_pxc,
            cont_l0_mi=cont_l0_mi,
            cont_l1_mi=cont_l1_mi,
            cont_mi=cont_mi
        )

    def compute_stability(self, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs):
        #H(X | H,C)
        stab_ent_xhc_l0 = self.compute_avg_of_cond_ents(l0_pxhs[0])
        stab_ent_xhc_l1 = self.compute_avg_of_cond_ents(l1_pxhs[1])
        stab_ent_xhc = (self.p_c * np.array([stab_ent_xhc_l0, stab_ent_xhc_l1])).sum()

        #H(X| H_bot,C)
        stab_l0_ent_qxhcs = self.compute_avg_of_cond_ents(l0_qxhs_bot[0])
        stab_l1_ent_qxhcs = self.compute_avg_of_cond_ents(l1_qxhs_bot[1])
        stab_ent_qxcs = (self.p_c * np.array([stab_l0_ent_qxhcs, stab_l1_ent_qxhcs])).sum()

        stab_l0_mi = stab_l0_ent_qxhcs - stab_ent_xhc_l0
        stab_l1_mi = stab_l1_ent_qxhcs - stab_ent_xhc_l1
        stab_mi = stab_ent_qxcs - stab_ent_xhc

        logging.info(f"Stability metrics: {stab_l0_mi}, {stab_l1_mi}, {stab_mi}")
        return dict(
            stab_l0_ent_qxhcs=stab_l0_ent_qxhcs,
            stab_l1_ent_qxhcs=stab_l1_ent_qxhcs,
            stab_ent_qxcs=stab_ent_qxcs,
            stab_ent_xhc_l0=stab_ent_xhc_l0,
            stab_ent_xhc_l1=stab_ent_xhc_l1,
            stab_ent_xhc=stab_ent_xhc,
            stab_l0_mi=stab_l0_mi,
            stab_l1_mi=stab_l1_mi,
            stab_mi=stab_mi
        )

    #########################################
    # Erasure and Encapsulation             #
    #########################################
    @staticmethod
    def compute_pchs_from_pxhs(pxhs):
        """ pxhs: tuple (p(l0_words | h), p(l1_words | h))
        for n h's, i.e., element 0 is a n x n_l0_words 
        dimensional np.array
        """
        pchs = []
        for pxh in zip(*pxhs):
            pch_l0 = pxh[0].sum()
            pch_l1 = pxh[1].sum()
            pch_bin = renormalize(np.array([pch_l0, pch_l1]))
            pchs.append(pch_bin)
        return np.vstack(pchs)

    def compute_concept_mis(self, l0_qxhs_par, l1_qxhs_par, \
        l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs):
        
        # H(C)
        ent_pc = entropy(self.p_c)

        # H(C | H_bot)
        l0_qchs_bot = self.compute_pchs_from_pxhs(l0_qxhs_bot)
        l1_qchs_bot = self.compute_pchs_from_pxhs(l1_qxhs_bot)
        qchs_bot = np.vstack([l0_qchs_bot, l1_qchs_bot])
        ent_qchs_bot = entropy(qchs_bot, axis=1).mean()

        # H(C | H_par)
        l0_qchs_par = self.compute_pchs_from_pxhs(l0_qxhs_par)
        l1_qchs_par = self.compute_pchs_from_pxhs(l1_qxhs_par)
        qchs_par = np.vstack([l0_qchs_par, l1_qchs_par])
        ent_qchs_par = entropy(qchs_par, axis=1).mean()

        # H(C | H)
        l0_pchs = self.compute_pchs_from_pxhs(l0_pxhs)
        l1_pchs = self.compute_pchs_from_pxhs(l1_pxhs)
        pchs = np.vstack([l0_pchs, l1_pchs])
        ent_pchs = entropy(pchs, axis=1).mean()

        res = dict(
            ent_qchs_bot = ent_qchs_bot,
            ent_qchs_par = ent_qchs_par,
            ent_pchs = ent_pchs,
            ent_pc = ent_pc,
            mi_c_hbot = ent_pc - ent_qchs_bot,
            mi_c_hpar = ent_pc - ent_qchs_par,
            mi_c_h = ent_pc - ent_pchs,
        )
        logging.info(f"Concept MI metrics I(C; H): {res['mi_c_h']},"
            f"I(C; Hbot): {res['mi_c_hbot']}, I(C; Hpar): {res['mi_c_hpar']}")
        return res

    #########################################
    # Run complete eval                     #
    #########################################
    def compute_run_eval(self):
        l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs = self.compute_all_pxs()
        containment_res = self.compute_containment(l0_qxhs_par, l1_qxhs_par, l0_pxhs, l1_pxhs)
        stability_res = self.compute_stability(l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs)
        concept_mis = self.compute_concept_mis( 
            l0_qxhs_par, l1_qxhs_par, l0_qxhs_bot, l1_qxhs_bot, l0_pxhs, l1_pxhs
        )
        return containment_res | stability_res | concept_mis


# %%
def compute_eval(model_name, concept, run_output_folder, k,
    nsamples, msamples, nucleus, output_folder, iteration):
    rundir = os.path.join(
        OUT, f"run_output/{concept}/{model_name}/{run_output_folder}"
    )
    
    outdir = os.path.join(RESULTS, f"{output_folder}/{concept}/{model_name}")
    os.makedirs(outdir, exist_ok=True)

    run_files = [x for x in os.listdir(rundir) if x.endswith(".pkl")]
    random.shuffle(run_files)

    for run_file in run_files:
        run_path = os.path.join(rundir, run_file)
        outpath = os.path.join(
            outdir, 
            f"{concept}_{model_name}_nuc_{nucleus}_{iteration}_{run_file[:-4]}.pkl"
        )

        run = load_run_output(run_path)
        if run["config"]["k"] != k:
            continue
        elif os.path.exists(outpath):
            logging.info(f"Run already evaluated: {run_path}")
            continue
        else:
            evaluator = MultiTokenEvaluator(
                model_name, 
                concept, 
                nucleus, # nucleus 
                nsamples, #nsamples
                msamples, #msamples
                None, #nwords
                run_path, #run_path
            )
            run_eval_output = evaluator.compute_run_eval()
            run_metadata = {
                "model_name": model_name,
                "concept": concept,
                "k": k,
                "nucleus": nucleus,
                "nsamples": nsamples,
                "msamples": msamples,
                "run_path": run_path,
                "iteration": iteration
            }
            full_run_output = run_metadata | run_eval_output
            with open(outpath, "wb") as f:
                pickle.dump(full_run_output, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Run eval exported: {run_path}")
    logging.info(f"Finished computing evals for pair {model_name}, {concept}, folder {run_output_folder}, k:{k}")

#%%#################
# Main             #
####################
def get_args():
    argparser = argparse.ArgumentParser(description='Formatting Results Tables')
    argparser.add_argument(
        "-concept",
        type=str,
        choices=["gender", "number"],
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
        "-out_folder",
        type=str,
        default="test",
        help="Directory for exporting run eval"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    #model_name = args.model
    #concept = args.concept
    #nucleus = args.nucleus
    #k = args.k
    #nsamples=args.nsamples
    #msamples=args.msamples
    #output_folder = args.out_folder
    #nruns = 3
    model_name = "gpt2-large"
    concept = "number"
    nucleus = True
    k=1
    nsamples=200
    msamples=10
    output_folder = "test_multitokeneval"
    nruns = 3
    run_output_folders = ["leace29022024"]
    
    for folder in run_output_folders:
        for i in range(nruns):
            compute_eval(
                model_name, concept, folder, k, nsamples, msamples, nucleus,
                output_folder, i
            )
    logging.info("Finished exporting all results.")


#%%
#evaluator = MultiTokenEvaluator(
#    "gpt2-large", 
#    "number", 
#    True, # nucleus 
#    2, #nsamples
#    2, #msamples
#    3, #nwords
#    os.path.join(OUT, 
#        "run_output/number/gpt2-large/leace29022024/run_leace_number_gpt2-large_2024-02-29-18:30:00_0_3.pkl"
#    ), #run_path
#)
#run_eval = evaluator.compute_run_eval()
