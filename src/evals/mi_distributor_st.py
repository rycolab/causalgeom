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


from evals.mi_distributor_utils import prep_generated_data, compute_inner_loop_qxhs,\
    compute_batch_inner_loop_qxhs
from utils.lm_loaders import SUPPORTED_AR_MODELS
from evals.eval_utils import load_run_Ps, load_run_output, load_model_eval,\
    renormalize
#from data.filter_generations import load_generated_hs_wff
#from data.data_utils import filter_hs_w_ys, sample_filtered_hs
from utils.lm_loaders import get_model, get_tokenizer, get_V
from utils.cuda_loaders import get_device

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%
class SingleTokenDistributor:
    
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

        # Load Run data
        run = load_run_output(run_path)
        
        # Load model
        self.device = get_device()
        self.model = get_model(model_name, device=self.device)
        self.tokenizer = get_tokenizer(model_name)

        # Load with device
        P, I_P, _ = load_run_Ps(run_path)
        self.P = torch.tensor(P, dtype=torch.float32).to(self.device) 
        self.I_P = torch.tensor(I_P, dtype=torch.float32).to(self.device)

        # test set version of the eval
        self.V = get_V(model_name, model=self.model, numpy_cpu=False)
        _, self.l0_tl, self.l1_tl = load_model_eval(
            model_name, concept, single_token=True
        )

        if self.nwords is not None:
            logging.warn(f"Applied nwords={self.nwords}, intended for DEBUGGING ONLY")
            self.l0_tl = self.l0_tl[:self.nwords]
            self.l1_tl = self.l1_tl[:self.nwords]

        # CEBaB prompts
        #self.prompt_set = self.load_suffixes(concept)
        #self.prompt_end_space = model_name == "llama2"

        #p_x = load_p_x(MODEL_NAME, NUCLEUS)
        self.p_c, self.gen_l0_hs, self.gen_l1_hs, gen_all_hs = prep_generated_data(
            model_name, concept, nucleus
        )
        self.gen_all_hs = torch.tensor(gen_all_hs, dtype=torch.float32)

        self.X_dev, self.y_dev, self.cxt_toks_dev = \
            run["X_val"], run["y_val"], run["cxt_toks_val"]
        self.X_test, self.y_test, self.cxt_toks_test = \
            run["X_test"], run["y_test"], run["cxt_toks_test"]
        #self.y_dev, self.cxt_toks_dev = run["y_val"], run["cxt_toks_val"]
        #self.y_test, self.cxt_toks_test = run["y_test"], run["cxt_toks_test"]

        #TODO: RIGHT NOW THIS IS THE ONLY WAY, NEED TO ENABLE GENERATED
        source = "natural"
        #NOTE:CHANGED
        self.l0_hs, self.l1_hs, self.l0_cxt_toks, self.l1_cxt_toks = self.get_concept_eval_contexts(source)

        self.new_word_tokens = self.get_new_word_tokens(model_name) 

    #########################################
    # Tokenizer specific new word tokens    #
    #########################################
    #NOTE: all of these are duplicated move to utils
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
    #NOTE: DUPLICATED BUT CHANGED!!!!!!!!!!
    def get_concept_eval_contexts(self, source):
        if source == "generated":
            raise NotImplementedError("This is currently not working cuz I save h's not tokens")
            #l0_hs_n, l1_hs_n = sample_filtered_hs(evaluator.l0_hs_wff, evaluator.l0_hs_wff, evaluator.nsamples)
        elif source == "natural":
            l0_hs = torch.tensor(self.X_test[self.y_test==0], dtype=torch.float32)
            l1_hs = torch.tensor(self.X_test[self.y_test==1], dtype=torch.float32)

            l0_cxt_toks = torch.tensor(self.cxt_toks_test[self.y_test==0])
            l1_cxt_toks = torch.tensor(self.cxt_toks_test[self.y_test==1])
            return l0_hs, l1_hs, l0_cxt_toks, l1_cxt_toks
        else: 
            raise ValueError(f"Evaluation context source {source} invalid")

    def sample_filtered_contexts(self):
        l0_len = self.l0_hs.shape[0]
        l1_len = self.l1_hs.shape[0]
        ratio = l1_len / l0_len
        l0_ind = np.arange(l0_len)
        l1_ind = np.arange(l1_len)
        np.random.shuffle(l0_ind)
        np.random.shuffle(l1_ind)
        if ratio > 1:
            l0_hs_sample = self.l0_hs[l0_ind[:self.nsamples]].to(self.device)
            l1_hs_sample = self.l1_hs[l1_ind[:int((self.nsamples*ratio))]].to(self.device)
            l0_cxt_toks_sample = self.l0_cxt_toks[l0_ind[:self.nsamples]].to(self.device)
            l1_cxt_toks_sample = self.l1_cxt_toks[l1_ind[:int((self.nsamples*ratio))]].to(self.device)
        else:
            ratio = l0_len / l1_len
            l0_hs_sample = self.l0_hs[l0_ind[:int((self.nsamples*ratio))]].to(self.device)
            l1_hs_sample = self.l1_hs[l1_ind[:self.nsamples]].to(self.device)
            l0_cxt_toks_sample = self.l0_cxt_toks[l0_ind[:int((self.nsamples*ratio))]].to(self.device)
            l1_cxt_toks_sample = self.l1_cxt_toks[l1_ind[:self.nsamples]].to(self.device)

        l0_h_cxt_tok_pairs = [*zip(l0_hs_sample, l0_cxt_toks_sample)]
        l1_h_cxt_tok_pairs = [*zip(l1_hs_sample, l1_cxt_toks_sample)]
        return l0_h_cxt_tok_pairs, l1_h_cxt_tok_pairs 

    #########################################
    # Probability computations              #
    #########################################
    # single h version
    def sample_gen_all_hs_singleh(self):
        idx = np.arange(0, self.gen_all_hs.shape[0])
        np.random.shuffle(idx)
        sampled_hs = self.gen_all_hs[idx[:self.msamples]].to(self.device)
        return sampled_hs

    def compute_p_word_handler(self, method, h):
        if method in ["hbot", "hpar"]:
            sample_hs = self.sample_gen_all_hs_singleh()
            return compute_inner_loop_qxhs(
                method, h, sample_hs, self.P, self.I_P, 
                self.V, processor=None
            ).mean(axis=0)
        elif method == "h":
            logits = self.V @ h
            pxh = softmax(logits.cpu())
            return pxh
        else:
            raise ValueError(f"Incorrect method argument {method}")

    # batch hs version
    def sample_gen_all_hs_batched(self, nH):
        nmH = nH[:, None, :].repeat(1, self.msamples, 1)
        idx = np.random.randint(
            0, self.gen_all_hs.shape[0], nH.shape[0]*self.msamples
        )
        other_hs = self.gen_all_hs[idx].to(self.device)
        other_hs_view = other_hs.view((nH.shape[0], self.msamples, nH.shape[1]))
        return nmH, other_hs_view

    def compute_p_word_batch_handler(self, method, nH):
        if method in ["hbot", "hpar"]:
            nmH, other_nmH = self.sample_gen_all_hs_batched(nH)
            qxhs = compute_batch_inner_loop_qxhs(
                method, nmH, other_nmH, 
                self.P, self.I_P, self.V
            )
            return qxhs.mean(axis=1)
        elif method == "h":
            logits = nH @ self.V.T
            pxh = softmax(logits.cpu(), axis=1)
            return pxh
        else:
            raise ValueError(f"Incorrect method argument {method}")

    def compute_tl_new_word_probs(self, token_list, cxt, method):
        cxt_np = cxt.clone().cpu().numpy()
        cxt_plus_tl = [np.append(cxt_np, x).tolist() for x in token_list]
        cxt_plus_tl_batched = torch.tensor(
            list(zip_longest(*cxt_plus_tl, fillvalue=self.tokenizer.pad_token_id))
        ).T.to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=cxt_plus_tl_batched, 
                #attention_mask=attention_mask, 
                labels=cxt_plus_tl_batched,
                output_hidden_states=True,
                #past_key_values= cxt_pkv
            )

        batch_hidden_states = output["hidden_states"][-1]#.cpu().numpy()
        next_word_hidden_states = batch_hidden_states[:,-1,:]

        new_word_probs = self.compute_p_word_batch_handler(
            method, next_word_hidden_states
        )
        return new_word_probs

    def compute_all_word_probs(self, h, cxt, method):
        # step 1: compute prob using the first h
        # step 2: compute next h with all tokens --> compute prob again
        # step 3: multiply and return

        pxh = self.compute_p_word_handler(method, h)

        l0_word_probs = pxh[self.l0_tl]
        l1_word_probs = pxh[self.l1_tl]

        l0_next_word_probs = self.compute_tl_new_word_probs(
            self.l0_tl, cxt, method
        )
        l0_new_word_probs = l0_next_word_probs[:, self.new_word_tokens].sum(axis=1)

        l1_next_word_probs = self.compute_tl_new_word_probs(
            self.l1_tl, cxt, method
        )
        l1_new_word_probs = l1_next_word_probs[:, self.new_word_tokens].sum(axis=1)

        l0_full_word_probs = l0_word_probs * l0_new_word_probs
        l1_full_word_probs = l1_word_probs * l1_new_word_probs
        return l0_full_word_probs, l1_full_word_probs


    def compute_lemma_probs(self, lemma_samples, method, pad_token=-1):
        l0_probs, l1_probs = [], []
        for i, (h, cxt_pad) in enumerate(tqdm(lemma_samples)):

            cxt = cxt_pad[cxt_pad != pad_token]

            logging.info(f"---New eval context: {self.tokenizer.decode(cxt)}---")

            l0_word_probs, l1_word_probs = self.compute_all_word_probs(
                h, cxt, method
            )

            export_path = os.path.join(self.outdir, f"word_probs_sample_{i}.pkl")
                
            with open(export_path, 'wb') as f:
                pickle.dump((l0_word_probs, l1_word_probs), f, protocol=pickle.HIGHEST_PROTOCOL)
                        
            l0_probs.append(l0_word_probs)
            l1_probs.append(l1_word_probs)

        return np.vstack(l0_probs), np.vstack(l1_probs)

    def compute_pxs(self, htype):
        assert self.nsamples is not None
        l0_inputs, l1_inputs = self.sample_filtered_contexts()
        if htype == "l0_cxt_qxhs_par": 
            l0_cxt_qxhs_par = self.compute_lemma_probs(l0_inputs, "hbot")
            return l0_cxt_qxhs_par
        elif htype == "l1_cxt_qxhs_par":
            l1_cxt_qxhs_par = self.compute_lemma_probs(l1_inputs, "hbot")
            return l1_cxt_qxhs_par
        elif htype == "l0_cxt_qxhs_bot":
            l0_cxt_qxhs_bot = self.compute_lemma_probs(l0_inputs, "hpar")
            return l0_cxt_qxhs_bot
        elif htype == "l1_cxt_qxhs_bot":
            l1_cxt_qxhs_bot = self.compute_lemma_probs(l1_inputs, "hpar")
            return l1_cxt_qxhs_bot
        elif htype == "l0_cxt_pxhs":
            l0_cxt_pxhs = self.compute_lemma_probs(l0_inputs, "h")
            return l0_cxt_pxhs
        elif htype == "l1_cxt_pxhs":
            l1_cxt_pxhs = self.compute_lemma_probs(l1_inputs, "h")
            return l1_cxt_pxhs
        else:
            raise ValueError(f"Incorrect htype: {htype}")   

# %%
def compute_eval(model_name, concept, run_path,
    nsamples, msamples, nwords, nucleus, output_folder, htype, iteration):
    #rundir = os.path.join(
    #    OUT, f"run_output/{concept}/{model_name}/{run_output_folder}"
    #)

    #TODO: make this nicer, problem is output folder.
    rundir = os.path.dirname(run_path)
    rundir_name = os.path.basename(rundir)

    run_id = run_path[-27:-4]
    outdir = os.path.join(
        os.path.dirname(rundir), 
        f"st_eval_{rundir_name}/{output_folder}/run_{run_id}/nuc_{nucleus}/evaliter_{iteration}/h_distribs/{htype}"
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
    evaluator = SingleTokenDistributor(
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
        choices=["number", "gender"],
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
    #concept = "number"
    #nucleus = True
    #k=1
    #nsamples=3
    #msamples=3
    #nwords=10
    #output_folder = "test"
    #nruns = 1
    #run_path="out/run_output/number/gpt2-large/leace26032024/run_leace_number_gpt2-large_2024-03-26-19:55:11_0_3.pkl"
    #htype = "l0_cxt_qxhs_bot"
    
    for i in range(nruns):
        logging.info(f"Computing eval number {i}")
        compute_eval(
            model_name, concept, run_path, nsamples, msamples, 
            nwords, nucleus, output_folder, htype, i
        )
    logging.info("Finished exporting all results.")

