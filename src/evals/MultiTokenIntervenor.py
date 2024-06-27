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

from utils.lm_loaders import SUPPORTED_AR_MODELS, GPT2_LIST,\
    get_max_cxt_length
from evals.mi_intervenor_utils import prep_int_generated_data
from evals.eval_utils import load_run_Ps, load_run_output, renormalize
from data.spacy_wordlists.embedder import load_concept_token_lists
from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST
from utils.cuda_loaders import get_device
from utils.dataset_loaders import load_processed_data
from evals.mi_distributor_utils import get_run_path_info
from evals.mt_eval_runner import get_data_type

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
# Default args not worth putting into command line args
CXT_MAX_LENGTH_PCT = 0.90 # keep only context strings of length less than % of context window
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
                 int_source, # ["test", "natural"]
                 nsamples, # number of test set samples
                 msamples, # number of dev set samples for each q computation
                 run_path, # path of LEACE training run output
                 output_folder_name, # directory for exporting individual distributions
                 iteration, # iteration of the eval for this run
                 batch_size, #batch size for the word list probability computation
                 p_new_word=True, # whether to multiply the p(word | h) by p(new_word | h)
                 exist_ok=False, # DEBUG ONLY: export directory exist_ok
                 torch_dtype=torch.float32, # torch data type to use for eval
                ):

        self.model_name = model_name
        self.concept = concept
        self.int_source = int_source
        self.nsamples = nsamples
        self.msamples = msamples
        self.run_path = run_path
        self.output_folder_name = output_folder_name
        self.iteration = iteration
        self.batch_size = batch_size
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
        #self.outdir = get_eval_directory(
        #    "int_eval", run_path, concept, model_name, 
        #    self.proj_source, output_folder_name, 
        #    int_source, iteration
        #)
        #os.makedirs(self.outdir, exist_ok=self.exist_ok)
        #logging.info(f"Created outdir: {self.outdir}")

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

        # Load generated samples
        nucleus = get_nucleus_arg(self.proj_source)
        self.gen_all_hs = prep_int_generated_data(
            model_name, concept, nucleus, self.torch_dtype,
            MAX_N_ALL_HS
        )

        # Load test set samples
        #self.cxt_toks_train, self.y_train = run["cxt_toks_train"], run["y_train"]
        self.cxt_toks_test, self.ys_test = run["cxt_toks_test"], run["y_test"]
        self.facts_test, self.foils_test = run["facts_test"], run["foils_test"]
        self.hs_val, self.y_val = run["X_val"], run["y_val"]

        # Select samples to use 
        # TODO: should probably remove "natural" as an option here and just do "test"
        self.test_samples, self.l0_hs, self.l1_hs = self.load_intervention_data(
            int_source
        )

        # Delete cxts for memory
        self.cxt_toks_test, self.ys_test = None, None
        self.facts_test, self.foils_test = None, None
        self.hs_val, self.y_val = None, None
        

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
    def load_intervention_data(self, int_source):
        logging.info(f"Loading data with int_source: {int_source}")
        if int_source == "test":
            test_samples = self.process_gen_test_samples()
            l0_hs = torch.tensor(
                self.hs_val[self.y_val == 0], dtype=self.torch_dtype
            )
            l1_hs = torch.tensor(
                self.hs_val[self.y_val == 1], dtype=self.torch_dtype
            )
        elif int_source == "natural":
            test_samples, l0_hs, l1_hs = self.load_and_prep_natural_data(
                self.model_name, self.concept
            )
        else:
            raise NotImplementedError(f"int_source: {int_source} not supported")
        return test_samples, l0_hs, l1_hs

    #########################################
    # Natural Data Processing               #
    #########################################
    def load_and_prep_natural_data(self, model_name, concept, pad_token=-1):
        X, U, y, facts, foils, cxt_toks = load_processed_data(
            concept, model_name
        )

        nobs = X.shape[0]
        idx = np.arange(0, nobs)
        np.random.shuffle(idx)

        # test set samples
        y_test = y[idx[:self.nsamples]]
        facts_test_pad = facts[idx[:self.nsamples]]
        facts_test = [x[x!=pad_token] for x in facts_test_pad]
        foils_test_pad = foils[idx[:self.nsamples]]
        foils_test = [x[x!=pad_token] for x in foils_test_pad]
        cxt_toks_test = torch.from_numpy(cxt_toks[idx[:self.nsamples]])
        test_samples = [
            x for x in zip(cxt_toks_test, y_test, facts_test, foils_test)
        ]
        
        # dev hs
        hs_val = X[idx[self.nsamples:self.nsamples + MAX_N_ALL_HS]]
        ys_val = y[idx[self.nsamples:self.nsamples + MAX_N_ALL_HS]]
        l0_hs = torch.tensor(hs_val[ys_val == 0], dtype=self.torch_dtype)
        l1_hs = torch.tensor(hs_val[ys_val == 1], dtype=self.torch_dtype)
        return test_samples, l0_hs, l1_hs

    #########################################
    # Generated Data Processing             #
    #########################################
    @staticmethod
    def filter_gens_by_concept(cxt_toks, facts, foils, ys):
        """ filters samples with y in [0,1,2] to y in [0,1] 
        """
        concept_filter = np.isin(ys, [0,1])
        c_cxt_toks = cxt_toks[concept_filter]
        c_facts = facts[concept_filter]
        c_foils = foils[concept_filter]
        c_ys = ys[concept_filter]
        return c_cxt_toks, c_facts, c_foils, c_ys

    @staticmethod
    def filter_gens_by_length(cxt_toks, facts, foils, ys, 
        model_name, cxt_max_length_pct):
        max_cxt_length = get_max_cxt_length(model_name)
        cxt_size_limit = int(max_cxt_length * cxt_max_length_pct)
        cxt_toks_notpad_count = (cxt_toks != -1).sum(1)
        cxt_toks_notpad_filter = cxt_toks_notpad_count < cxt_size_limit
        cxt_toks_sub = cxt_toks[cxt_toks_notpad_filter]
        facts_sub = facts[cxt_toks_notpad_filter]
        foils_sub = foils[cxt_toks_notpad_filter]
        ys_sub = ys[cxt_toks_notpad_filter]
        return cxt_toks_sub, facts_sub, foils_sub, ys_sub

    @staticmethod
    def subsample_gens(cxt_toks, facts, foils, ys, nsamples, pad_token=-1):
        idx = np.arange(ys.shape[0])
        np.random.shuffle(idx)
        n_cxt_toks_test = torch.from_numpy(cxt_toks[idx[:nsamples]])
        n_y_test = ys[idx[:nsamples]]
        n_facts_test_pad = facts[idx[:nsamples]]
        n_facts_test = [x[x!=pad_token] for x in n_facts_test_pad]
        n_foils_test_pad = foils[idx[:nsamples]]
        n_foils_test = [x[x!=pad_token] for x in n_foils_test_pad]

        test_nsamples = [
            x for x in zip(n_cxt_toks_test, n_y_test, n_facts_test, n_foils_test)
        ]
        return test_nsamples
        
    def process_gen_test_samples(self):
        cxt_toks, facts, foils, ys = self.filter_gens_by_concept(
            self.cxt_toks_test, self.facts_test, 
            self.foils_test, self.ys_test
        )
        cxt_toks, facts, foils, ys = self.filter_gens_by_length(
            cxt_toks, facts, foils, ys, 
            self.model_name, CXT_MAX_LENGTH_PCT
        )
        test_samples = self.subsample_gens(
            cxt_toks, facts, foils, ys, 
            self.nsamples
        )
        return test_samples
    
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
        # corr erased p(x|hbot): 1x2
        cxt_hbot = cxt_hidden_state @ self.I_P
        p_x_mid_hbot = self.compute_pxhs(
            cxt_hbot, pkv_batch_hs, batch_tokens
        )
        # qbot(x|hbot): msamples x 2
        q_x_mid_hbot = self.compute_qxhs(
            cxt_hidden_state, pkv_batch_hs, "hpar", 
            batch_tokens
        )
        # p(x | hbot, do(C=0)): msamples x 2
        p_x_mid_do0 = self.compute_do_qxhs(
            "c=0", cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        # p(x | hbot, do(C=1)): msamples x 2 
        p_x_mid_do1 = self.compute_do_qxhs(
            "c=1", cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        return (
            p_x_mid_h.squeeze(0),
            p_x_mid_hbot.squeeze(0),
            q_x_mid_hbot.mean(0),
            p_x_mid_do0.mean(0),
            p_x_mid_do1.mean(0)
        )
            
    @staticmethod
    def score_interventions(y: int, 
        p_x_mid_h: torch.tensor, p_x_mid_hbot: torch.tensor,
        q_x_mid_hbot: torch.tensor, p_x_mid_do0: torch.tensor, 
        p_x_mid_do1: torch.tensor
    ) -> dict:
        """ Scores word probabilities. 
        orig, erased, do_c0 and d0_c1 have shape (2)
        where position 0 is the probability of the fact word
        and position 1 is the probability of the foil word
        """
        # fact > foil
        base_correct = p_x_mid_h[0] > p_x_mid_h[1]
        # fact > foil
        corr_erased_correct = p_x_mid_hbot[0] > p_x_mid_hbot[1]
        # fact > foil
        erased_correct = q_x_mid_hbot[0] > q_x_mid_hbot[1]
        if y == 0:
            # fact is c=0, foil is c=1
            do_c0_correct = p_x_mid_do0[0] > p_x_mid_do0[1] # c=0 > c=1
            do_c1_correct = p_x_mid_do1[1] > p_x_mid_do1[0] # c=1 > c=0
        elif y == 1:
            # fact is c=1, foil is c=0
            do_c0_correct = p_x_mid_do0[1] > p_x_mid_do0[0] # c=0 > c=1
            do_c1_correct = p_x_mid_do1[0] > p_x_mid_do1[1] # c=1 > c=0
        else:
            raise ValueError(f"Incorrect label y: {y}")
        return dict(
            y = y,
            base_correct = base_correct.item(),
            corr_erased_correct = corr_erased_correct.item(),
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
        p_x_mid_h, p_x_mid_hbot, q_x_mid_hbot, p_x_mid_do0, p_x_mid_do1 = self.compute_all_intervention_distribs(
            cxt_hidden_state, pkv_batch_hs, batch_tokens
        )
        scores = self.score_interventions(
            y, p_x_mid_h, p_x_mid_hbot, q_x_mid_hbot, 
            p_x_mid_do0, p_x_mid_do1
        )
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

    def compute_intervention_eval(self, pad_token=-1):
        run_metadata = {
            "model_name": self.model_name,
            "concept": self.concept,
            "proj_source": self.proj_source,
            "int_source": self.int_source,
            "eval_name": self.output_folder_name,
            "run_path": self.run_path,
            "iteration": self.iteration,
        }
        scores = []
        for i, (cxt_pad, y, fact, foil) in enumerate(tqdm(self.test_samples)):

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
                    
            scores.append(run_metadata | sample_score)

        return scores

# %%
def compute_int(
    model_name, concept, int_source, nsamples, msamples, 
    run_path, output_folder_name, iteration, 
    batch_size, p_new_word=True, exist_ok=False, 
    torch_dtype=torch.float16):

    # export directory creation
    run_output_dir, run_id = get_run_path_info(run_path)

    actual_outdir = os.path.join(RESULTS, f"int/{output_folder_name}")
    os.makedirs(actual_outdir, exist_ok=True)

    # intervention scoring
    intervenor = MultiTokenIntervenor(
        model_name,
        concept,
        int_source,
        nsamples,
        msamples,
        run_path,
        output_folder_name,
        iteration,
        batch_size,
        p_new_word=p_new_word,
        exist_ok=exist_ok,
        torch_dtype=torch_dtype
    )

    scores = intervenor.compute_intervention_eval()
    scores_df = pd.DataFrame(scores).groupby(
        ["model_name", "concept", "proj_source", "int_source", "eval_name", "run_path", "iteration", "y"]
    ).mean().reset_index()

    # export
    outpath = os.path.join(
        actual_outdir, 
        f"intres_{model_name}_{concept}_{intervenor.proj_source}_"
        f"{run_output_dir}_{output_folder_name}_run_{run_id}_"
        f"{int_source}_evaliter_{iteration}.csv"
    )
    scores_df.to_csv(outpath)
    logging.info(f"Run intervention scores exported: {outpath}")
        

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
        "-int_source",
        type=str,
        choices=["test", "natural"],
        help=("Which samples to use for eval."
             "test: test set samples from LEACE train/val/test data, filtered to concept values"
             "natural: natural data samples")
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
        "-batch_size",
        type=int,
        default=64,
        help="Batch size for word probability computation"
    )
    argparser.add_argument(
        "-torch_dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        help="Data type to cast all tensors to during evaluation",
        default="float16"
    )
    return argparser.parse_args()

if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    concept = args.concept
    int_source = args.int_source
    nsamples= args.nsamples
    msamples = args.msamples
    run_path = args.run_path
    output_folder_name = args.out_folder
    batch_size = args.batch_size
    nruns = 3
    p_new_word = True
    exist_ok = False
    torch_dtype = get_data_type(args.torch_dtype)

    #model_name = "gpt2-large"
    #concept = "number"
    #int_source = "test"
    #nsamples= 5
    #msamples = 10
    #run_path = os.path.join(
    #    OUT,
    #    "run_output/june2/number/gpt2-large/gen_ancestral_all/run_leace_number_gpt2-large_gen_ancestral_all_2024-06-02-15:54:03_0_3.pkl"
    #)
    #output_folder_name = "int_test"
    #batch_size = 64
    #nruns = 1
    #p_new_word = True
    #exist_ok = True
    #torch_dtype = torch.float16

    for i in range(nruns):
        logging.info(f"Computing intervention eval number {i}")
        compute_int(
            model_name, concept, int_source, nsamples, msamples, 
            run_path, output_folder_name, i, 
            batch_size, p_new_word=p_new_word, exist_ok=exist_ok, 
            torch_dtype=torch_dtype
        )
    logging.info("Finished exporting all corr eval results.")