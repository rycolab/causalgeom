#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse

import numpy as np
import torch
import random 
from itertools import zip_longest
from concept_erasure import LeaceEraser

sys.path.append('..')

from data.filter_generations import load_filtered_generations
from utils.dataset_loaders import load_processed_data
from evals.mi_distributor_utils import get_nucleus_arg

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
class ProjectionTrainer:
    
    def __init__(self, 
                 model_name, # name of AR model 
                 concept, # concept name
                 source, # ["gen_ancestral_all", "gen_nucleus_all"]
                 train_nsamples, # number of train samples
                 val_nsamples, # number of val samples
                 test_nsamples, # number of test samples
                 train_share=0.6, #gender natural_concept specific arg
                 val_share=0.1, #gender natural_concept specific arg
        ):
        self.model_name = model_name
        self.concept = concept
        self.source = source
        self.nucleus = get_nucleus_arg(source)
        self.train_nsamples = train_nsamples
        self.val_nsamples = val_nsamples
        self.test_nsamples = test_nsamples
        self.train_share = train_share
        self.val_share = val_share
    
    #########################################
    # Generated Data Handling               #
    #########################################
    @staticmethod
    def load_format_generated_data(model_name, concept, nucleus):
        l0_gens, l1_gens, other_gens = load_filtered_generations(
            model_name, concept, nucleus=nucleus
        )

        # (h, fact, foil, cxt_tok, y)
        l0_gens = [(h, fact, foil, cxt_tok, 0) for 
                        h, fact, foil, cxt_tok in l0_gens]
        l1_gens = [(h, fact, foil, cxt_tok, 1) for 
                        h, fact, foil, cxt_tok in l1_gens]
        other_gens = [(h, [], [], cxt_tok, 2) for 
                        h, cxt_tok in other_gens]

        logging.info(
            f"Generated samples loaded:\n"
            f"- c=0 {len(l0_gens)} \n"
            f"- c=1 {len(l1_gens)} \n"
            f"- c=n/a: {len(other_gens)}"
        )
        return l0_gens + l1_gens + other_gens

    @staticmethod
    def train_val_test_split(all_gens, train_nsamples, 
        val_nsamples, test_nsamples):
        """ Splits all generated samples into train, val and test sets"""
        all_samples = random.sample(
            all_gens, train_nsamples+val_nsamples+test_nsamples 
        )
        val_lastind = train_nsamples + val_nsamples
        train_samples = all_samples[:train_nsamples]
        val_samples = all_samples[train_nsamples:val_lastind]
        test_samples = all_samples[val_lastind:]
        return train_samples, val_samples, test_samples

    @staticmethod
    def split_samples(samples):
        """ split into X, facts, foils, cxt_toks, ys """
        X = np.array([h for h,_,_,_,_ in samples])
        raw_facts = [fact for _,fact,_,_,_ in samples]
        facts = np.array(list(zip_longest(*raw_facts, fillvalue=-1))).T
        raw_foils = [foil for _,_,foil,_,_ in samples]
        foils = np.array(list(zip_longest(*raw_foils, fillvalue=-1))).T
        raw_cxt_toks = [cxt_tok for _,_,_,cxt_tok,_ in samples]
        cxt_toks = np.array(list(zip_longest(*raw_cxt_toks, fillvalue=-1))).T
        y = np.array([y for _,_,_,_,y in samples])
        return X, facts, foils, cxt_toks, y
    
    def load_generated_data(self):
        all_gens = self.load_format_generated_data(
            self.model_name, self.concept, self.nucleus
        )
        train_samples, val_samples, test_samples = self.train_val_test_split(
            all_gens, self.train_nsamples, self.val_nsamples, self.test_nsamples
        )
        X_train, facts_train, foils_train, cxt_toks_train, y_train \
            = self.split_samples(train_samples)
        X_val, facts_val, foils_val, cxt_toks_val, y_val \
            = self.split_samples(val_samples)
        X_test, facts_test, foils_test, cxt_toks_test, y_test \
            = self.split_samples(test_samples)
        
        return {
            "X_train": X_train, 
            "y_train": y_train,
            "facts_train": facts_train,
            "foils_train": foils_train,
            "cxt_toks_train": cxt_toks_train,
            "X_val": X_val, 
            "y_val": y_val,
            "facts_val": facts_val,
            "foils_val": foils_val,
            "cxt_toks_val": cxt_toks_val,
            "X_test": X_test, 
            "y_test": y_test,
            "facts_test": facts_test,
            "foils_test": foils_test,
            "cxt_toks_test": cxt_toks_test,
        }
        
    #########################################
    # Natural Data Handling                 #
    #########################################
    @staticmethod
    def get_data_indices(nobs, concept, train_obs, val_obs, test_obs,
            train_share, val_share):
        """ applies either nsamples or share depending on concept,
        because linzen data is too big to use fully, vs. 
        gender data is small enough to use entirely """
        idx = np.arange(0, nobs)
        np.random.shuffle(idx)

        if concept == "number":
            train_lastind = train_obs
            val_lastind = train_lastind + val_obs
            test_lastind = val_lastind + test_obs
        elif concept == "gender":
            train_lastind = int(nobs*train_share)
            val_lastind = int(nobs*(cfg["train_share"] + val_share))
            test_lastind = nobs
        else:
            raise ValueError("Concept value not supported.")
        return idx[:train_lastind], idx[train_lastind:val_lastind], idx[val_lastind:test_lastind]

    @staticmethod
    def create_run_datasets(X, U, y, facts, foils, cxt_toks, 
        idx_train, idx_val, idx_test):
        """ Train val test split of X, U, y, facts, foils, cxt_toks """
        X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
        #U_train, U_val, U_test = U[idx_train], U[idx_val], U[idx_test]
        y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        facts_train, facts_val, facts_test = facts[idx_train], facts[idx_val], facts[idx_test]
        foils_train, foils_val, foils_test = foils[idx_train], foils[idx_val], foils[idx_test]
        cxt_toks_train, cxt_toks_val, cxt_toks_test = cxt_toks[idx_train], cxt_toks[idx_val], cxt_toks[idx_test]
        
        return {
            "X_train": X_train, 
            "y_train": y_train,
            "facts_train": facts_train,
            "foils_train": foils_train,
            "cxt_toks_train": cxt_toks_train,
            "X_val": X_val, 
            "y_val": y_val,
            "facts_val": facts_val,
            "foils_val": foils_val,
            "cxt_toks_val": cxt_toks_val,
            "X_test": X_test, 
            "y_test": y_test,
            "facts_test": facts_test,
            "foils_test": foils_test,
            "cxt_toks_test": cxt_toks_test,
        }
        
    def load_natural_data(self):
        """ Applies different preprocessing steps for natural data,
        outputting dict of data components split into train, val, test"""
        if self.concept in ["number", "gender"]:    
            X, U, y, facts, foils, cxt_toks = load_processed_data(
                self.concept, self.model_name
            )
            idx_train, idx_val, idx_test = self.get_data_indices(
                X.shape[0], self.concept, 
                self.train_nsamples, self.val_nsamples, self.test_nsamples,
                self.train_share, self.val_share
            )
            run_data = self.create_run_datasets(
                X, U, y, facts, foils, cxt_toks, idx_train, idx_val, idx_test
            )
        elif self.concept in ["food", "ambiance", "service", "noise"]:
            run_data = load_processed_data(self.concept, self.model_name)
        else:
            raise NotImplementedError(f"Concept {self.concept} not supported")
        return run_data
    
    ##########################################
    # LEACE training                         #
    ##########################################
    @staticmethod
    def compute_leace_affine(X_train, y_train):
        X_train_f32 = X_train.astype("float32", casting="safe")
        X_torch = torch.from_numpy(X_train_f32)
        y_torch = torch.from_numpy(y_train)

        eraser = LeaceEraser.fit(X_torch, y_torch)

        P = (eraser.proj_right.mH @ eraser.proj_left.mH).numpy()
        I_P = np.eye(X_train.shape[1]) - P
        bias = eraser.bias.numpy()
        return P, I_P, bias
    
    ##########################################
    # Main Training function                 # 
    ##########################################
    def load_run_data(self):
        if self.source in ["gen_ancestral_all", "gen_nucleus_all"]:
            logging.info(f"Loading generated samples, source: {self.source}")
            return self.load_generated_data()
        elif self.source in ["natural_concept", "natural_all"]:
            logging.info(f"Loading natural samples, source: {self.source}")
            return self.load_natural_data()
        else:
            raise ValueError(f"Incorrect source parameter: {self.source}")
            
    def train_and_format(self):
        run_data = self.load_run_data()

        X_train, X_val, X_test = \
            run_data["X_train"], run_data["X_val"], run_data["X_test"]
        y_train, y_val, y_test = \
            run_data["y_train"], run_data["y_val"], run_data["y_test"]
        facts_train, facts_val, facts_test = \
            run_data["facts_train"], run_data["facts_val"], run_data["facts_test"]
        foils_train, foils_val, foils_test = \
            run_data["foils_train"], run_data["foils_val"], run_data["foils_test"]
        cxt_toks_train, cxt_toks_val, cxt_toks_test = \
            run_data["cxt_toks_train"], run_data["cxt_toks_val"], run_data["cxt_toks_test"]

        P, I_P, bias = self.compute_leace_affine(X_train, y_train)
        output = {
            "bias": bias,
            "P": P,
            "I_P": I_P
        }
        full_results = dict(
            output=output,
            nobs_train = X_train.shape[0],
            nobs_val = X_val.shape[0],
            nobs_test = X_test.shape[0],
            #maj_acc_train=get_majority_acc(y_train),
            #maj_acc_val=get_majority_acc(y_val),
            #maj_acc_test=get_majority_acc(y_test),
            X_val=X_val,
            y_val=y_val,
            foils_val=foils_val,
            facts_val=facts_val,
            cxt_toks_val = cxt_toks_val,
            X_test=X_test,
            y_test=y_test,
            foils_test=foils_test,
            facts_test=facts_test,
            cxt_toks_test = cxt_toks_test,
        )
        return full_results