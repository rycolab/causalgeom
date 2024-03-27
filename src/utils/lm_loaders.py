import warnings
import logging
import os
import sys
import coloredlogs
import torch
import pickle
import numpy as np
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import BertTokenizerFast, BertForMaskedLM
from transformers import CamembertForMaskedLM, CamembertTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('./src/')

from paths import HF_CACHE, DATASETS


coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

GPT2_LIST = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "gpt2-base-french", "gpt2-french-small"]
BERT_LIST = ["bert-base-uncased", "camembert-base"]
SUPPORTED_AR_MODELS = GPT2_LIST + ["llama2"]

#%%

def get_tokenizer(model_name, token=None):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name, model_max_length=512
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif model_name == "gpt2-base-french":
        tokenizer = GPT2TokenizerFast.from_pretrained(
            "ClassCat/gpt2-base-french", model_max_length=512
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif model_name == "gpt2-french-small":
        tokenizer = GPT2TokenizerFast.from_pretrained(
            "dbddv01/gpt2-french-small", model_max_length=512
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif model_name == "bert-base-uncased":
        return BertTokenizerFast.from_pretrained(
            model_name, model_max_length=512
        )
    elif model_name == "camembert-base":
        return CamembertTokenizer.from_pretrained(
            model_name, model_max_length=512
        )
    elif model_name in ["llama2", "llama2_32bit"]:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", 
            token=token, model_max_length=4096
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise ValueError(f"Model name {model_name} not supported")


def get_model(model_name, token=None, device="cpu"):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        model = GPT2LMHeadModel.from_pretrained(
            model_name, 
            cache_dir=HF_CACHE,
            device_map="auto"
        )
        if torch.cuda.device_count() == 1:
            return model.to(device)
        else:
            return model
    elif model_name == "gpt2-base-french":
        return GPT2LMHeadModel.from_pretrained(
            "ClassCat/gpt2-base-french", 
            cache_dir=HF_CACHE
        ).to(device)
    elif model_name == "gpt2-french-small":
        return GPT2LMHeadModel.from_pretrained(
            "dbddv01/gpt2-french-small", 
            cache_dir=HF_CACHE
        ).to(device)
    elif model_name == "bert-base-uncased":
        return BertForMaskedLM.from_pretrained(
            model_name, 
            cache_dir=HF_CACHE, 
            is_decoder=False
        ).to(device)
    elif model_name == "camembert-base":
        return CamembertForMaskedLM.from_pretrained(
            model_name, 
            cache_dir=HF_CACHE, 
            #is_decoder=False
        ).to(device)
    elif model_name == "llama2":
        return AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=HF_CACHE,
            token=token,
            device_map="auto", 
            #context_length=4096
            torch_dtype=torch.float16
        )
    elif model_name == "llama2_32bit":
        return AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=HF_CACHE,
            token=token,
            device_map="auto", 
            #context_length=4096
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")


def get_V(model_name, model=None):
    if model is None:
        model = get_model(model_name)

    if model_name in GPT2_LIST:
        return model.lm_head.weight.detach().cpu().numpy()
    elif model_name == "bert-base-uncased":
        word_embeddings = model.bert.embeddings.word_embeddings.weight
        bias = model.cls.predictions.decoder.bias
        return torch.cat(
            (word_embeddings, bias.view(-1, 1)), dim=1).detach().numpy()
    elif model_name == "camembert-base":
        # i checked that the decoder linear layer weights are tied to the embeddings
        # that said there is no bias at the embedding level, had to fetch the 
        # decoder bias.
        word_embeddings = model.lm_head.decoder.weight
        bias = model.lm_head.decoder.bias
        return torch.cat(
            (word_embeddings, bias.view(-1, 1)), dim=1).detach().numpy()
    elif model_name == "llama2":
        return model.lm_head.weight.detach().cpu().numpy()
    else:
        raise ValueError(f"Model name {model_name} not supported")

#def get_concept_name(model_name):
#    """ Model name to concept mapper, for now it's one to one so this
#    can exist. """
#    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "bert-base-uncased"]:
#        return "number"
#    elif model_name in ["gpt2-base-french", "camembert-base"]:
#        return "gender"
#    else:
#        raise ValueError(f"No model to concept mapping")

