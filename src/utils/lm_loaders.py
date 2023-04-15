import warnings
import logging
import os
import sys
import coloredlogs
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from transformers import BertTokenizerFast, BertForMaskedLM

sys.path.append('./src/')
from paths import HF_CACHE

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


def get_tokenizer(model_name):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name, model_max_length=512
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif model_name == "bert-base-uncased":
        return BertTokenizerFast.from_pretrained(
            model_name, model_max_length=512
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")


def get_model(model_name):
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        return GPT2LMHeadModel.from_pretrained(
            model_name, 
            cache_dir=HF_CACHE
        )
    elif model_name == "bert-base-uncased":
        return BertForMaskedLM.from_pretrained(
            model_name, 
            cache_dir=HF_CACHE, 
            is_decoder=False
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")


def get_V(model_name, model=None):
    if model is None:
        model = get_model(model_name)

    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        return model.lm_head.weight.detach().numpy()
    elif model_name == "bert-base-uncased":
        word_embeddings = model.bert.embeddings.word_embeddings.weight
        bias = model.cls.predictions.decoder.bias
        return torch.cat(
            (word_embeddings, bias.view(-1, 1)), dim=1).detach().numpy()
    else:
        raise ValueError(f"Model name {model_name} not supported")