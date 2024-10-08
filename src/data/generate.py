#%%
import warnings
import logging
import os
import sys
import coloredlogs
import argparse
import csv
import time

import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import torch
import random 
from transformers import TopPLogitsWarper, LogitsProcessorList

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT


from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST, SUPPORTED_AR_MODELS
from utils.cuda_loaders import get_device
from evals.eval_utils import load_run_Ps
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


#%%#################
# Generating Text  #
####################
def check_eos(latest_tokens, eos_token):
    if latest_tokens.unique().shape[0] == 1 and latest_tokens.unique()[0] == eos_token:
        return False
    else:
        return True

def adjust_past_key_values(past_key_values, max_len):
    new_past_key_values = ()
    for i in range(len(past_key_values)):
        new_sub_values = ()
        for j in range(len(past_key_values[i])):
            vals = past_key_values[i][j]
            assert vals.shape[2]==max_len, "Incorrect shape past_key_values"
            new_sub_values += (vals[:,:,-(max_len - 1):,:],)
        new_past_key_values += (new_sub_values,)
    return new_past_key_values
    
def get_model_cxt_length(model_name):
    if model_name in GPT2_LIST:
        return 1024
    elif model_name == "llama2":
        return 4096
    else: 
        raise NotImplementedError(f"Model {model_name} not supported")

def generate_sequence_until_eos(model_name, model, prompt_ids, 
    batch_size, max_length, eos_token, 
    device="cpu", P=None, nucleus=False, top_p=0.9):
    tokens = prompt_ids.repeat(batch_size, 1)
    all_tokens = [tokens]
    if torch.cuda.device_count() == 1:
        tokens = tokens.to(device)
    token_h_pairs = []
    sm = torch.nn.Softmax(1)#.to(device)
    past_key_values = None
    counter = 0
    processor = LogitsProcessorList()
    processor.append(TopPLogitsWarper(0.9))
    # NOTE: I've gotten rid of the context extender in favor of just cutting it off
    # once the context length maxes out.
    while (check_eos(tokens[:,-1], eos_token) or (counter==0)) and counter < max_length:
        if past_key_values is not None and past_key_values[0][0].shape[2] == max_length:
            #copy = past_key_values[0][0].clone().detach()
            past_key_values = adjust_past_key_values(past_key_values, max_length)
        #if tokens.shape[1] > model_max_cxt_length:
        #    tokens = tokens[:,-model_max_cxt_length:]
        #else:
        #    input_tokens = tokens
        with torch.no_grad():
            output = model(
                input_ids=tokens,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
        #final_logits = output.logits[-1]#.cpu()
        past_key_values = output.past_key_values
        hs = output.hidden_states[-1][:,-1,:]#.squeeze().squeeze()#.cpu()
        if P is not None:
            hs = torch.matmul(P, hs.T).T
        logits = model.lm_head(hs)
        if model_name == "gpt2-base-french" and counter == 0 and eos_token == 0: 
            #hack cuz gpt2fr only generates eos unless i do this
            logits = logits[:,1:]
        if nucleus:
            logits = processor(tokens, logits)
        #assert (torch.isclose(final_logits[-1,:], logits, atol = 1e-05)).all().item(), "Logits not equal"
        probs = sm(logits)
        sampled_tokens = torch.multinomial(probs, 1)
        save_h_indicator = []
        for i in range(batch_size):
            if tokens[i, -1] == eos_token and counter > 0:
                sampled_tokens[i] = eos_token
                save_h_indicator.append(False)
            else:
                save_h_indicator.append(True)
                #torch.cat((zip_tokens, zip_sampled_token.unsqueeze(0)), dim=-1)
        #tokens = torch.cat((tokens, sampled_tokens), dim=-1)
        tokens = sampled_tokens
        all_tokens.append(sampled_tokens.clone().cpu())
        for h, sample_all_tokens, sample_new_token, save_h in zip(hs.cpu(), torch.hstack(all_tokens), sampled_tokens, save_h_indicator):
            if save_h:
                token_h_pairs.append((h, sample_new_token.item(), sample_all_tokens))
        counter+=1
        if counter % 100 == 0:
            logging.info(f"Number of tokens so far: {counter}") 
    try:
        all_tokens = torch.hstack(all_tokens).cpu()
    except RuntimeError:
        gen_tokens = torch.hstack(all_tokens[1:]).cpu()
        all_tokens = torch.hstack([all_tokens[0], gen_tokens])
    return all_tokens, token_h_pairs

#%%
def generate_multiple_sequences(model_name, model, tokenizer, prompt_ids, 
    batch_size, seed, export_index, outdir, max_length, device="cpu", P=None, nucleus=False, top_p=0.9):
    all_token_h_pairs = []
    all_lengths = []
    eos_token_id = tokenizer.eos_token_id
    torch.manual_seed(seed)
    #for i in range(n_generations):
    while len(all_token_h_pairs) < 1000:
        all_tokens, token_h_pairs = generate_sequence_until_eos(
            model_name, model, prompt_ids, 
            batch_size, max_length, eos_token_id, 
            device=device, P=P, nucleus=nucleus, top_p=top_p
        )
        torch.cuda.empty_cache()
        all_lengths.append(len(token_h_pairs))
        all_token_h_pairs += token_h_pairs
        for i in range(all_tokens.shape[0]):
            non_eos_text = tokenizer.decode(
                all_tokens[i, torch.nonzero(all_tokens[i,:] != eos_token_id)].T.squeeze(0)
            )
            logging.info(f"Generation {i} ---------------------- \n{non_eos_text}")
    export_path = os.path.join(outdir, f"generations_seed_{seed}_{export_index}.pkl")
    with open(export_path, 'wb') as f:
        pickle.dump(all_token_h_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Exported generation file: {export_path}")
    torch.cuda.empty_cache()

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Generate from model')
    argparser.add_argument(
        "-model",
        type=str,
        choices=SUPPORTED_AR_MODELS,
        help="Model for computing hidden states"
    )
    argparser.add_argument(
        "-nucleus",
        action="store_true",
        default=False,
        help="Whether to use nucleus sampling",
    )
    #argparser.add_argument(
    #    "-useP",
    #    action="store_true",
    #    default=False,
    #    help="Whether to load and apply a P for this set of generations",
    #)
    argparser.add_argument(
        "-export_index",
        type=int,
        default=None,
        help="Special index for this job",
    )
    return argparser.parse_args()

#%%
if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    nucleus = args.nucleus
    export_index = args.export_index

    #model_name = "gpt2-large"
    #nucleus=True
    #export_index = 0
    
    #if export_index is None:
    #    export_index=random.randint(1,1000000)
    
    batch_size = 3
    useP=False
    I_P=None


    device = get_device()
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, device=device)

    if model_name in GPT2_LIST:    
        prompt_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids #.to(device)
    elif model_name == "llama2":
        prompt_ids = torch.tensor([tokenizer.bos_token_id])
    else: 
        raise NotImplementedError(f"Model {model_name} not supported")
    
    if nucleus:
        outdir = os.path.join(OUT, f"generated_text_nucleus/{model_name}")
    else:
        outdir = os.path.join(OUT, f"generated_text/{model_name}")

    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(outdir, f"no_I_P/{timestr}_{export_index}")

    os.makedirs(outdir, exist_ok=False)

    max_length = get_model_cxt_length(model_name)
    
    count = 0
    while count < 99999999999:
        seed = random.randint(0, 999999999)
        generate_multiple_sequences(
            model_name,
            model, 
            tokenizer,
            prompt_ids, 
            batch_size, 
            seed, 
            export_index, 
            outdir, 
            max_length,
            device=device, 
            P=I_P,
            nucleus=nucleus)
        count += 1
