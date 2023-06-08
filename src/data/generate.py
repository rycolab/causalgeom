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
import torch
import random 

#sys.path.append('..')
sys.path.append('./src/')

from paths import DATASETS, OUT


from utils.lm_loaders import get_model, get_tokenizer, get_V, GPT2_LIST, BERT_LIST
from utils.cuda_loaders import get_device
from evals.kl_eval import load_run_output
#from evals.usage_eval import diag_eval, usage_eval

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")



#%%#################
# Generating Text  #
####################
def check_eos(latest_tokens, eos_token=50256):
    if latest_tokens.unique().shape[0] == 1 and latest_tokens.unique()[0] == eos_token:
        return False
    else:
        return True

def adjust_past_key_values(past_key_values, max_len = 1024):
    new_past_key_values = ()
    for i in range(len(past_key_values)):
        new_sub_values = ()
        for j in range(len(past_key_values[i])):
            vals = past_key_values[i][j]
            assert vals.shape[2]==max_len, "Incorrect shape past_key_values"
            new_sub_values += (vals[:,:,-1023:,:],)
        new_past_key_values += (new_sub_values,)
    return new_past_key_values
    
def generate_sequence_until_eos(model, prompt_ids, batch_size, seed, device="cpu", max_length=1024, eos_token=50256, P=None):
    torch.manual_seed(seed)
    tokens = prompt_ids.repeat(batch_size, 1)
    token_h_pairs = []
    sm = torch.nn.Softmax(1)#.to(device)
    past_key_values = None
    counter = 0
    all_tokens = [tokens]
    while (check_eos(tokens[:,-1], eos_token) or (counter==0)):
        if past_key_values is not None and past_key_values[0][0].shape[2] == max_length:
            #copy = past_key_values[0][0].clone().detach()
            past_key_values = adjust_past_key_values(past_key_values, max_length)
        #if tokens.shape[1] > 1024:
        #    tokens = tokens[:,-1024:]
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
        check_logits = model.lm_head(hs)
        #assert (torch.isclose(final_logits[-1,:], check_logits, atol = 1e-05)).all().item(), "Logits not equal"
        probs = sm(check_logits)
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
        all_tokens.append(sampled_tokens)
        tokens = sampled_tokens
        for h, token, save_h in zip(hs.cpu(), sampled_tokens, save_h_indicator):
            if save_h:
                token_h_pairs.append((h, token.item()))
        counter+=1
        if counter % 100 == 0:
            logging.info(f"Number of tokens so far: {counter}") 
    all_tokens = torch.hstack(all_tokens).cpu()
    return all_tokens, token_h_pairs

#%%
def generate_multiple_sequences(model, tokenizer, prompt_ids, batch_size, seed, export_index, 
    outdir, n_generations=1, device="cpu", max_length=1024, eos_token=50256, P=None):
    all_token_h_pairs = []
    all_lengths = []
    for i in range(n_generations):
        all_tokens, token_h_pairs = generate_sequence_until_eos(model, prompt_ids, batch_size, seed, 
            device=device, max_length=max_length, eos_token=eos_token, P=P)
        all_lengths.append(len(token_h_pairs))
        all_token_h_pairs += token_h_pairs
        for i in range(all_tokens.shape[0]):
            non_eos_text = tokenizer.decode(all_tokens[i, torch.nonzero(all_tokens[i,:] != eos_token)].T.squeeze(0))
            logging.info(f"Generation {i}: {non_eos_text}")
    export_path = os.path.join(outdir, f"generations_seed_{seed}_{export_index}.pkl")
    with open(export_path, 'wb') as f:
        pickle.dump(all_token_h_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    torch.cuda.empty_cache()

#%%
def get_args():
    argparser = argparse.ArgumentParser(description='Generate from model')
    argparser.add_argument(
        "-model",
        type=str,
        choices=GPT2_LIST,
        help="Model for computing hidden states"
    )
    argparser.add_argument(
        "-useP",
        action="store_true",
        default=False,
        help="Whether to load and apply a P for this set of generations",
    )
    argparser.add_argument(
        "-export_index",
        type=int,
        help="Special index for this job",
    )
    return argparser.parse_args()

#%%
if __name__=="__main__":
    args = get_args()
    logging.info(args)

    model_name = args.model
    useP = args.useP
    export_index = args.export_index
    batch_size = 3
    #model_name = "gpt2-large"
    #use_P = True
    #export_index = 0


    device = get_device()
    #device = "cpu"
    model = get_model(model_name).to(device)
    tokenizer = get_tokenizer(model_name)

    prompt_ids = tokenizer(tokenizer.bos_token, return_tensors="pt").input_ids.to(device)

    if model_name == "gpt2-large":
        run_output = os.path.join(OUT, "run_output/linzen/gpt2-large/230415/run_gpt2-large_k1_Pms31_Pg0.5_clfms31_clfg0.5_2023-04-15-20:20:45_0_1.pkl")
    else:
        run_output = None

    outdir = os.path.join(OUT, f"generated_text/{model_name}")
    if useP:
        P, I_P = load_run_output(run_output)
        I_P = torch.tensor(I_P, device=device).float()
        outdir = os.path.join(outdir, "I_P")
    else:
        outdir = os.path.join(outdir, "no_I_P")
        I_P = None

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    count = 0
    while count < 1000:
        seed = random.randint(0, 999999999)
        generate_multiple_sequences(
            model, 
            tokenizer,
            prompt_ids, 
            batch_size, 
            seed, 
            export_index, 
            outdir, 
            device=device, 
            P=I_P)
        count += 1
