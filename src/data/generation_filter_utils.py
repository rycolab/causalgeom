import warnings
import logging
import os
import sys
import coloredlogs

import numpy as np
import torch

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
def get_matching_h(
    all_data: list, 
    reverse_from_idx: int, 
    cxt_to_match: list[int]
) -> torch.tensor:
    while reverse_from_idx >= 0:
        prev_h, _, prev_all_tokens = all_data[reverse_from_idx]
        if prev_all_tokens[:-1].tolist() == cxt_to_match:
            return prev_h
        else:
            reverse_from_idx-=1
    return None

def process_match(idx, h, all_tokens, file_data, word_ntok):
    if word_ntok > 1:
        matched_h = get_matching_h(
            file_data, idx-1, all_tokens[:-word_ntok]
        )
        if matched_h is not None:
            return matched_h, all_tokens[:-word_ntok]
        else:
            return None, None
    else:
        return h, all_tokens[:-1]

def process_sample(idx, h, all_tokens, file_data, l0_tl, l1_tl):
    for l0_wordtok, l1_wordtok in zip(l0_tl, l1_tl):
        l0_ntok = len(l0_wordtok)
        l1_ntok = len(l1_wordtok)
        if all_tokens[-l0_ntok:] == l0_wordtok:
            matched_h, cxt_tok = process_match(
                idx, h, all_tokens, file_data, l0_ntok
            )
            return "l0", (matched_h.numpy(), l0_wordtok, l1_wordtok, cxt_tok)
        elif all_tokens[-l1_ntok:] == l1_wordtok:
            matched_h, cxt_tok = process_match(
                idx, h, all_tokens, file_data, l1_ntok
            )
            return "l1", (matched_h.numpy(), l1_wordtok, l0_wordtok, cxt_tok)
        else:
            continue
    return "other", (h.numpy(), None, None, all_tokens[:-1])
    
