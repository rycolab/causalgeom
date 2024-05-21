#%%
import warnings
import logging
import os
import sys
import coloredlogs

import unittest
import torch
import numpy as np
import math

#sys.path.append('..')
sys.path.append('./src/')

from evals.mi_distributor_utils import compute_p_words, \
    compute_m_p_words

#%%
class TestMIDistributorUtils(unittest.TestCase):

    def test_compute_p_words(self):
        # INPUT VALUES
        # n_words x max_n_tokens = (2x2)
        batch_token_list = torch.tensor([
            [0, -1],
            [1, 0]
        ])
        # n_words x (max_n_tokens + 1) x |vocab| = (2 x 3 x 3)
        batch_pxh = torch.tensor(
            [
                [
                    [1/3, 1/3, 1/3],
                    [1/2, 1/4, 1/4],
                    [2/3, 1/6, 1/6]
                ],
                [
                    [1/3, 1/3, 1/3],
                    [1/2, 1/4, 1/4],
                    [2/3, 1/6, 1/6]
                ]
            ]
        )
        pad_token_id = -1
        new_word_tokens = [0, 1]
        device = "cpu"

        # EXPECTED OUTPUT: nwords tensor (1 x nwords)
        output = torch.tensor([1/4, 5/36]).unsqueeze(0)

        # Function calls
        pwords = compute_p_words(
            batch_token_list, batch_pxh, pad_token_id, 
            new_word_tokens, device
        )
        
        # ASSERTIONS
        self.assertTrue((output == pwords).all().item)
        
    def test_compute_p_words_no_new(self):
        # INPUT VALUES
        # n_words x max_n_tokens = (2x2)
        batch_token_list = torch.tensor([
            [0, -1],
            [1, 0]
        ])
        # n_words x (max_n_tokens + 1) x |vocab| = (2 x 3 x 3)
        batch_pxh = torch.tensor(
            [
                [
                    [1/3, 1/3, 1/3],
                    [1/2, 1/4, 1/4],
                    [2/3, 1/6, 1/6]
                ],
                [
                    [1/3, 1/3, 1/3],
                    [1/2, 1/4, 1/4],
                    [2/3, 1/6, 1/6]
                ]
            ]
        )
        pad_token_id = -1
        new_word_tokens = None
        device = "cpu"

        # EXPECTED OUTPUT: nwords tensor (1 x nwords)
        output = torch.tensor([1/3, 1/6]).unsqueeze(0)

        # Function calls
        pwords = compute_p_words(
            batch_token_list, batch_pxh, pad_token_id, 
            new_word_tokens, device
        )
        
        # ASSERTIONS
        self.assertTrue((output == pwords).all().item)        
    
    def test_compute_m_p_words(self):
        # batch_size = 2
        # max_ntok = 2
        # msamples = 3
        # |V| = 3
        # bs x max_n_tokens: (2 x 2)
        batch_tokens = torch.tensor([
            [2, -1],
            [0, 1]
        ])
        # first_log_pxh: (msamples x |V|)
        first_log_pxh = torch.tensor([
            [
                [math.log(1/5),math.log(2/5),math.log(2/5)],
                [math.log(1/3),math.log(1/3),math.log(1/3)],
                [math.log(1/2),math.log(1/4),math.log(1/4)],
            ]
        ])
        # next_log_pxh: (bs x max_ntok x |V|)
        next_log_pxh = torch.tensor([
            [
                [math.log(1/6),math.log(1/2),math.log(1/3)],
                [math.log(1/3),math.log(1/3),math.log(1/3)],
            ],
            [
                [math.log(1/4),math.log(1/4),math.log(1/2)],
                [math.log(3/5),math.log(1/5),math.log(1/5)],
            ]
        ])
        pad_token_id = -1
        new_word_tokens = [0, 2]
        device = "cpu"

        expected_output = torch.tensor([
            [
                [2/10, 1/25],
                [1/6, 1/15],
                [1/8, 1/10],
            ]
        ])

        pwords = compute_m_p_words(
            batch_tokens, first_log_pxh, next_log_pxh,
            pad_token_id, new_word_tokens, device
        )
        self.assertTrue(
            torch.isclose(expected_output, pwords, atol = 1e-10).all().item
        )      
    
    def test_compute_m_p_words_no_new(self):
        # batch_size = 2
        # max_ntok = 2
        # msamples = 3
        # |V| = 3
        # bs x max_n_tokens: (2 x 2)
        batch_tokens = torch.tensor([
            [2, -1],
            [0, 1]
        ])
        # first_log_pxh: (msamples x |V|)
        first_log_pxh = torch.tensor([
            [
                [math.log(1/5),math.log(2/5),math.log(2/5)],
                [math.log(1/3),math.log(1/3),math.log(1/3)],
                [math.log(1/2),math.log(1/4),math.log(1/4)],
            ]
        ])
        # next_log_pxh: (bs x max_ntok x |V|)
        next_log_pxh = torch.tensor([
            [
                [math.log(1/6),math.log(1/2),math.log(1/3)],
                [math.log(1/3),math.log(1/3),math.log(1/3)],
            ],
            [
                [math.log(1/4),math.log(1/4),math.log(1/2)],
                [math.log(3/5),math.log(1/5),math.log(1/5)],
            ]
        ])
        pad_token_id = -1
        new_word_tokens = None
        device = "cpu"

        expected_output = torch.tensor([
            [
                [2/5, 1/20],
                [1/3, 1/12],
                [1/4, 1/8],
            ]
        ])

        pwords = compute_m_p_words(
            batch_tokens, first_log_pxh, next_log_pxh,
            pad_token_id, new_word_tokens, device
        )
        self.assertTrue(
            torch.isclose(expected_output, pwords, atol = 1e-10).all().item
        )      

if __name__ == '__main__':
    unittest.main()