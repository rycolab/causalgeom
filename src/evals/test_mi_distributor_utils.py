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
from mock import patch

#sys.path.append('..')
sys.path.append('./src/')

from evals.mi_distributor_utils import fast_compute_p_words, \
    fast_compute_m_p_words

#%%
class TestMIDistributorUtils(unittest.TestCase):

    def test_fast_compute_p_words(self):
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
        pwords = fast_compute_p_words(
            batch_token_list, batch_pxh, pad_token_id, 
            new_word_tokens, device
        )
        
        # ASSERTIONS
        self.assertTrue((output == pwords).all().item)
        
    def test_fast_compute_p_words_no_new(self):
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
        pwords = fast_compute_p_words(
            batch_token_list, batch_pxh, pad_token_id, 
            new_word_tokens, device
        )
        
        # ASSERTIONS
        self.assertTrue((output == pwords).all().item)        
    
    #%%
    def test_fast_compute_m_p_words(self):
        # n_words x (max_n_tokens): (2 x 2)
        batch_token_list = torch.tensor([
            [2, -1],
            [0, 1]
        ])
        # msamples x n_words x (max_n_tokens + 1) x |vocab| = (3 x 2 x 3 x 3)
        batch_pxh = torch.tensor([
            [
                [
                    [1/3,1/3,1/3],
                    [1/4,1/4,1/2],
                    [2/3,1/6,1/6],
                ],
                [
                    [2/3,1/6,1/6],
                    [3/4,1/8,1/8],
                    [1/6,2/3,1/6],
                ]
            ],
            [
                [
                    [1/6,2/3,1/6],
                    [1/4,1/2,1/4],
                    [1/6,1/6,2/3],
                ],
                [
                    [3/5,1/5,1/5],
                    [1/6,2/3,1/6],
                    [1/6,2/3,1/6],
                ]
            ],
            [
                [
                    [3/4,1/8,1/8],
                    [3/5,1/5,1/5],
                    [2/3,1/6,1/6],
                ],
                [
                    [1/2,1/4,1/4],
                    [1/3,1/3,1/3],
                    [1/6,1/6,2/3],
                ]
            ],
        ])
        batch_log_pxh = batch_pxh.log()
        pad_token_id = -1
        new_word_tokens = [0, 2]
        device = "cpu"

        expected_output = torch.tensor([
            [
                [3/12, 2/(8*9)],
                [1/12, 6/45],
                [1/10, 5/36],
            ]
        ])

        pwords = fast_compute_m_p_words(
            batch_token_list, batch_log_pxh, pad_token_id, 
            new_word_tokens, device
        )
        self.assertTrue(
            torch.isclose(expected_output, pwords, atol = 1e-10).all().item
        )
    
    def test_fast_compute_m_p_words_no_new(self):
        # n_words x (max_n_tokens): (2 x 2)
        batch_token_list = torch.tensor([
            [2, -1],
            [0, 1]
        ])
        # msamples x n_words x (max_n_tokens + 1) x |vocab| = (3 x 2 x 3 x 3)
        batch_pxh = torch.tensor([
            [
                [
                    [1/3,1/3,1/3],
                    [1/4,1/4,1/2],
                    [2/3,1/6,1/6],
                ],
                [
                    [2/3,1/6,1/6],
                    [3/4,1/8,1/8],
                    [1/6,2/3,1/6],
                ]
            ],
            [
                [
                    [1/6,2/3,1/6],
                    [1/4,1/2,1/4],
                    [1/6,1/6,2/3],
                ],
                [
                    [3/5,1/5,1/5],
                    [1/6,2/3,1/6],
                    [1/6,2/3,1/6],
                ]
            ],
            [
                [
                    [3/4,1/8,1/8],
                    [3/5,1/5,1/5],
                    [2/3,1/6,1/6],
                ],
                [
                    [1/2,1/4,1/4],
                    [1/3,1/3,1/3],
                    [1/6,1/6,2/3],
                ]
            ],
        ])
        batch_log_pxh = batch_pxh.log()
        pad_token_id = -1
        new_word_tokens = None
        device = "cpu"

        expected_output = torch.tensor([
            [
                [1/3, 1/12],
                [1/6, 2/5],
                [1/8, 1/6],
            ]
        ])

        pwords = fast_compute_m_p_words(
            batch_token_list, batch_log_pxh, pad_token_id, 
            new_word_tokens, device
        )
        self.assertTrue(
            torch.isclose(expected_output, pwords, atol = 1e-10).all().item
        )

if __name__ == '__main__':
    unittest.main()