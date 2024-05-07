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

from evals.mi_distributor_utils import prep_generated_data, \
    compute_batch_inner_loop_qxhs, get_nucleus_arg, \
        sample_gen_all_hs_batched, compute_pxh_batch_handler,\
            fast_compute_p_words, fast_compute_m_p_words,\
                compute_pxh_batch_handler

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
        pwords = fast_compute_p_words(
            batch_token_list, batch_pxh, pad_token_id, 
            new_word_tokens, device
        )
        
        # ASSERTIONS
        self.assertTrue((output == pwords).all().item)
        
    #%%
    def test_compute_m_p_words(self):
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

    def test_compute_batch_inner_loop_qxhs_hbot(self):
        # SETUP
        mode = "hbot"
        nmH = torch.tensor([
            [
                [
                    [1,2,3],
                    [1,2,3],
                    [1,2,3],
                ],
                [
                    [4,5,6],
                    [4,5,6],
                    [4,5,6],
                ]
            ],
            [
                [
                    [7,8,9],
                    [7,8,9],
                    [7,8,9],
                ],
                [
                    [10,11,12],
                    [10,11,12],
                    [10,11,12],
                ]
            ],
        ], dtype=torch.float32)
        other_nmH = torch.tensor([
            [
                [
                    [1,2,3],
                    [4,5,6],
                    [7,8,9],
                ],
                [
                    [10,11,12],
                    [13,14,15],
                    [16,17,18],
                ]
            ],
            [
                [
                    [19,20,21],
                    [22,23,24],
                    [25,26,27],
                ],
                [
                    [28,29,30],
                    [31,32,33],
                    [34,35,36],
                ]
            ],
        ], dtype=torch.float32)

        P = torch.zeros((3,3)).to(dtype=torch.float32)
        I_P = torch.eye(3).to(dtype=torch.float32)
        P[0,0] = 1
        I_P[0,0] = 0
        V = torch.eye(3).to(dtype=torch.float32)
        gpu_out = False

        # Expected output
        expected_output = torch.tensor([
            [
                [
                    [math.exp(1)/(math.exp(1) + math.exp(2) + math.exp(3)),
                    math.exp(2)/(math.exp(1) + math.exp(2) + math.exp(3)),
                    math.exp(3)/(math.exp(1) + math.exp(2) + math.exp(3))],
                    [math.exp(1)/(math.exp(1) + math.exp(5) + math.exp(6)),
                    math.exp(5)/(math.exp(1) + math.exp(5) + math.exp(6)),
                    math.exp(6)/(math.exp(1) + math.exp(5) + math.exp(6))], 
                    [math.exp(1)/(math.exp(1) + math.exp(8) + math.exp(9)),
                    math.exp(8)/(math.exp(1) + math.exp(8) + math.exp(9)),
                    math.exp(9)/(math.exp(1) + math.exp(8) + math.exp(9))], 
                ],
                [
                    [math.exp(4)/(math.exp(4) +math.exp(11) +math.exp(12)),
                    math.exp(11)/(math.exp(4) +math.exp(11) +math.exp(12)),
                    math.exp(12)/(math.exp(4) +math.exp(11) +math.exp(12))], 
                    [math.exp(4)/(math.exp(4) + math.exp(14) + math.exp(15)),
                    math.exp(14)/(math.exp(4) + math.exp(14) + math.exp(15)),
                    math.exp(15)/(math.exp(4) + math.exp(14) + math.exp(15))], 
                    [math.exp(4)/(math.exp(4) + math.exp(17) + math.exp(18)),
                    math.exp(17)/(math.exp(4) + math.exp(17) + math.exp(18)),
                    math.exp(18)/(math.exp(4) + math.exp(17) + math.exp(18))], 
                ]
            ],
            [
                [
                    [math.exp(7)/(math.exp(7) + math.exp(20) + math.exp(21)),
                    math.exp(20)/(math.exp(7) + math.exp(20) + math.exp(21)),
                    math.exp(21)/(math.exp(7) + math.exp(20) + math.exp(21))],  
                    [math.exp(7)/(math.exp(7) + math.exp(23) + math.exp(24)),
                    math.exp(23)/(math.exp(7) + math.exp(23) + math.exp(24)),
                    math.exp(24)/(math.exp(7) + math.exp(23) + math.exp(24))], 
                    [math.exp(7)/(math.exp(7)+math.exp(26)+math.exp(27)),
                    math.exp(26)/(math.exp(7)+math.exp(26)+math.exp(27)),
                    math.exp(27)/(math.exp(7)+math.exp(26)+math.exp(27))], 
                ],
                [
                    [math.exp(10)/(math.exp(10) + math.exp(29) + math.exp(30)),
                    math.exp(29)/(math.exp(10) + math.exp(29) + math.exp(30)),
                    math.exp(30)/(math.exp(10) + math.exp(29) + math.exp(30))], 
                    [math.exp(10)/(math.exp(10) + math.exp(32) + math.exp(33)),
                    math.exp(32)/(math.exp(10) + math.exp(32) + math.exp(33)),
                    math.exp(33)/(math.exp(10) + math.exp(32) + math.exp(33))], 
                    [math.exp(10)/(math.exp(10) + math.exp(35) + math.exp(36)),
                    math.exp(35)/(math.exp(10) + math.exp(35) + math.exp(36)),
                    math.exp(36)/(math.exp(10) + math.exp(35) + math.exp(36))], 
                ]
            ],
        ]).log()

        # ASSERTIONS
        output = compute_batch_inner_loop_qxhs(
            "hbot", nmH, other_nmH, P, I_P, V, gpu_out
        )

        self.assertTrue(
            torch.isclose(expected_output, output, atol = 1e-10).all().item
        )


    def test_compute_batch_inner_loop_qxhs_hbot(self):
        # SETUP
        mode = "hpar"
        nmH = torch.tensor([
            [
                [
                    [1,2,3],
                    [1,2,3],
                    [1,2,3],
                ],
                [
                    [4,5,6],
                    [4,5,6],
                    [4,5,6],
                ]
            ],
            [
                [
                    [7,8,9],
                    [7,8,9],
                    [7,8,9],
                ],
                [
                    [10,11,12],
                    [10,11,12],
                    [10,11,12],
                ]
            ],
        ], dtype=torch.float32)
        other_nmH = torch.tensor([
            [
                [
                    [1,2,3],
                    [4,5,6],
                    [7,8,9],
                ],
                [
                    [10,11,12],
                    [13,14,15],
                    [16,17,18],
                ]
            ],
            [
                [
                    [19,20,21],
                    [22,23,24],
                    [25,26,27],
                ],
                [
                    [28,29,30],
                    [31,32,33],
                    [34,35,36],
                ]
            ],
        ], dtype=torch.float32)

        P = torch.zeros((3,3)).to(dtype=torch.float32)
        I_P = torch.eye(3).to(dtype=torch.float32)
        P[0,0] = 1
        I_P[0,0] = 0
        V = torch.eye(3).to(dtype=torch.float32)
        gpu_out = False

        # Expected output
        expected_output = torch.tensor([
            [
                [
                    [math.exp(1)/(math.exp(1) + math.exp(2) + math.exp(3)),
                    math.exp(2)/(math.exp(1) + math.exp(2) + math.exp(3)),
                    math.exp(3)/(math.exp(1) + math.exp(2) + math.exp(3))],
                    [math.exp(1)/(math.exp(1) + math.exp(5) + math.exp(6)),
                    math.exp(5)/(math.exp(1) + math.exp(5) + math.exp(6)),
                    math.exp(6)/(math.exp(1) + math.exp(5) + math.exp(6))], 
                    [math.exp(1)/(math.exp(1) + math.exp(8) + math.exp(9)),
                    math.exp(8)/(math.exp(1) + math.exp(8) + math.exp(9)),
                    math.exp(9)/(math.exp(1) + math.exp(8) + math.exp(9))], 
                ],
                [
                    [math.exp(4)/(math.exp(4) +math.exp(11) +math.exp(12)),
                    math.exp(11)/(math.exp(4) +math.exp(11) +math.exp(12)),
                    math.exp(12)/(math.exp(4) +math.exp(11) +math.exp(12))], 
                    [math.exp(4)/(math.exp(4) + math.exp(14) + math.exp(15)),
                    math.exp(14)/(math.exp(4) + math.exp(14) + math.exp(15)),
                    math.exp(15)/(math.exp(4) + math.exp(14) + math.exp(15))], 
                    [math.exp(4)/(math.exp(4) + math.exp(17) + math.exp(18)),
                    math.exp(17)/(math.exp(4) + math.exp(17) + math.exp(18)),
                    math.exp(18)/(math.exp(4) + math.exp(17) + math.exp(18))], 
                ]
            ],
            [
                [
                    [math.exp(7)/(math.exp(7) + math.exp(20) + math.exp(21)),
                    math.exp(20)/(math.exp(7) + math.exp(20) + math.exp(21)),
                    math.exp(21)/(math.exp(7) + math.exp(20) + math.exp(21))],  
                    [math.exp(7)/(math.exp(7) + math.exp(23) + math.exp(24)),
                    math.exp(23)/(math.exp(7) + math.exp(23) + math.exp(24)),
                    math.exp(24)/(math.exp(7) + math.exp(23) + math.exp(24))], 
                    [math.exp(7)/(math.exp(7)+math.exp(26)+math.exp(27)),
                    math.exp(26)/(math.exp(7)+math.exp(26)+math.exp(27)),
                    math.exp(27)/(math.exp(7)+math.exp(26)+math.exp(27))], 
                ],
                [
                    [math.exp(10)/(math.exp(10) + math.exp(29) + math.exp(30)),
                    math.exp(29)/(math.exp(10) + math.exp(29) + math.exp(30)),
                    math.exp(30)/(math.exp(10) + math.exp(29) + math.exp(30))], 
                    [math.exp(10)/(math.exp(10) + math.exp(32) + math.exp(33)),
                    math.exp(32)/(math.exp(10) + math.exp(32) + math.exp(33)),
                    math.exp(33)/(math.exp(10) + math.exp(32) + math.exp(33))], 
                    [math.exp(10)/(math.exp(10) + math.exp(35) + math.exp(36)),
                    math.exp(35)/(math.exp(10) + math.exp(35) + math.exp(36)),
                    math.exp(36)/(math.exp(10) + math.exp(35) + math.exp(36))], 
                ]
            ],
        ]).log()

        # ASSERTIONS
        output = compute_batch_inner_loop_qxhs(
            "hbot", nmH, other_nmH, P, I_P, V, gpu_out
        )

        self.assertTrue(
            torch.isclose(expected_output, output, atol = 1e-10).all().item
        )

    @patch("numpy.random.randint")
    def test_sample_gen_all_hs_batched(self, randint_mock):
        """ 
        Dims:
        - nwords = 2
        - max_ntokens = 2
        - d = 3
        - msamples = 2

        """
        # nwords x (max_ntokens + 1) x d: (2 x 3 x 3)
        n_ntok_H = torch.tensor(
            [
                [
                    [1,2,3],
                    [4,5,6],
                    [7,8,9],
                ],
                [
                    [10,11,12],
                    [13,14,15],
                    [16,17,18],
                ],
            ]
        )
        msamples = 2
        gen_all_hs = torch.tensor(
            [
                [101,102,103],
                [104,105,106],
                [107,108,109]
            ]
        )
        # need nwords x msamples = 4 indices
        randint_mock.return_value = np.array([0,1,2,1])

        # Expected output: msamples x nwords x (max_ntokens+1) x d
        expected_m_n_ntok_H = torch.tensor(
            [
                [
                    [
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                ],
                [
                    [
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                ],
            ]
        )
        expected_other_hs = torch.tensor(
            [
                [
                    [
                        [101,102,103],
                        [101,102,103],
                        [101,102,103],
                    ],
                    [
                        [104,105,106],
                        [104,105,106],
                        [104,105,106],
                    ],
                ],
                [
                    [
                        [107,108,109],
                        [107,108,109],
                        [107,108,109],
                    ],
                    [
                        [104,105,106],
                        [104,105,106],
                        [104,105,106],
                    ],
                ],
            ]
        )

        m_n_ntok_H, other_hs = sample_gen_all_hs_batched(
            n_ntok_H, msamples, gen_all_hs, "cpu"
        )
        # Assertions
        self.assertTrue((expected_m_n_ntok_H == m_n_ntok_H).all().item)       
        self.assertTrue((expected_other_hs == other_hs).all().item)       


    def test_msamples_compute_batch_inner_loop_qxhs(self):
        #INPUT VALUES
        mode = "hbot"
        nmH = torch.tensor([
            [
                [
                    [1,2,3],
                    [4,5,6],
                ],
                [
                    [7,8,9],
                    [10,11,12],
                ]
            ],
            [
                [
                    [1,2,3],
                    [4,5,6],
                ],
                [
                    [7,8,9],
                    [10,11,12],
                ]
            ],
            [
                [
                    [1,2,3],
                    [4,5,6],
                ],
                [
                    [7,8,9],
                    [10,11,12],
                ]
            ],
        ], dtype=torch.float32)
        other_nmH = torch.tensor([
            [
                [
                    [1,2,3],
                    [1,2,3],
                ],
                [
                    [4,5,6],
                    [4,5,6],
                ]
            ],
            [
                [
                    [7,8,9],
                    [7,8,9],
                ],
                [
                    [10,11,12],
                    [10,11,12],
                ]
            ],
            [
                [
                    [13,14,15],
                    [13,14,15],
                ],
                [
                    [16,17,18],
                    [16,17,18],
                ]
            ],
        ], dtype=torch.float32)

        P = torch.zeros((3,3)).to(dtype=torch.float32)
        I_P = torch.eye(3).to(dtype=torch.float32)

        P[0,0] = 1
        I_P[0,0] = 0

        V = torch.eye(3).to(dtype=torch.float32)

        gpu_out = False

        # EXPECTED OUTPUT
        expected_output = np.array([
            [
                [
                    [math.exp(1)/(math.exp(1) + math.exp(2) + math.exp(3)),
                    math.exp(2)/(math.exp(1) + math.exp(2) + math.exp(3)),
                    math.exp(3)/(math.exp(1) + math.exp(2) + math.exp(3))],
                    [math.exp(1)/(math.exp(1) + math.exp(5) + math.exp(6)),
                    math.exp(5)/(math.exp(1) + math.exp(5) + math.exp(6)),
                    math.exp(6)/(math.exp(1) + math.exp(5) + math.exp(6))], 
                    [math.exp(1)/(math.exp(1) + math.exp(8) + math.exp(9)),
                    math.exp(8)/(math.exp(1) + math.exp(8) + math.exp(9)),
                    math.exp(9)/(math.exp(1) + math.exp(8) + math.exp(9))], 
                ],
                [
                    [math.exp(4)/(math.exp(4) +math.exp(11) +math.exp(12)),
                    math.exp(11)/(math.exp(4) +math.exp(11) +math.exp(12)),
                    math.exp(12)/(math.exp(4) +math.exp(11) +math.exp(12))], 
                    [math.exp(4)/(math.exp(4) + math.exp(14) + math.exp(15)),
                    math.exp(14)/(math.exp(4) + math.exp(14) + math.exp(15)),
                    math.exp(15)/(math.exp(4) + math.exp(14) + math.exp(15))], 
                    [math.exp(4)/(math.exp(4) + math.exp(17) + math.exp(18)),
                    math.exp(17)/(math.exp(4) + math.exp(17) + math.exp(18)),
                    math.exp(18)/(math.exp(4) + math.exp(17) + math.exp(18))], 
                ]
            ],
            [
                [
                    [math.exp(7)/(math.exp(7) + math.exp(20) + math.exp(21)),
                    math.exp(20)/(math.exp(7) + math.exp(20) + math.exp(21)),
                    math.exp(21)/(math.exp(7) + math.exp(20) + math.exp(21))],  
                    [math.exp(7)/(math.exp(7) + math.exp(23) + math.exp(24)),
                    math.exp(23)/(math.exp(7) + math.exp(23) + math.exp(24)),
                    math.exp(24)/(math.exp(7) + math.exp(23) + math.exp(24))], 
                    [math.exp(7)/(math.exp(7)+math.exp(26)+math.exp(27)),
                    math.exp(26)/(math.exp(7)+math.exp(26)+math.exp(27)),
                    math.exp(27)/(math.exp(7)+math.exp(26)+math.exp(27))], 
                ],
                [
                    [math.exp(10)/(math.exp(10) + math.exp(29) + math.exp(30)),
                    math.exp(29)/(math.exp(10) + math.exp(29) + math.exp(30)),
                    math.exp(30)/(math.exp(10) + math.exp(29) + math.exp(30))], 
                    [math.exp(10)/(math.exp(10) + math.exp(32) + math.exp(33)),
                    math.exp(32)/(math.exp(10) + math.exp(32) + math.exp(33)),
                    math.exp(33)/(math.exp(10) + math.exp(32) + math.exp(33))], 
                    [math.exp(10)/(math.exp(10) + math.exp(35) + math.exp(36)),
                    math.exp(35)/(math.exp(10) + math.exp(35) + math.exp(36)),
                    math.exp(36)/(math.exp(10) + math.exp(35) + math.exp(36))], 
                ]
            ],
        ])

        output = compute_batch_inner_loop_qxhs("hbot", nmH, other_nmH, P, I_P, V, gpu_out)

        #(expected_output - output).max()
        #TEST UNFINISHED

if __name__ == '__main__':
    unittest.main()