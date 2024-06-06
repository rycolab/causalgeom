import warnings
import logging
import os
import sys
import coloredlogs

sys.path.append('./src/')
from evals.mi_computer_utils import (
    pxhs_to_p_x_c_h, qxhs_to_q_x_c_h, compute_p_h_c, 
    compute_p_c_mid_h, compute_p_x_mid_h_c
)

import unittest
import torch
import numpy as np
import math

#sys.path.append('..')

#%%
class TestMIComputerUtils(unittest.TestCase):

    def test_qxhs_to_q_x_c_h(self):
        qxhpars = (
            np.array(
                [
                    [
                        [1/20,2/20],
                        [1/40,3/40],
                    ],
                    [
                        [3/20,1/20],
                        [2/40,1/40],
                    ],
                ],
            ),
            np.array(
                [
                    [
                        [1/20,1/20],
                        [3/40,1/40],
                    ],
                    [
                        [2/20,1/20],
                        [2/40,2/40],
                    ],
                ],
            ),
            np.array(
                [
                    [
                        [2/20,3/20,1/20],
                        [1/40,5/40,2/40],
                    ],
                    [
                        [1/20,4/20,2/20],
                        [5/40,3/40,2/40],
                    ],
                ],
            )
        )
        expected_output = (
            np.array(
                [
                    [
                        [1/80,2/80],
                        [1/160,3/160],
                    ],
                    [
                        [3/80,1/80],
                        [2/160,1/160],
                    ],
                ],
            ),
            np.array(
                [
                    [
                        [1/80,1/80],
                        [3/160,1/160],
                    ],
                    [
                        [2/80,1/80],
                        [2/160,2/160],
                    ],
                ],
            ),
            np.array(
                [
                    [
                        [30/480,45/480,15/480],
                        [32/1280,160/1280,64/1280],
                    ],
                    [
                        [13/560,52/560,26/560],
                        [165/1600,99/1600,66/1600],
                    ],
                ],
            )
        )
        output = qxhs_to_q_x_c_h(qxhpars)
        for x, y in zip(output, expected_output):
            assert (np.isclose(x, y, atol=1e-10).all()), (x, y)


    def test_pxhs_to_p_x_c_h(self):
        log= logging.getLogger( "test_pxhs_to_p_x_c_h" )
        # Test case 1:
        # n = 2, c0 words = 2, c1 words = 2, na words = 3
        pxhs = (
            np.array([
                [1/10, 1/10],
                [2/20, 1/20]
            ]),
            np.array([
                [2/10, 1/10],
                [3/20, 2/20]
            ]),
            np.array([
                [1/10, 1/10, 1/20],
                [1/20, 4/20, 2/20]
            ])
        )
        expected_output = (
                np.array([
                    [1/20, 1/20],
                    [1/20, 1/40]
                ]),
                np.array([
                    [1/10, 1/20],
                    [3/40, 1/20]
                ]),
                np.array([
                    [1/10, 1/10, 1/20],
                    [6/140, 24/140, 12/140]
                ]),
        )
        output = pxhs_to_p_x_c_h(pxhs)
        log.debug(output)
        for x, y in zip(output, expected_output):
            assert (np.isclose(x, y, atol=1e-10).all()), (x, y)

    def test_p_h_c(self):
        p_c_h_x = (
            np.array([
                [1/20, 2/20],
                [1/20, 1/20]
            ]),
            np.array([
                [2/20, 1/20],
                [2/20, 2/20]
            ]),
            np.array([
                [2/20, 1/20, 1/20],
                [1/20, 2/20, 1/20]
            ])
        )
        expected_output = np.array(
            [[
                3/20, 2/20
            ],
            [
                3/20, 4/20
            ],
            [
                4/20, 4/20
            ]]
        ).T
        output = compute_p_h_c(p_c_h_x)
        self.assertTrue(
            np.isclose(output, expected_output, atol=1e-10).all()
        )
    
    def test_p_c_mid_h(self):
        p_c_h_x = (
            np.array([
                [1/20, 2/20],
                [1/20, 1/20]
            ]),
            np.array([
                [2/20, 1/20],
                [2/20, 2/20]
            ]),
            np.array([
                [2/20, 1/20, 1/20],
                [1/20, 2/20, 1/20]
            ])
        )
        expected_output = np.array(
            [
                [6/20, 6/20, 8/20],
                [4/20, 8/20, 8/20]
            ]
        )
        output = compute_p_c_mid_h(p_c_h_x)
        self.assertTrue(
            np.isclose(output, expected_output, atol=1e-10).all()
        )

    def test_compute_p_x_mid_h_c(self):

        p_c_h_x = (
            np.array([
                [1/20, 2/20],
                [1/20, 1/20]
            ]),
            np.array([
                [2/20, 1/20],
                [2/20, 2/20]
            ]),
            np.array([
                [2/20, 1/20, 1/20],
                [1/20, 2/20, 1/20]
            ])
        )
        expected_output = np.array(
            [
                [
                    [1/3,2/3,0,0,0,0,0],
                    [0,0,2/3,1/3,0,0,0],
                    [0,0,0,0,1/2,1/4,1/4],
                ],
                [
                    [1/2,1/2,0,0,0,0,0],
                    [0,0,1/2,1/2,0,0,0],
                    [0,0,0,0,1/4,1/2,1/4],
                ]
            ]
        )

        output = compute_p_x_mid_h_c(p_c_h_x)
        print(output)
        print(expected_output)
        self.assertTrue(
            np.isclose(output, expected_output, atol=1e-10).all()
        )



if __name__ == '__main__':
    unittest.main()