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

from evals.mi_computer_utils import compute_p_x_mid_c

#%%
class TestMIComputerUtils(unittest.TestCase):

    def test_compute_p_x_mid_c(self):
        pxhs = (
            np.array([
                [1/3,1/3,1/3], 
                [1/2,1/4,1/4],
                [2/5,2/5,1/5],
            ]),
            np.array([
                [1/4,1/2,1/4], 
                [1/3,1/2,1/6],
                [1/5,1/5,3/5],
            ]),
            np.array([
                [1/4,1/4,1/4,1/4], 
                [1/6,1/6,1/2,1/6], 
                [1/5,2/5,1/5,1/5], 
            ])
        )
        expected_output = np.array(
            [
                [37/90,59/180,47/180,0,0,0,0,0,0,0],
                [0,0,0,47/180,24/60,122/360,0,0,0,0],
                [0,0,0,0,0,0,74/360,98/360,38/120,74/360],
            ]
        )

        output = compute_p_x_mid_c(pxhs)
        self.assertTrue(np.isclose(output, expected_output, atol=1e-10).all())

#%%
if __name__ == '__main__':
    unittest.main()