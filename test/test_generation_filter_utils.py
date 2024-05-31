#%%
import warnings
import logging
import os
import sys
import coloredlogs

import unittest
import numpy as np
import math
import torch

sys.path.append('./src/')
from data.generation_filter_utils import (
    get_matching_h, process_match, process_sample
)

#sys.path.append('..')

#%%
class TestGenerationFilterUtils(unittest.TestCase):
    def test_get_matching_h_normal(self):
        # get_matching_h
        all_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])),
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])),
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), #starting obs
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),
        ]
        reverse_from_idx = 4 
        cxt_to_match = [0,20]
        
        expected_output = torch.tensor([3,2,3])

        output = get_matching_h(
            all_data, reverse_from_idx, cxt_to_match
        )
        self.assertTrue((expected_output == output).all())

    def test_get_matching_h_nomatch(self):
        # get_matching_h
        all_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])), #0
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])), #1
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), 
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),#starting obs
        ]
        reverse_from_idx = 5 
        cxt_to_match = [0,20,10]
        
        expected_output = None

        output = get_matching_h(
            all_data, reverse_from_idx, cxt_to_match
        )
        self.assertEqual(expected_output, output)

    def test_get_matching_h_noearliermatch(self):
        # get_matching_h
        all_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])), #0
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])), #1
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), 
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),#starting obs
        ]
        reverse_from_idx = -1
        cxt_to_match = [0]
        
        expected_output = None

        output = get_matching_h(
            all_data, reverse_from_idx, cxt_to_match
        )
        self.assertEqual(expected_output, output)

    def test_process_match_multiple(self):
        # process_match
        file_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])),
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])),
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), #starting obs
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),
        ]
        idx = 5
        h = torch.tensor([6,2,3])
        all_tokens = [0,20,40,60]
        word_ntok = 2
        
        expected_output_0 = torch.tensor([3,2,3])
        expected_output_1 = [0,20]

        output_0, output_1 = process_match(
            idx, h, all_tokens, file_data, word_ntok
        )
        self.assertTrue((expected_output_0 == output_0).all())
        self.assertTrue(expected_output_1 == output_1)

    def test_process_match_single(self):
        # process_match
        file_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])),
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])),
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), #starting obs
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),
        ]
        idx = 5
        h = torch.tensor([6,2,3])
        all_tokens = [0,20,40,60]
        word_ntok = 1
        
        expected_output_0 = torch.tensor([6,2,3])
        expected_output_1 = [0,20,40]

        output_0, output_1 = process_match(
            idx, h, all_tokens, file_data, word_ntok
        )
        self.assertTrue((expected_output_0 == output_0).all())
        self.assertTrue(expected_output_1 == output_1)

    def test_process_sample_0(self):
        # process_sample(idx, h, all_tokens, file_data, l0_tl, l1_tl)
        idx = 5
        h = torch.tensor([6,2,3])
        all_tokens = [0,20,40,60]
        file_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])),
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])),
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), #starting obs
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),
        ]
        l0_tl = [[100], [40,60]]
        l1_tl = [[101], [102]]
        
        expected_output_0 = "l0"
        expected_output_1 = (np.array([3,2,3]), [40,60], [102], [0,20])

        output_0, output_1 = process_sample(
            idx, h, all_tokens, file_data, l0_tl, l1_tl
        )
        self.assertEqual(expected_output_0, output_0)
        self.assertTrue(
            (expected_output_1[0] == output_1[0]).all()
        )
        self.assertEqual(expected_output_1[1], output_1[1])
        self.assertEqual(expected_output_1[2], output_1[2])
        self.assertEqual(expected_output_1[3], output_1[3])

    def test_process_sample_1(self):
        # process_sample(idx, h, all_tokens, file_data, l0_tl, l1_tl)
        idx = 2
        h = torch.tensor([8,2,3])
        all_tokens = [0,20,10]
        file_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])),
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])),
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), #starting obs
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),
        ]
        l0_tl = [[100], [40,60]]
        l1_tl = [[10], [102]]
        
        expected_output_0 = "l1"
        expected_output_1 = (np.array([8,2,3]), [10], [100], [0,20])

        output_0, output_1 = process_sample(
            idx, h, all_tokens, file_data, l0_tl, l1_tl
        )
        self.assertEqual(expected_output_0, output_0)
        self.assertTrue(
            (expected_output_1[0] == output_1[0]).all()
        )
        self.assertEqual(expected_output_1[1], output_1[1])
        self.assertEqual(expected_output_1[2], output_1[2])
        self.assertEqual(expected_output_1[3], output_1[3])

    def test_process_sample_2(self):
        # process_sample(idx, h, all_tokens, file_data, l0_tl, l1_tl)
        idx = 4
        h = torch.tensor([4,2,3])
        all_tokens = [0,30,50]
        file_data = [
            (torch.tensor([1,2,3]), 20, torch.tensor([0,20])),
            (torch.tensor([2,2,3]), 30, torch.tensor([0,30])),
            (torch.tensor([8,2,3]), 10, torch.tensor([0,20,10])),
            (torch.tensor([3,2,3]), 40, torch.tensor([0,20,40])),
            (torch.tensor([4,2,3]), 50, torch.tensor([0,30,50])),
            (torch.tensor([6,2,3]), 60, torch.tensor([0,20,40,60])), #starting obs
            (torch.tensor([7,2,3]), 20, torch.tensor([0,20,20])),
        ]
        l0_tl = [[100], [101,103]]
        l1_tl = [[103], [104]]
        
        expected_output_0 = "other"
        expected_output_1 = (np.array([4,2,3]), None, None, [0,30])

        output_0, output_1 = process_sample(
            idx, h, all_tokens, file_data, l0_tl, l1_tl
        )
        self.assertEqual(expected_output_0, output_0)
        self.assertTrue(
            (expected_output_1[0] == output_1[0]).all()
        )
        self.assertEqual(expected_output_1[1], output_1[1])
        self.assertEqual(expected_output_1[2], output_1[2])
        self.assertEqual(expected_output_1[3], output_1[3])


if __name__ == '__main__':
    unittest.main()