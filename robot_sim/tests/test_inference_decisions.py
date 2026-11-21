"""Decision outputs, ties and rejection match the original float32 operations."""
import unittest
import torch
from robot_sim.candidate_nav.inference_decisions import checked_argmax, checked_probabilities


class DecisionTests(unittest.TestCase):
    def test_argmax_and_probabilities_are_exact_for_full_and_constrained_choices(self):
        generator = torch.Generator().manual_seed(19)
        for dtype in (torch.float32, torch.bfloat16):
            for size in (8, 1000):
                values = torch.randn(size, generator=generator).to(dtype)
                ids = [size-1, 0, size//2, 1]
                self.assertEqual(checked_argmax(values,error='bad'),values.float().argmax().item())
                self.assertEqual(checked_argmax(values,ids,error='bad'),values.float()[ids].argmax().item())
                self.assertEqual(checked_probabilities(values,ids,error='bad'),values.float()[ids].softmax(0).tolist())

    def test_equal_logits_keep_first_allowed_choice(self):
        values = torch.zeros(10)
        self.assertEqual(checked_argmax(values,[7,2],error='bad'),0)
        self.assertEqual(checked_probabilities(values,[7,2],error='bad'),[.5,.5])

    def test_nonfinite_logits_are_rejected_even_outside_allowed_choices(self):
        for invalid in (float('nan'), float('inf'), -float('inf')):
            values=torch.tensor([1.,2.,invalid])
            for decide in (checked_argmax,checked_probabilities):
                with self.assertRaisesRegex(FloatingPointError,'original error'):
                    decide(values,[0,1],error='original error')
