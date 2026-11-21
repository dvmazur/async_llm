import unittest
import torch
from robot_sim.candidate_nav.inference_tuning import append_exact_tokens,graph_options

class TuningTests(unittest.TestCase):
    def test_explicit_disable_keeps_graphs_off(self):
        for profile in ('legacy','h100'):
            self.assertEqual(graph_options(profile,False),dict(cuda_graph_max_bs=0,cuda_graph_bs=None))
    def test_only_h100_defaults_to_validated_decode_graphs(self):
        self.assertEqual(graph_options('h100')['cuda_graph_bs'],[1,2,4])
        self.assertEqual(graph_options('legacy')['cuda_graph_max_bs'],0)

    def test_h100_graphs_cover_three_concurrent_policy_streams(self):
        self.assertEqual(graph_options('h100',True)['cuda_graph_bs'],[1,2,4])
        with self.assertRaises(ValueError):graph_options('legacy',True)
    def test_appended_report_keeps_all_image_rows_and_exact_tokens(self):
        pixels=torch.randn(3,8);grid=torch.tensor([[1,2,2]])
        ids=torch.tensor([[10,11,12]]);types=torch.tensor([[0,1,0]])
        source=dict(input_ids=ids,mm_token_type_ids=types,attention_mask=torch.ones_like(ids),
                    pixel_values=pixels,image_grid_thw=grid)
        result=append_exact_tokens(source,[20,21])
        self.assertEqual(result['input_ids'].tolist(),[[10,11,12,20,21]])
        self.assertEqual(result['mm_token_type_ids'].tolist(),[[0,1,0,0,0]])
        self.assertEqual(result['attention_mask'].tolist(),[[1,1,1,1,1]])
        self.assertTrue(torch.equal(ids,torch.tensor([[10,11,12]])))
        self.assertIs(result['pixel_values'],pixels);self.assertIs(result['image_grid_thw'],grid)
    def test_invalid_mask_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            append_exact_tokens(dict(input_ids=torch.ones((1,3),dtype=torch.int64),
                                     mm_token_type_ids=torch.zeros((1,2))),[5])

if __name__=='__main__':unittest.main()
