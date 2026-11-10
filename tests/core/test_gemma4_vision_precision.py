"""Padding cannot contribute to real image features through Attention or pooling."""
import pytest
import torch

from test_gemma4_vision import models, inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('num_images', [1, 2])
def test_padding_values_cannot_change_real_features(models, num_images):
    reference, actual, _ = models
    data = inputs(num_images)
    positions = torch.nn.functional.pad(data['image_position_ids'], (0, 0, 0, 18), value=-1)
    pixels = torch.nn.functional.pad(data['pixel_values'], (0, 0, 0, 18))
    noisy = pixels.clone()
    noisy[:, -18:] = 100
    with torch.inference_mode():
        baseline = actual.encode_images(pixels, positions)
        result = actual.encode_images(noisy, positions)
        expected = torch.cat(reference.get_image_features(
            noisy, positions, return_dict=True).pooler_output)
    torch.testing.assert_close(result, baseline, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
