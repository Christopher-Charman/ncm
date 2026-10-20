"""CPU proof that final-only decoding matches the last progressive stage.

Run: python -m unittest discover -s tests -p 'test_final_only_decode.py'
"""
import unittest
from types import SimpleNamespace

import torch

from models.quant import MultiscaleResidualVectorQuantizer


class FinalOnlyDecodeTest(unittest.TestCase):
    def test_final_latents_match_for_multiple_batches_and_residual_modes(self):
        levels = [2, 4, 8]
        for residual in (False, True):
            torch.manual_seed(23)
            vq = MultiscaleResidualVectorQuantizer(
                vqvae_vocab_size=[8, 8, 8],
                embed_dim=4,
                levels=SimpleNamespace(use_manual_levels=True, manual_levels=levels),
                quant_resi={"enabled": residual, "ratio": 0.5},
            ).eval()

            for batch in (1, 4):
                with self.subTest(residual=residual, batch=batch):
                    codes = torch.randint(0, 8, (batch, sum(levels)))
                    with torch.no_grad():
                        intermediate, per_level = vq.idx_to_zhats(codes, return_zlevels=True)
                        direct = vq.idx_to_z_q(codes)
                    self.assertEqual(len(intermediate), len(levels))
                    self.assertEqual(tuple(direct.shape), (batch, levels[-1], 4))
                    torch.testing.assert_close(direct, per_level[-1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
