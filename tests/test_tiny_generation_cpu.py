"""Tiny checkpoint-free CPU smoke for complete NCM generation and decoding.

Run: python -m unittest discover -s tests -p 'test_tiny_generation_cpu.py' -v
"""
import unittest

import torch
from omegaconf import OmegaConf

from models.vqvae import VQVAE
from models.next_concept import NextConceptTransformer


class TinyGenerationTest(unittest.TestCase):
    def test_random_initialized_cpu_generation_and_final_decode(self):
        torch.set_num_threads(1)
        torch.manual_seed(20)
        vq_cfg = OmegaConf.create({
            "in_vocab_size": 16, "context_length": 8, "embed_dim": 16,
            "vqvae_vocab_size": [8, 8, 8], "n_layers": 2,
            "compression_factor": 1, "dropout": 0.0, "dec_layers": 1,
            "use_qk_norm": False,
            "vector_quantizer_config": {
                "clss": "multiscale_residual_vector_quantizer",
                "decay": 0.99, "commitment_cost": 0.25,
                "quant_resi": {"enabled": False},
                "levels": {"use_manual_levels": True, "manual_levels": [2, 4, 8]},
            },
        })
        vqvae = VQVAE(**vq_cfg).eval()
        generator_cfg = OmegaConf.create({
            "n_embd": 32, "n_layer": 2, "n_head": 4,
            "bias": True, "dropout": 0.0, "vocab_size": [8, 8, 8],
            "attn_pattern": "block_diagonal", "prefix_len": 4,
            "context_length": 12, "use_positional_encoding": True,
            "use_level_encoding": True, "use_rope": True,
            "rope_base": 10000, "use_qk_norm": True,
            "raw_token_prefix": True, "query_level0": True,
            "predict_level0": False,
        })
        generator = NextConceptTransformer(vqvae_local=vqvae, config=generator_cfg).eval()
        for batch in (1, 2):
            with self.subTest(batch=batch), torch.no_grad():
                prefix = torch.randint(0, 16, (batch, 4))
                codes = generator.generate(None, prefix, temperature=0.0, top_k=0, top_p=1.0)
                final_only = vqvae.vq.idx_to_z_q(codes)
                _, stages = vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
                logits = vqvae.decoder(final_only)
                self.assertEqual(tuple(codes.shape), (batch, 14))
                self.assertEqual(tuple(final_only.shape), (batch, 8, 16))
                self.assertEqual(tuple(logits.shape), (batch, 8, 16))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                torch.testing.assert_close(final_only, stages[-1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
