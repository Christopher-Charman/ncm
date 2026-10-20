"""CPU proof that final-only decoding matches the last progressive stage.

Run: python -m unittest discover -s tests -p 'test_final_only_decode.py'
"""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import torch

from models.quant import MultiscaleResidualVectorQuantizer
from utils.generation import decode_final_logits


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


    def test_evaluation_consumer_decodes_without_progressive_stages(self):
        codes = torch.zeros((1, 14), dtype=torch.long)
        latent = torch.ones((1, 8, 4))
        logits = torch.ones((1, 8, 16))
        quantizer = Mock()
        quantizer.idx_to_z_q.return_value = latent
        decoder = Mock(return_value=logits)
        vqvae = SimpleNamespace(vq=quantizer, decoder=decoder)

        self.assertIs(decode_final_logits(vqvae, codes), logits)
        quantizer.idx_to_z_q.assert_called_once_with(codes)
        quantizer.idx_to_zhats.assert_not_called()
        decoder.assert_called_once_with(latent)

        # Also guard the actual evaluation caller, which cannot be imported in
        # a CPU test because eval_shard.py executes CUDA CLI setup on import.
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "scripts/eval_shard.py").read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "run")
        # Narrow structural guard: the logits passed into the consumer must
        # come directly from the intended final-only decoder. This does not
        # attempt Python dynamic alias/monkeypatch analysis.
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        picks = [node for node in calls
                 if isinstance(node.func, ast.Name) and node.func.id == "pick_tokens"]
        self.assertEqual(len(picks), 1)
        self.assertEqual(len(picks[0].args), 1)
        logits_call = picks[0].args[0]
        self.assertIsInstance(logits_call, ast.Call)
        self.assertIsInstance(logits_call.func, ast.Name)
        self.assertEqual(logits_call.func.id, "decode_final_logits")
        self.assertFalse(any(
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("idx_to_zhats", "reconstruct_multiscales", "decode_multiscales")
            for node in calls
        ))


if __name__ == "__main__":
    unittest.main()
