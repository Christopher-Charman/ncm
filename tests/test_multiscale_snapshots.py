"""CPU regression for independent coarse-to-fine reconstruction snapshots.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_multiscale_snapshots.py'
"""
import unittest
from types import SimpleNamespace

import torch

from models.quant import MultiscaleResidualVectorQuantizer


class MultiscaleSnapshotsTest(unittest.TestCase):
    def test_decoded_level_snapshots_do_not_alias_later_levels(self):
        vq = MultiscaleResidualVectorQuantizer(
            vqvae_vocab_size=[2, 2],
            embed_dim=2,
            levels=SimpleNamespace(use_manual_levels=True, manual_levels=[2, 4]),
            quant_resi={"enabled": False},
        )
        with torch.no_grad():
            vq.embeddings[0].weight[0].copy_(torch.tensor([1.0, 0.0]))
            vq.embeddings[1].weight[0].copy_(torch.tensor([0.0, 2.0]))

        indices = torch.zeros((1, 2 + 4), dtype=torch.long)
        snapshots, level_inputs = vq.idx_to_zhats(indices, return_zlevels=True)

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(len(level_inputs), 2)
        self.assertEqual(tuple(level_inputs[0].shape), (1, 2, 2))
        self.assertEqual(tuple(level_inputs[1].shape), (1, 4, 2))
        self.assertNotEqual(snapshots[0].data_ptr(), snapshots[1].data_ptr())
        torch.testing.assert_close(
            snapshots[0], torch.tensor([[[1.0, 0.0]] * 4])
        )
        torch.testing.assert_close(
            snapshots[1], torch.tensor([[[1.0, 2.0]] * 4])
        )

        # The final snapshot can change without rewriting the coarse stage.
        snapshots[1].add_(5.0)
        torch.testing.assert_close(
            snapshots[0], torch.tensor([[[1.0, 0.0]] * 4])
        )


if __name__ == "__main__":
    unittest.main()
