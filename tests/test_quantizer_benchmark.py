"""Checkpoint-free smoke for the portable CPU/CUDA benchmark driver."""
import unittest
from unittest.mock import patch

import torch

from benchmarks.quantizer_decode import benchmark_one, select_device


class QuantizerBenchmarkTest(unittest.TestCase):
    def test_cpu_matrix_returns_exactly_matching_latents(self):
        result = benchmark_one(
            batch=1, stages=3, vocab=8, embed=4, max_seq=12,
            device=torch.device("cpu"), warmup=1, repeat=2,
        )
        self.assertTrue(result["final_exact_match"])
        self.assertEqual(result["stages"], 3)
        self.assertEqual(result["final_seq"], 12)
        self.assertGreater(result["progressive_ms"], 0)
        self.assertGreater(result["final_only_ms"], 0)

    def test_unavailable_cuda_refuses_to_report_gpu_measurements(self):
        with patch("benchmarks.quantizer_decode.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU unavailable"):
                select_device("cuda")
            self.assertEqual(select_device("auto"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
