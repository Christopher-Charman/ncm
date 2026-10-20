"""CPU-only checkpoint gate tests; never deserialize downloaded checkpoints.

Run: python -m unittest discover -s tests -p 'test_checkpoint_security.py' -v
"""
import unittest
from unittest.mock import patch

from utils.checkpoint import load_restricted_checkpoint, torch_version_tuple


class CheckpointGateTest(unittest.TestCase):
    def test_torch_release_parser(self):
        self.assertEqual(torch_version_tuple("2.10.0+cpu"), (2, 10, 0))
        self.assertEqual(torch_version_tuple("2.10.1"), (2, 10, 1))
        self.assertEqual(torch_version_tuple("3.0.0.dev1"), (3, 0, 0))
        for version in ("", "nightly", "2.10", "v2.10.0"):
            with self.subTest(version=version):
                with self.assertRaises(RuntimeError):
                    torch_version_tuple(version)

    def test_known_vulnerable_versions_do_not_touch_disk(self):
        for version in ("2.6.0+cpu", "2.9.1", "2.9.2"):
            with self.subTest(version=version):
                with patch("utils.checkpoint.torch.__version__", version), \
                     patch("utils.checkpoint.torch.load") as loader:
                    with self.assertRaisesRegex(RuntimeError, "CVE-2026-24747"):
                        load_restricted_checkpoint("untrusted.pt")
                    loader.assert_not_called()

    def test_patched_version_uses_restricted_cpu_loader(self):
        marker = {"model": {"weight": 1}}
        with patch("utils.checkpoint.torch.__version__", "2.10.0+cpu"), \
             patch("utils.checkpoint.torch.load", return_value=marker) as loader:
            result = load_restricted_checkpoint("model.pt")
        self.assertIs(result, marker)
        loader.assert_called_once_with("model.pt", map_location="cpu", weights_only=True)

    def test_loader_failure_never_falls_back_to_unsafe_unpickler(self):
        with patch("utils.checkpoint.torch.__version__", "2.10.0"), \
             patch("utils.checkpoint.torch.load", side_effect=ValueError("bad checkpoint")) as loader:
            with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                load_restricted_checkpoint("broken.pt")
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
