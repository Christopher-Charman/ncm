"""CPU-only checkpoint gate tests; never deserialize downloaded checkpoints.

Run: python -m unittest discover -s tests -p 'test_checkpoint_security.py' -v
"""
import ast
from pathlib import Path
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


    def test_checkpoint_calls_route_through_one_gate(self):
        root = Path(__file__).resolve().parents[1]
        for file in sorted(root.rglob("*.py")):
            if ".git" in file.parts:
                continue
            with self.subTest(source=str(file.relative_to(root))):
                tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    fn = node.func
                    direct_load = (
                        isinstance(fn, ast.Attribute) and fn.attr == "load"
                        and isinstance(fn.value, ast.Name) and fn.value.id == "torch"
                    )
                    if direct_load:
                        self.assertEqual(file.relative_to(root).as_posix(), "utils/checkpoint.py")
                        self.assertTrue(any(
                            kw.arg == "weights_only"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                            for kw in node.keywords
                        ))


if __name__ == "__main__":
    unittest.main()
