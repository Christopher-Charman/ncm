"""CPU-only checkpoint gate tests; never deserialize downloaded checkpoints.

Run: python -m unittest discover -s tests -p 'test_checkpoint_security.py' -v
"""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from utils.checkpoint import load_restricted_checkpoint, torch_version_tuple


def direct_torch_load_calls(tree):
    """Find common direct loader aliases, not a full Python taint analyser."""
    torch_module_names = set()
    torch_load_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_module_names.add(alias.asname or "torch")
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            for alias in node.names:
                if alias.name == "load":
                    torch_load_names.add(alias.asname or "load")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in torch_load_names:
            yield node
        elif (isinstance(fn, ast.Attribute) and fn.attr == "load"
              and isinstance(fn.value, ast.Name)
              and fn.value.id in torch_module_names):
            yield node


class CheckpointGateTest(unittest.TestCase):
    def test_torch_release_parser(self):
        self.assertEqual(torch_version_tuple("2.10.0+cpu"), (2, 10, 0))
        self.assertEqual(torch_version_tuple("2.10.1"), (2, 10, 1))
        self.assertEqual(torch_version_tuple("3.0.0"), (3, 0, 0))
        self.assertEqual(torch_version_tuple("2.10.0+cpu"), (2, 10, 0))
        for version in ("", "nightly", "2.10", "v2.10.0", "2.10.0.dev1",
                        "3.0.0.dev1", "2.10.0rc1", "2.10.0a0+git"):
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


    def test_import_aliases_are_not_allowed_to_bypass_the_scan(self):
        for snippet in (
            "import torch; torch.load('untrusted.pt')",
            "import torch as t; t.load('untrusted.pt')",
            "from torch import load; load('untrusted.pt')",
            "from torch import load as open_weights; open_weights('untrusted.pt')",
        ):
            with self.subTest(snippet=snippet):
                calls = list(direct_torch_load_calls(ast.parse(snippet)))
                self.assertEqual(len(calls), 1)

    def test_checkpoint_calls_route_through_one_gate(self):
        root = Path(__file__).resolve().parents[1]
        for file in sorted(root.rglob("*.py")):
            if ".git" in file.parts:
                continue
            with self.subTest(source=str(file.relative_to(root))):
                tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
                for call in direct_torch_load_calls(tree):
                    self.assertEqual(file.relative_to(root).as_posix(), "utils/checkpoint.py")
                    self.assertTrue(any(
                        kw.arg == "weights_only"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                        for kw in call.keywords
                    ))


if __name__ == "__main__":
    unittest.main()
