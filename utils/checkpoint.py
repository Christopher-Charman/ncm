"""Restricted PyTorch checkpoint loading for NCM research.

PyTorch <=2.9.1 is affected by CVE-2026-24747 even with
weights_only=True (GHSA-63cw-57p8-fm3p). Never silently fall back to
unsafe unpickling or treat a known hash as proof of safety.

This gate does not make arbitrary third-party model files trustworthy.
Review artifact provenance and resource limits independently.
"""
import re

import torch


MINIMUM_RESTRICTED_TORCH = (2, 10, 0)


def torch_version_tuple(version):
    """Read stable numeric prefix, including builds such as '2.10.0+cpu'."""
    match = re.match(r"^(\\d+)\\.(\\d+)\\.(\\d+)(?:$|[+.-])", str(version))
    if match is None:
        raise RuntimeError("Cannot establish installed PyTorch security version")
    return tuple(int(part) for part in match.groups())


def load_restricted_checkpoint(path):
    """Load checkpoint tensors on CPU, refusing known-vulnerable PyTorch.

    There is intentionally no override for the vulnerable version gate.
    This is not a validator for arbitrary pickle or untrusted artifacts.
    """
    installed = torch_version_tuple(torch.__version__)
    if installed < MINIMUM_RESTRICTED_TORCH:
        raise RuntimeError(
            "NCM checkpoint loading disabled: PyTorch <=2.9.1 has a "
            "weights_only deserialization vulnerability (CVE-2026-24747). "
            "Use a separately qualified PyTorch >=2.10.0 environment; "
            "do not fall back to weights_only=False."
        )
    return torch.load(path, map_location="cpu", weights_only=True)
