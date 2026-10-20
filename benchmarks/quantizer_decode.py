"""Checkpoint-free, CPU/CUDA NCM final-only quantizer benchmark.

Usage (isolated Miniconda environment with torch):
  python benchmarks/quantizer_decode.py --device cpu --batches 1,8 --stages 4,12
  python benchmarks/quantizer_decode.py --device cuda --batches 1,8,32 --stages 4,8,12 --vocab 256,1024

Measures the multiscale VQ decoder *only*, not trained-model latency or quality.
CUDA measurements synchronize explicitly. Results are JSON, with no cloud API,
model download, checkpoint deserialization, or live router interaction.
"""
import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch

from models.quant import MultiscaleResidualVectorQuantizer


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable: refusing to invent NVIDIA timings")
    return torch.device(requested)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(fn, device, warmup, repeat):
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeat):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def benchmark_one(batch, stages, vocab, embed, max_seq, device, warmup=3, repeat=12):
    if any(x < 1 for x in (batch, stages, vocab, embed, max_seq, repeat)):
        raise ValueError("batch, stages, vocab, embed, sequence and repeats must be positive")
    if stages > max_seq:
        raise ValueError("stages may not exceed final sequence length")
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    levels = [(i + 1) * max_seq // stages for i in range(stages)]
    torch.manual_seed(20261020)
    vq = MultiscaleResidualVectorQuantizer(
        vqvae_vocab_size=[vocab] * stages,
        embed_dim=embed,
        levels=SimpleNamespace(use_manual_levels=True, manual_levels=levels),
        quant_resi={"enabled": False},
    ).to(device).eval()
    ids = torch.randint(vocab, (batch, sum(levels)), device=device)

    def progressive():
        return vq.idx_to_zhats(ids, return_zlevels=True)[1][-1]

    def final_only():
        return vq.idx_to_z_q(ids)

    with torch.inference_mode():
        reference = progressive()
        candidate = final_only()
        torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
        # Alternate order to limit systematic warm-cache advantage.
        original_ms = _median_ms(progressive, device, warmup, repeat)
        final_ms = _median_ms(final_only, device, warmup, repeat)
        final_reverse_ms = _median_ms(final_only, device, warmup, repeat)
        original_reverse_ms = _median_ms(progressive, device, warmup, repeat)

    progressive_ms = statistics.median([original_ms, original_reverse_ms])
    final_only_ms = statistics.median([final_ms, final_reverse_ms])
    return {
        "batch": batch,
        "stages": stages,
        "vocab": vocab,
        "embed": embed,
        "final_seq": max_seq,
        "progressive_ms": round(progressive_ms, 5),
        "final_only_ms": round(final_only_ms, 5),
        "speedup_x": round(progressive_ms / final_only_ms, 4) if final_only_ms else None,
        "final_exact_match": True,
        "progressive_snapshot_payload_mib": round(
            batch * max_seq * embed * 4 * stages / (1024 ** 2), 3
        ),
    }


def _ints(raw):
    return [int(x.strip()) for x in raw.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--stages", default="4,12")
    parser.add_argument("--vocab", default="256")
    parser.add_argument("--embed", type=int, default=32)
    parser.add_argument("--final-seq", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    a = parser.parse_args()
    device = select_device(a.device)
    results = [
        benchmark_one(b, s, v, a.embed, a.final_seq, device, a.warmup, a.repeat)
        for b in _ints(a.batches)
        for s in _ints(a.stages)
        for v in _ints(a.vocab)
    ]
    print(json.dumps({
        "kind": "ncm_quantizer_decode_microbenchmark",
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "trained_weights": False,
        "end_to_end_measurement": False,
        "warmup": a.warmup,
        "repeat_per_order": a.repeat,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
