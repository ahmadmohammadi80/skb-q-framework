"""Stage 3: real calibration-based node-wise sensitivity estimation for OPT-125M.

Designed for Google Colab with a CUDA runtime. This script intentionally performs
real forward passes; it does not generate or assume placeholder sensitivity values.
"""
import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fake_quantize_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor fake quantization, returning dequantized weights."""
    if bits >= 16:
        return w
    qmax = (2 ** (bits - 1)) - 1
    scale = w.detach().abs().amax().clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale


def make_calibration(tokenizer, samples: int, max_length: int, seed: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"].strip() for x in ds if x["text"].strip()]
    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:samples]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    # Ignore padding in the causal-LM loss.
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


@torch.inference_mode()
def eval_loss(model, input_ids, attention_mask, labels, batch_size, device):
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    for start in range(0, input_ids.size(0), batch_size):
        end = min(start + batch_size, input_ids.size(0))
        out = model(
            input_ids=input_ids[start:end].to(device),
            attention_mask=attention_mask[start:end].to(device),
            labels=labels[start:end].to(device),
        )
        n = (labels[start:end] != -100).sum().item()
        total_loss += float(out.loss) * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4, 8])
    ap.add_argument("--max-nodes", type=int, default=0, help="0 = all eligible Linear modules")
    ap.add_argument("--output-dir", default="artifacts/sensitivity")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA Colab runtime is required for the intended experiment.")
    seed_everything(args.seed)
    device = torch.device("cuda")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)
    input_ids, attention_mask, labels = make_calibration(tokenizer, args.samples, args.max_length, args.seed)

    baseline = eval_loss(model, input_ids, attention_mask, labels, args.batch_size, device)
    baseline_ppl = float(np.exp(min(baseline, 20)))

    linear_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    if args.max_nodes:
        linear_modules = linear_modules[:args.max_nodes]

    rows = []
    start_time = time.time()
    for idx, (name, module) in enumerate(linear_modules, 1):
        original = module.weight.data.clone()
        for bits in args.bits:
            module.weight.data.copy_(fake_quantize_weight(original, bits))
            loss = eval_loss(model, input_ids, attention_mask, labels, args.batch_size, device)
            delta = loss - baseline
            rows.append({
                "node_index": idx,
                "module_name": name,
                "bits": bits,
                "baseline_loss": baseline,
                "quantized_loss": loss,
                "loss_delta": delta,
                "sensitivity_score": delta,
            })
            module.weight.data.copy_(original)
        print(f"[{idx}/{len(linear_modules)}] {name}")

    # Rank by worst measured degradation at each candidate bit width.
    for bits in args.bits:
        subset = [r for r in rows if r["bits"] == bits]
        subset.sort(key=lambda r: r["sensitivity_score"], reverse=True)
        for rank, row in enumerate(subset, 1):
            row["sensitivity_rank"] = rank

    meta = {
        "validated": True,
        "experiment_type": "calibration_based_nodewise_fake_quantization",
        "model": args.model,
        "dataset": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "train",
        "samples": args.samples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "candidate_bits": args.bits,
        "quantization": "symmetric_per_tensor_weight_only_fake_quant",
        "baseline_loss": baseline,
        "baseline_perplexity": baseline_ppl,
        "num_eligible_linear_modules": len(linear_modules),
        "runtime_seconds": time.time() - start_time,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "note": "Sensitivity is measured as calibration loss increase from temporary node-wise fake quantization. It is not a final mixed-precision allocation result.",
    }

    with open(outdir / "opt125m_real_sensitivity_dataset.json", "w") as f:
        json.dump({"metadata": meta, "rows": rows}, f, indent=2)
    with open(outdir / "opt125m_sensitivity_report.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(outdir / "opt125m_calibration_config.json", "w") as f:
        json.dump({
            "model": args.model,
            "dataset": "wikitext/wikitext-2-raw-v1",
            "split": "train",
            "samples": args.samples,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "candidate_bits": args.bits,
            "quantization": "symmetric_per_tensor_weight_only_fake_quant",
        }, f, indent=2)

    print("\nDONE")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
