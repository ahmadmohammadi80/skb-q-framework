"""Auditable calibration diagnostic; never certifies manuscript claims."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['model']['name'] != 'facebook/opt-125m':
        raise ValueError('This protocol supports OPT-125M only')
    for section in ('model', 'tokenizer', 'dataset'):
        if not re.fullmatch(r'[0-9a-f]{40}', str(cfg[section]['revision'])):
            raise ValueError(f'{section}: immutable revision required; run freeze_protocol.py')
    c = cfg['calibration']
    if any(type(c[k]) is not int for k in ('samples', 'max_length', 'batch_size', 'seed')):
        raise ValueError('Calibration counts and seed must be integers')
    if c['samples'] < 1 or not 2 <= c['max_length'] <= 2048 or c['batch_size'] < 1:
        raise ValueError('Invalid sample/length/batch count')
    if cfg['tokenizer']['name'] != cfg['model']['name'] or cfg['dataset']['name'] != 'Salesforce/wikitext' or cfg['dataset']['config'] != 'wikitext-2-raw-v1' or cfg['dataset']['split'] != 'train':
        raise ValueError('Unexpected tokenizer or dataset protocol')
    if any(not isinstance(cfg['checks'][k], (int, float)) or not math.isfinite(cfg['checks'][k]) or cfg['checks'][k] < 0 for k in ('atol', 'rtol')):
        raise ValueError('Invalid control tolerance')
    bits = cfg['candidate_bits']
    if not bits or len(bits) != len(set(bits)) or any(type(b) is not int or b not in (2, 3, 4, 8) for b in bits):
        raise ValueError('Candidate bits must be unique members of 2/3/4/8')
    return cfg


def fake_quantize_weight(w, bits):
    import torch
    if bits == 16:
        return w.clone()
    if bits not in (2, 3, 4, 8):
        raise ValueError('Unsupported bit width')
    x = w.detach().float()
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite weight')
    maximum = x.abs().amax()
    if maximum.item() == 0:
        return torch.zeros_like(w)
    qmax = 2 ** (bits - 1) - 1
    scale = maximum / qmax
    # Narrow symmetric signed range; round-to-nearest, ties-to-even.
    return (torch.round(x / scale).clamp(-qmax, qmax) * scale).to(w.dtype)


def eval_loss(model, inputs, batch_size, device):
    import torch
    import torch.nn.functional as F
    records = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, inputs['input_ids'].size(0), batch_size):
            ids = inputs['input_ids'][start:start + batch_size].to(device)
            mask = inputs['attention_mask'][start:start + batch_size].to(device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            targets = ids[:, 1:].clone()
            targets[mask[:, 1:] == 0] = -100
            losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets,
                                     reduction='none', ignore_index=-100)
            for offset, (nll, count) in enumerate(zip(losses.double().sum(1), (targets != -100).sum(1))):
                n, v = int(count), float(nll)
                if n == 0 or not math.isfinite(v):
                    raise ValueError('Zero-target sample or nonfinite NLL')
                records.append({'sample_index': start + offset, 'nll_sum': v, 'target_count': n})
    return aggregate(records), records


def aggregate(records):
    count = sum(r['target_count'] for r in records)
    if count <= 0 or any(r['target_count'] <= 0 or not math.isfinite(r['nll_sum']) for r in records):
        raise ValueError('Invalid NLL records')
    return math.fsum(r['nll_sum'] for r in records) / count


def eligible_modules(model):
    import torch
    pattern = re.compile(r'model\.decoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)$')
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and pattern.fullmatch(name)]
    # Exclude head/embedding; also reject any unexpected shared selected weight.
    aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(p.data_ptr(), []).append(name)
    for name, m in modules:
        if len(aliases[m.weight.data_ptr()]) != 1:
            raise ValueError(f'Shared selected weight: {name}')
    return modules


def measure_node(model, module, bits, inputs, batch_size, device):
    import torch
    original = module.weight.detach().clone()
    try:
        with torch.no_grad():
            module.weight.copy_(fake_quantize_weight(original, bits))
        return eval_loss(model, inputs, batch_size, device)
    finally:
        with torch.no_grad():
            module.weight.copy_(original)
        if not torch.equal(module.weight, original):
            raise RuntimeError('Weight restoration failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output-root', default='artifacts/sensitivity')
    ap.add_argument('--max-nodes', type=int, default=0)
    args = ap.parse_args()
    cfg = read_config(args.config)
    if args.max_nodes < 0:
        raise ValueError('max-nodes must be nonnegative')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 for the frozen environment')
    lock = Path(__file__).resolve().parents[1] / 'requirements-lock.txt'
    for line in lock.read_text().splitlines():
        if line and not line.startswith('#'):
            package, expected = line.split('==', 1)
            actual = importlib.metadata.version(package)
            if package.lower() == 'torch':
                actual = actual.split('+')[0]
            if actual != expected:
                raise RuntimeError(f'Environment mismatch: {package} {actual}, expected {expected}')
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU tests do not count as this experiment')
    if torch.version.cuda != '12.4':
        raise RuntimeError('Frozen CUDA environment requires PyTorch cu124')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = cfg['calibration']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = Path(__file__).resolve().parents[1]
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip():
        raise RuntimeError('Tracked source modifications: commit before running')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12]
    outdir = Path(args.output_root) / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / 'status.json', {'execution_status': 'running', 'validation_status': 'pending', 'run_id': run_id})
    started = time.monotonic()
    try:
        write_json(outdir / 'effective_config.json', cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer']['name'], revision=cfg['tokenizer']['revision'])
    except Exception:
        write_json(outdir / 'status.json', {'execution_status': 'failed', 'validation_status': 'pending', 'run_id': run_id})
        raise
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg['model']['name'], revision=cfg['model']['revision'],
                    torch_dtype=torch.float16, attn_implementation='eager').to('cuda')
        ds = load_dataset(cfg['dataset']['name'], cfg['dataset']['config'],
                          split=cfg['dataset']['split'], revision=cfg['dataset']['revision'])
        indices = [i for i, row in enumerate(ds) if row['text'].strip()]
        random.Random(seed).shuffle(indices)
        c = cfg['calibration']
        indices = indices[:c['samples']]
        if len(indices) != c['samples']:
            raise ValueError('Insufficient samples')
        texts = [ds[i]['text'].strip() for i in indices]
        tokenizer.padding_side = 'right'
        inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True,
                           max_length=c['max_length'], add_special_tokens=True)
        write_json(outdir / 'samples.json', {'dataset_fingerprint': ds._fingerprint,
            'row_indices': indices, 'texts': texts, 'input_ids': inputs['input_ids'].tolist(),
            'attention_mask': inputs['attention_mask'].tolist(),
            'special_tokens_map': tokenizer.special_tokens_map, 'padding_side': 'right'})
        modules = eligible_modules(model)
        if len(modules) != 72:
            raise ValueError(f'Expected 72 block Linear modules, found {len(modules)}')
        selected = modules[:args.max_nodes] if args.max_nodes else modules
        manifest_modules = [{'name': n, 'shape': list(m.weight.shape), 'numel': m.weight.numel()} for n, m in selected]
        baseline, baseline_records = eval_loss(model, inputs, c['batch_size'], 'cuda')
        baseline_ppl = math.exp(baseline)
        if not math.isfinite(baseline_ppl):
            raise ValueError('Perplexity overflow')
        before_hash = hashlib.sha256(b''.join(m.weight.detach().cpu().numpy().tobytes() for _, m in modules)).hexdigest()
        tolerance = cfg['checks']
        checks = []
        def check(label, value):
            delta = abs(value - baseline)
            passed = math.isclose(value, baseline, abs_tol=tolerance['atol'], rel_tol=tolerance['rtol'])
            checks.append({'name': label, 'loss': value, 'absolute_delta': delta, 'passed': passed})
            if not passed:
                raise RuntimeError(f'Control failed: {label}')
        check('repeat_baseline', eval_loss(model, inputs, c['batch_size'], 'cuda')[0])
        check('batch_size_one', eval_loss(model, inputs, 1, 'cuda')[0])
        check('identity_16bit', measure_node(model, selected[0][1], 16, inputs, c['batch_size'], 'cuda')[0])
        rows = []
        for name, module in selected:
            for bits in cfg['candidate_bits']:
                loss, raw = measure_node(model, module, bits, inputs, c['batch_size'], 'cuda')
                rows.append({'module_name': name, 'bits': bits, 'quantized_loss': loss,
                             'loss_delta': loss - baseline, 'per_sample': raw})
            check('restored:' + name, eval_loss(model, inputs, c['batch_size'], 'cuda')[0])
            write_json(outdir / 'partial_rows.json', rows)
            print(f'{len(rows)} measurements completed', flush=True)
        after_hash = hashlib.sha256(b''.join(m.weight.detach().cpu().numpy().tobytes() for _, m in modules)).hexdigest()
        if before_hash != after_hash:
            raise RuntimeError('Final weight hash mismatch')
        for bits in cfg['candidate_bits']:
            ordered = sorted((r for r in rows if r['bits'] == bits), key=lambda r: (-r['loss_delta'], r['module_name']))
            for rank, row in enumerate(ordered, 1):
                row['sensitivity_rank'] = rank
        env = {'python': platform.python_version(), 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0), 'gpu_capability': list(torch.cuda.get_device_capability()),
               'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,uuid', '--format=csv,noheader'], text=True).strip(),
               'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}
        write_json(outdir / 'environment.json', env)
        write_json(outdir / 'partial_rows.json', rows)
        result = {'schema_version': 1, 'execution_status': 'completed', 'validation_status': 'pending', 'run_id': run_id,
            'experiment_type': 'calibration_diagnostic_not_allocation', 'code_commit': code_sha,
            'partial_scope': len(selected) != len(modules), 'total_eligible': len(modules),
            'requested_max_nodes': args.max_nodes, 'model_dtype': str(model.dtype),
            'quantization': 'per_tensor_symmetric_narrow_range_fp32_rtn_weight_only_fake',
            'determinism': {'enabled': True, 'attention': 'eager', 'tf32': False, 'cublas_workspace': ':4096:8'},
            'modules': manifest_modules, 'candidate_bits': cfg['candidate_bits'],
            'baseline_loss': baseline, 'calibration_perplexity': baseline_ppl,
            'baseline_per_sample': baseline_records, 'rows': rows, 'controls': checks,
            'weights_before_sha256': before_hash, 'weights_after_sha256': after_hash,
            'runtime_seconds_including_setup': time.monotonic() - started}
        write_json(outdir / 'measurements.json', result)
        write_json(outdir / 'manifest.json', {'run_id': run_id, 'files': {p.name: digest(p) for p in sorted(outdir.glob('*.json')) if p.name != 'status.json'}})
        write_json(outdir / 'status.json', {'execution_status': 'completed', 'validation_status': 'pending', 'run_id': run_id})
        print(f'COMPLETED, NOT VALIDATED: {outdir}')
    except Exception as exc:
        write_json(outdir / 'status.json', {'execution_status': 'failed', 'validation_status': 'pending', 'run_id': run_id, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
