"""Independent structural/numerical audit; passing is not scientific certification."""
import argparse
import json
import math
import re
from pathlib import Path
from estimate_sensitivity_colab import aggregate, digest, write_json


def read(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON number')
        return result
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, parse_float=finite_float, object_pairs_hook=unique_object)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(run):
    run = Path(run)
    manifest = read(run / 'manifest.json')
    required = {'effective_config.json', 'samples.json', 'environment.json', 'measurements.json', 'partial_rows.json'}
    require(set(manifest['files']) == required, 'Incomplete manifest')
    for name, expected in manifest['files'].items():
        require(Path(name).name == name, 'Unsafe manifest path')
        require(digest(run / name) == expected, f'Hash mismatch: {name}')
    status, data, cfg, samples = (read(run / f) for f in ('status.json', 'measurements.json', 'effective_config.json', 'samples.json'))
    require(status['execution_status'] == data['execution_status'] == 'completed', 'Incomplete execution')
    require(status['run_id'] == data['run_id'] == manifest['run_id'], 'Mixed runs')
    require(data['validation_status'] == 'pending', 'Producer cannot certify itself')
    require(data['schema_version'] == 1, 'Unsupported schema')
    require(re.fullmatch(r'[0-9a-f]{40}', data['code_commit']) is not None, 'Invalid source commit')
    for section in ('model', 'tokenizer', 'dataset'):
        require(re.fullmatch(r'[0-9a-f]{40}', str(cfg[section]['revision'])) is not None, 'Mutable asset revision')
    count = cfg['calibration']['samples']
    require(len(samples['row_indices']) == len(set(samples['row_indices'])) == count, 'Invalid sample selection')
    require(len(samples['input_ids']) == len(samples['attention_mask']) == len(samples['texts']) == count, 'Sample count mismatch')
    for ids, mask in zip(samples['input_ids'], samples['attention_mask']):
        require(len(ids) == len(mask) == cfg['calibration']['max_length'], 'Token length mismatch')
        require(all(v in (0, 1) for v in mask) and mask == sorted(mask, reverse=True), 'Invalid right padding')
    expected_counts = [sum(mask[1:]) for mask in samples['attention_mask']]
    def check_records(records):
        require([r['sample_index'] for r in records] == list(range(count)), 'Sample order mismatch')
        require([r['target_count'] for r in records] == expected_counts, 'Shifted token count mismatch')
        return aggregate(records)
    base = check_records(data['baseline_per_sample'])
    require(math.isclose(base, data['baseline_loss'], rel_tol=1e-12, abs_tol=1e-12), 'Baseline aggregate mismatch')
    require(math.isclose(math.exp(base), data['calibration_perplexity'], rel_tol=1e-12), 'PPL mismatch')
    names = [m['name'] for m in data['modules']]
    require(len(names) == len(set(names)) > 0, 'Invalid module list')
    expected_names = {f'model.decoder.layers.{i}.{op}' for i in range(12) for op in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', 'fc1', 'fc2')}
    require(set(names) <= expected_names, 'Unexpected module')
    for module in data['modules']:
        expected_shape = [3072, 768] if module['name'].endswith('.fc1') else ([768, 3072] if module['name'].endswith('.fc2') else [768, 768])
        require(module['shape'] == expected_shape and module['numel'] == math.prod(expected_shape), 'Wrong parameter shape/count')
    require(data['total_eligible'] == 72 and len(names) <= 72, 'Invalid module scope')
    require(data['partial_scope'] == (len(names) != 72), 'Partial scope mislabeled')
    bits = cfg['candidate_bits']
    require(data['candidate_bits'] == bits and len(set(bits)) == len(bits), 'Bit mismatch')
    expected = {(name, b) for name in names for b in bits}
    pairs = [(r['module_name'], r['bits']) for r in data['rows']]
    require(len(pairs) == len(expected) and set(pairs) == expected, 'Missing or duplicate measurements')
    for row in data['rows']:
        loss = check_records(row['per_sample'])
        require(math.isclose(loss, row['quantized_loss'], rel_tol=1e-12, abs_tol=1e-12), 'Loss mismatch')
        require(math.isclose(loss - base, row['loss_delta'], rel_tol=1e-12, abs_tol=1e-12), 'Delta mismatch')
    for b in bits:
        rows = sorted((r for r in data['rows'] if r['bits'] == b), key=lambda r: (-r['loss_delta'], r['module_name']))
        require([r['sensitivity_rank'] for r in rows] == list(range(1, len(rows) + 1)), 'Rank mismatch')
    require(re.fullmatch(r'[0-9a-f]{64}', data['weights_before_sha256']) is not None, 'Invalid weight hash')
    require(data['weights_before_sha256'] == data['weights_after_sha256'], 'Restoration mismatch')
    expected_controls = {'repeat_baseline', 'batch_size_one', 'identity_16bit'} | {'restored:' + n for n in names}
    require(len(data['controls']) == len(expected_controls) and {c['name'] for c in data['controls']} == expected_controls, 'Missing/duplicate controls')
    for c in data['controls']:
        require(c['passed'] and math.isclose(c['loss'], base, **{'abs_tol': cfg['checks']['atol'], 'rel_tol': cfg['checks']['rtol']}), 'Failed control')
    require(read(run / 'partial_rows.json') == data['rows'], 'Checkpoint/final measurements disagree')
    return {'artifact_checks': 'passed', 'scientific_validation': 'pending', 'partial_scope': data['partial_scope'],
            'run_id': data['run_id'], 'manifest_sha256': digest(run / 'manifest.json'),
            'note': 'Integrity and arithmetic checks cannot prove authenticity or manuscript superiority.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_directory')
    args = ap.parse_args()
    report = validate(args.run_directory)
    write_json(Path(args.run_directory) / 'validation_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
