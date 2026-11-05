"""Resolve mutable Hub references once, before any measurement; never silently re-resolve."""
import argparse
import json
from pathlib import Path
import re


def main():
    import yaml
    from huggingface_hub import HfApi
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/opt125m_calibration.yaml')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    api = HfApi()
    for key in ('model', 'tokenizer', 'dataset'):
        spec = cfg[key]
        revision = spec['revision'] or 'main'
        info = api.dataset_info if key == 'dataset' else api.model_info
        resolved = info(spec['name'], revision=revision).sha
        if not re.fullmatch(r'[0-9a-f]{40}', resolved):
            raise ValueError('Hub did not return immutable revision')
        spec['revision'] = resolved
    # JSON is a YAML subset. Exclusive create prevents replacing a frozen protocol.
    with Path(args.output).open('x') as f:
        json.dump(cfg, f, indent=2, allow_nan=False)
    print(args.output)


if __name__ == '__main__':
    main()
