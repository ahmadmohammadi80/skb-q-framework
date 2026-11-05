"""Standard Python launcher. Run in a fresh CUDA Colab runtime.
Usage: python colab_stage3_opt125m.py --commit <reviewed-40-character-SHA>
No mutable branch checkout, shell magics, silent failures, or automatic certification.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit', required=True)
    ap.add_argument('--workdir', default='/content/skbq-reviewed')
    ap.add_argument('--max-nodes', type=int, default=1, help='1 for smoke run; 0 for full diagnostic')
    args = ap.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.commit) or args.max_nodes < 0:
        raise ValueError('Provide immutable commit and nonnegative max-nodes')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('This environment lock was tested with Python 3.12')
    dest = Path(args.workdir).resolve()
    if dest.exists():
        raise FileExistsError('Use a fresh workdir; existing results are never reused')
    def run(command, cwd=None):
        subprocess.run(command, cwd=cwd, check=True)
    run(['git', 'clone', 'https://github.com/ahmadmohammadi80/skb-q-framework.git', str(dest)])
    run(['git', 'checkout', '--detach', args.commit], dest)
    # Isolated environment avoids Colab's preinstalled torchvision/torch mismatches.
    env = dest / '.venv'
    run([sys.executable, '-m', 'venv', str(env)])
    python = str(env / 'bin/python')
    run([python, '-m', 'pip', 'install', 'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'], dest)
    run([python, '-m', 'pip', 'install', '-r', 'requirements-lock.txt'], dest)
    run([python, '-m', 'pip', 'check'], dest)
    run([python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], dest)
    run([python, 'scripts/freeze_protocol.py', '--output', 'frozen_protocol.json'], dest)
    run([python, 'scripts/estimate_sensitivity_colab.py', '--config', 'frozen_protocol.json', '--max-nodes', str(args.max_nodes)], dest)
    runs = list((dest / 'artifacts/sensitivity').iterdir())
    if len(runs) != 1:
        raise RuntimeError('Expected exactly one new run')
    run([python, 'scripts/validate_sensitivity.py', str(runs[0])], dest)
    print('Artifact checks completed; scientific review remains pending:', runs[0])


if __name__ == '__main__':
    main()
