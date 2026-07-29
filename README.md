# SKB-Q Framework

Research framework for Structural Knowledge Bridge quantization experiments.

The repository currently contains deterministic infrastructure for the SKB-Q bridge,
operator vocabulary, baseline interfaces, backbone contracts, configuration validation,
and reproducibility metadata capture. It does not contain model implementations,
experiment executions, benchmark numbers, or generated results.

## Reproducible setup

Create an isolated environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Run the unit tests:

```bash
python3 -m unittest discover -s tests
```

## Experiment configuration

Experiment configuration is validated through `skbq.config.ExperimentConfig`.
The schema records reproducibility-critical parameters without instantiating
models or running experiments:

- vocabulary registry and operator subset
- backbone encoder and policy identifiers
- budget
- temperature `tau`
- `k_prime`
- confidence threshold
- lambda weights for semantic, structural, and functional channels
- named random seeds

Example JSON shape:

```json
{
  "schema_version": "1.0",
  "experiment_id": "example-run",
  "vocabulary": {
    "registry": "default",
    "operators": ["Attention", "GQA"]
  },
  "backbone": {
    "encoder": "frozen-placeholder",
    "policy": "frozen-placeholder",
    "frozen": true
  },
  "budget": {
    "total": 8,
    "unit": "bits"
  },
  "tau": 1.0,
  "k_prime": 3,
  "confidence_threshold": 0.7,
  "lambda_weights": {
    "semantic": 1.0,
    "structural": 1.0,
    "functional": 1.0
  },
  "random_seeds": {
    "python": 0
  }
}
```

Load and validate a config:

```python
from skbq.config import load_experiment_config

config = load_experiment_config("configs/example.json")
```

## Metadata capture

Use `capture_experiment_metadata` before a run to record environment facts:

```python
from skbq.config import capture_experiment_metadata

metadata = capture_experiment_metadata(package_names=("skb-q-framework",))
```

Captured metadata includes the current git commit hash, Python version, selected
package versions, and a UTC timestamp.

## Experiment runner

`ExperimentRunner` creates reproducible result artifacts without running real
benchmarks or fabricating metric values. If no workload supplies observations,
registered metrics are serialized with `not_computed` status.

```python
from skbq.config import load_experiment_config
from skbq.experiments import ExperimentRunner

config = load_experiment_config("configs/example.json")
run = ExperimentRunner().run(config)
```

The runner writes:

```text
results/
  <experiment_id>/
    config.json
    metadata.json
    metrics.json
    warnings.json
```

Existing run directories are never overwritten. If `<experiment_id>` already
exists, a deterministic suffix such as `-001` is used for the next run.

## Results policy

Do not commit generated benchmark results or fabricated numbers. Experiment
outputs should be produced only by reproducible run scripts and stored with
their validated configuration and captured metadata.
