# Google Colab launcher for SKB-Q Stage 3
# Run in a CUDA runtime.

!pip -q install -U torch transformers datasets accelerate sentencepiece

!python scripts/estimate_sensitivity_colab.py \
  --model facebook/opt-125m \
  --samples 256 \
  --max-length 512 \
  --batch-size 8 \
  --seed 42 \
  --bits 2 3 4 8 \
  --output-dir artifacts/sensitivity

# Inspect the generated artifacts:
import json, os
for p in [
    "artifacts/sensitivity/opt125m_real_sensitivity_dataset.json",
    "artifacts/sensitivity/opt125m_sensitivity_report.json",
    "artifacts/sensitivity/opt125m_calibration_config.json",
]:
    print("\n===", p, "===")
    with open(p) as f:
        obj = json.load(f)
    if "rows" in obj:
        print("rows:", len(obj["rows"]))
        print("first row:", obj["rows"][0])
    else:
        print(json.dumps(obj, indent=2))
