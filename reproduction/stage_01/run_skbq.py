
# ============================================
# SKB-Q برای OPT-125M (نسخه نهایی پایدار)
# ============================================
# برای اجرا در کولب: کافیست این فایل را اجرا کنید
# مدل: facebook/opt-125m

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import requests
import os
import re
import json
import random
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

# ============================================
# تنظیمات
# ============================================
MODEL_NAME = "facebook/opt-125m"  # برای 1.3B این را تغییر دهید
CANDIDATE_BITS = [4, 8]
TARGET_BITS = 4.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_SAMPLES = 256
NUM_SALIENCE_SAMPLES = 32
BITS_TO_QUANTIZE = [4]
NUM_LAYERS = 12  # برای 1.3B این را به 24 تغییر دهید

print("=" * 60)
print("🚀 SKB-Q روی", MODEL_NAME)
print("   بیت‌های کاندید:", CANDIDATE_BITS)
print("   بودجه هدف:", TARGET_BITS)
print("=" * 60)

# ============================================
# 1. بارگذاری مدل
# ============================================
print("\n[1/6] بارگذاری مدل...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("✅ مدل بارگذاری شد")

# ============================================
# 2. بارگذاری دیتاست
# ============================================
print("\n[2/6] بارگذاری دیتاست...")
parquet_path = "/tmp/wikitext_valid.parquet"
if not os.path.exists(parquet_path):
    url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    response = requests.get(url)
    with open(parquet_path, "wb") as f:
        f.write(response.content)

df = pd.read_parquet(parquet_path)
texts = [str(t) for t in df["text"].tolist() if len(str(t).strip()) > 0][:NUM_EVAL_SAMPLES]
print(f"✅ {len(texts)} نمونه بارگذاری شد")

# ============================================
# 3. استخراج اپراتورها
# ============================================
print("\n[3/6] استخراج اپراتورهای خطی...")
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        linear_layers.append((name, module))
print(f"✅ {len(linear_layers)} اپراتور خطی پیدا شد")

# ============================================
# 4. امتیاز ساختاری
# ============================================
def compute_structure_scores(layers, num_layers=12):
    scores = {}
    for name, module in layers:
        if 'q_proj' in name or 'k_proj' in name:
            type_score = 0.9
        elif name.rsplit('.', 1)[-1] in {'v_proj', 'o_proj', 'out_proj'}:
            type_score = 0.7
        elif 'fc1' in name or 'fc2' in name:
            type_score = 0.3
        else:
            type_score = 0.5
        
        match = re.search(r'(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)', name)
        layer_idx = int(match.group(1)) if match else 0
        depth_norm = layer_idx / (num_layers - 1) if num_layers > 1 else 0.0
        
        raw_score = 0.5 * type_score + 0.3 * depth_norm + 0.2 * depth_norm
        scores[name] = raw_score
    
    min_val = min(scores.values())
    max_val = max(scores.values())
    if max_val > min_val:
        for k in scores:
            scores[k] = (scores[k] - min_val) / (max_val - min_val)
    return scores

structure_scores = compute_structure_scores(linear_layers, NUM_LAYERS)
print(f"✅ امتیاز ساختاری برای {len(structure_scores)} اپراتور محاسبه شد")

# ============================================
# 5. سالینس
# ============================================
def compute_salience_with_hooks(model, layers, tokenizer, texts, num_samples=32):
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                act_norm = torch.norm(output.float()).item()
                if name not in activations:
                    activations[name] = []
                activations[name].append(act_norm)
        return fn
    
    for name, module in layers:
        hook = module.register_forward_hook(hook_fn(name))
        hooks.append(hook)
    
    model.eval()
    sample_texts = random.sample(texts, min(num_samples, len(texts)))
    
    with torch.no_grad():
        for t in tqdm(sample_texts, desc="  محاسبه سالینس"):
            inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            _ = model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    scores = {}
    for name, module in layers:
        if name in activations:
            act_norm = np.mean(activations[name])
        else:
            act_norm = 1.0
        weight_norm = torch.norm(module.weight.float()).item()
        scores[name] = act_norm * weight_norm
    
    min_val = min(scores.values())
    max_val = max(scores.values())
    if max_val > min_val:
        for k in scores:
            scores[k] = (scores[k] - min_val) / (max_val - min_val)
    else:
        for k in scores:
            scores[k] = 0.5
    return scores

print("\n[4/6] محاسبه سالینس با هوک...")
salience_scores = compute_salience_with_hooks(model, linear_layers, tokenizer, texts, NUM_SALIENCE_SAMPLES)
print(f"✅ سالینس برای {len(salience_scores)} اپراتور محاسبه شد")

# ============================================
# 6. تابع تخصیص رتبه‌ای
# ============================================
def allocate_bits_by_rank(scores, target_bits, candidate_bits):
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    num_layers = len(sorted_items)
    num_8bit = int(round((target_bits - 4) * num_layers / 4))
    num_8bit = max(0, min(num_8bit, num_layers))
    
    allocation = {}
    for i, (name, score) in enumerate(sorted_items):
        if i < num_8bit:
            allocation[name] = 8
        else:
            allocation[name] = 4
    
    avg_bit = (num_8bit * 8 + (num_layers - num_8bit) * 4) / num_layers
    return allocation, avg_bit

# ============================================
# 7. کوانتیزاسیون
# ============================================
def quantize_weight(weight, bits):
    if bits >= 16:
        return weight
    max_val = weight.abs().max(dim=1, keepdim=True)[0]
    scale = max_val / (2**(bits-1) - 1)
    scale = torch.clamp(scale, min=1e-10)
    return torch.round(weight / scale) * scale

def apply_quantization(model, allocation, bits_to_quantize=[4]):
    quantized_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in allocation:
            bits = allocation[name]
            if bits in bits_to_quantize:
                module.weight.data = quantize_weight(module.weight.data, bits)
                quantized_count += 1
    return quantized_count

def evaluate_ppl(model, tokenizer, texts, max_length=512, stride=512):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="  ارزیابی PPL"):
            if len(text.strip()) == 0:
                continue
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs.input_ids.to(DEVICE)
            
            if input_ids.shape[1] < 2:
                continue
            
            for i in range(0, input_ids.shape[1], stride):
                end = min(i + max_length, input_ids.shape[1])
                if end - i < 2:
                    continue
                seq = input_ids[:, i:end]
                outputs = model(seq, labels=seq)
                total_nll += outputs.loss.item() * (end - i)
                total_tokens += end - i
    
    if total_tokens == 0:
        return float('inf')
    return np.exp(total_nll / total_tokens)

# ============================================
# 8. اجرای اصلی
# ============================================
print("\n[5/6] آماده‌سازی امتیازها...")
combined_scores = {}
for name in set(structure_scores.keys()) | set(salience_scores.keys()):
    combined_scores[name] = 0.5 * structure_scores.get(name, 0.5) + 0.5 * salience_scores.get(name, 0.5)

random_scores = {name: random.random() for name, _ in linear_layers}

methods = {
    "Structure-Only": structure_scores,
    "Salience-Only": salience_scores,
    "Full-Framework": combined_scores,
    "Random": random_scores
}

results = {}

print("\n[6/6] اجرای تخصیص بیت و ارزیابی...")

for method_name, scores in methods.items():
    print(f"\n  ▶ {method_name}")
    
    allocation, avg_bit = allocate_bits_by_rank(scores, TARGET_BITS, CANDIDATE_BITS)
    print(f"    میانگین بیت: {avg_bit:.2f}")
    
    bit_counts = {b: 0 for b in CANDIDATE_BITS}
    for b in allocation.values():
        bit_counts[b] = bit_counts.get(b, 0) + 1
    print(f"    توزیع بیت‌ها: {bit_counts}")
    
    model_copy = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    
    quant_count = apply_quantization(model_copy, allocation, BITS_TO_QUANTIZE)
    print(f"    لایه‌های کوانتیزه شده (بیت ۴): {quant_count}")
    
    ppl = evaluate_ppl(model_copy, tokenizer, texts)
    results[method_name] = {
        "avg_bits": avg_bit,
        "ppl": ppl,
        "quantized_count": quant_count,
        "bit_distribution": bit_counts
    }
    print(f"    ✅ PPL: {ppl:.2f}")
    
    del model_copy
    gc.collect()
    torch.cuda.empty_cache()

# نمایش نتایج
print("\n" + "=" * 60)
print("📊 نتایج نهایی")
print("=" * 60)
print(f"\n{'Method':<20} {'AvgBit':<10} {'PPL':<10} {'Quantized':<10}")
print("-" * 55)
for method, data in results.items():
    print(f"{method:<20} {data['avg_bits']:<10.2f} {data['ppl']:<10.2f} {data['quantized_count']:<10}")

with open("/content/skbq_results_final.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✅ نتایج ذخیره شد")
