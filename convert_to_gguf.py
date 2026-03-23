import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "yandex/YandexGPT-5-Lite-8B-instruct"
adapter_path = "./yandexgpt-5-lite-finetuned-final"
output_dir = Path("./yandexgpt-5-lite-finetuned-gguf")
merged_dir = output_dir.with_name(output_dir.name + "-merged")
final_gguf = output_dir / "model-full.gguf"

output_dir.mkdir(parents=True, exist_ok=True)
merged_dir.mkdir(parents=True, exist_ok=True)

print('1/4: Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.save_pretrained(merged_dir)

print('2/4: Loading base model in CPU float16...')
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print('3/4: Applying LoRA adapter and merging...')
model = PeftModel.from_pretrained(model, adapter_path, device_map={"": "cpu"})
model = model.merge_and_unload()

print('Saving merged model to', merged_dir)
model.save_pretrained(merged_dir, safe_serialization="safetensors")

print('4/4: Converting merged model to GGUF...')
converter = Path(sys.executable).resolve().parents[1] / "lib" / "python3.14" / "site-packages" / "bin" / "convert_hf_to_gguf.py"
if not converter.exists():
    raise FileNotFoundError(f"Could not find convert_hf_to_gguf.py at {converter}")

subprocess.run(
    [sys.executable, str(converter), "--outtype", "f16", "--outfile", str(final_gguf), str(merged_dir)],
    check=True,
)

print('Final GGUF saved at:', final_gguf)
