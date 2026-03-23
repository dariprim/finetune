import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base='yandex/YandexGPT-5-Lite-8B-instruct'
adapter='./yandexgpt-5-lite-finetuned-final'
out='./yandexgpt-5-lite-finetuned-gguf-merged'

os.makedirs(out, exist_ok=True)

try:
    print('Loading tokenizer...')
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.save_pretrained(out)
    print('Loading base model...')
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map={'': 'cpu'}, trust_remote_code=True, low_cpu_mem_usage=True)
    print('Applying LoRA...')
    model = PeftModel.from_pretrained(model, adapter, device_map={'': 'cpu'})
    print('Merging...')
    model = model.merge_and_unload()
    print('Saving merged...')
    model.save_pretrained(out, safe_serialization='safetensors')
    print('Done success')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('FAILED', type(e), e)
