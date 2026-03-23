import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 0) Настройки: модель Yandex, 4-bit, короткая последовательность
model_name = "yandex/YandexGPT-5-Lite-8B-instruct"
max_seq_length = 128
sample_size = 100  # для тестового обучения на 100 примерах

# 1) Токенизатор + модель 4-bit
print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Загрузка модели Yandex GPT-5 Lite 8B (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 2) Данные
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": enc["input_ids"][0],
        }


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["question_body", "answers"])
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    df["text"] = df.apply(lambda r: f"{r.question_body.strip()} Ассистент:[SEP] {r.answers.strip()}", axis=1)
    return df["text"].tolist()

print("Загрузка данных...")
texts = load_data("psychology_data.csv")

train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
train_dataset = TextDataset(train_texts, tokenizer, max_seq_length)
val_dataset = TextDataset(val_texts, tokenizer, max_seq_length)

# 3) Тренинг
training_args = TrainingArguments(
    output_dir="./yandexgpt-5-lite-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_steps=200,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=300,
    load_best_model_at_end=False,
    fp16=True,
    bf16=False,
    optim="adamw_torch_fused",
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("Начинаем дообучение...")
trainer.train()

print("Сохранение модели...")
model.save_pretrained("./yandexgpt-5-lite-finetuned-final")
tokenizer.save_pretrained("./yandexgpt-5-lite-finetuned-final")
print("Готово")
