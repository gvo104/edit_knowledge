"""
Дообучение модели с помощью LoRA на SFT-датасете (LocFT-BF).
Может автоматически запустить probing после обучения.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json
import subprocess
from pathlib import Path
import sys
import argparse
from torch.utils.data import Dataset, DataLoader

# Импортируем конфиг
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config
from model_training.prepare_sft_dataset import create_sft_dataset

class SimpleDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx]
        }

def train_lora(dataset_name: str, version: str = None, probe_after: bool = True,
               force_retrain: bool = False):
    # 1. Подготовка данных
    sft_path = create_sft_dataset(dataset_name, version)

    # 2. Путь для сохранения адаптера
    adapter_dir = Path(config.BASE_DIR) / "data" / "lora_adapters" / f"{dataset_name}_{config.VERSION_TIMESTAMP}"

    if adapter_dir.exists() and not force_retrain:
        print(f"Адаптер уже существует: {adapter_dir}")
        if probe_after:
            _run_probing(dataset_name, str(adapter_dir), version)
        return adapter_dir

    # 3. Загрузка модели и токенизатора
    print(f"Загрузка модели {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=getattr(torch, config.TORCH_DTYPE),
        device_map=config.DEVICE,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Применяем LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Загружаем датасет
    with open(sft_path, 'r', encoding='utf-8') as f:
        messages_list = json.load(f)
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in messages_list]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    dataset = SimpleDataset(encodings["input_ids"], encodings["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 6. Обучение вручную
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    num_epochs = 5
    model.train()
    print("Начинаем обучение LoRA...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"  Эпоха {epoch+1}/{num_epochs}, средний loss: {avg_loss:.4f}")

    # 7. Сохраняем адаптер
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Адаптер сохранён в {adapter_dir}")

    # 8. Probing после обучения
    if probe_after:
        _run_probing(dataset_name, str(adapter_dir), version)

    return adapter_dir

def _run_probing(dataset_name: str, lora_path: str, version: str = None):
    probing_script = PROJECT_ROOT / "model_probing" / "run_probing.py"
    cmd = [sys.executable, str(probing_script), '--dataset', dataset_name,
           '--lora_path', lora_path]
    if version:
        cmd += ['--version', version]
    print("\n>>> Запуск probing на дообученной модели...")
    subprocess.run(cmd, cwd=PROJECT_ROOT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение LoRA на SFT-датасете")
    parser.add_argument('--dataset', type=str, default=config.DATASET_NAME)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--no_probe', action='store_true', help='Не запускать probing после обучения')
    parser.add_argument('--force_retrain', action='store_true', help='Переобучить, даже если адаптер существует')
    parser.add_argument('--lora_path', type=str, default=None, help='Путь к готовому адаптеру (обучение пропускается, используется этот)')
    args = parser.parse_args()

    if args.lora_path:
        if not args.no_probe:
            _run_probing(args.dataset, args.lora_path, args.version)
        else:
            print("Адаптер указан, но probing отключён.")
    else:
        train_lora(args.dataset, args.version, probe_after=not args.no_probe,
                   force_retrain=args.force_retrain)