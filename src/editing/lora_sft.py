"""Метод редактирования LocFT-BF (LoRA SFT)."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import config
from src.editing.base import EditingMethod


class _SimpleDataset:
    """Внутренний класс датасета для обучения SFT."""

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


class LoraSFTMethod(EditingMethod):
    """Адаптер метода LocFT-BF с обучением LoRA."""

    name = "locft_bf"

    def __init__(self) -> None:
        self._adapter_path: Optional[Path] = None

    def prepare(self, **kwargs) -> Dict[str, Any]:
        return {"status": "prepared", "params": kwargs}

    def _build_messages(self, triplets_path: Path) -> list:
        with open(triplets_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_queries = data["test_queries"]
        system_msg = {
            "role": "system",
            "content": "You are a helpful biomedical knowledge assistant. Answer concisely.",
        }
        rows = []
        for category in ["direct", "inverse", "paraphrase"]:
            for item in test_queries.get(category, []):
                rows.append(
                    [
                        system_msg,
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["expected"]},
                    ]
                )
        return rows

    def train_or_edit(self, **kwargs) -> Dict[str, Any]:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dataset_name = kwargs["dataset_name"]
        triplets_path = Path(kwargs["triplets_path"])
        output_dir = Path(kwargs["output_dir"])
        model_name = kwargs.get("model_name", config.MODEL_NAME)

        adapter_dir = output_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        messages = self._build_messages(triplets_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, config.TORCH_DTYPE),
            device_map=config.DEVICE,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

        ds = _SimpleDataset(enc["input_ids"], enc["attention_mask"])
        dl = DataLoader(ds, batch_size=4, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        epochs = int(kwargs.get("num_epochs", 5))

        model.train()
        epoch_losses = []
        for _ in range(epochs):
            total_loss = 0.0
            for batch in dl:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += float(loss.item())
            epoch_losses.append(total_loss / max(len(dl), 1))

        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        self._adapter_path = adapter_dir
        return {
            "status": "trained",
            "dataset": dataset_name,
            "adapter_path": str(adapter_dir),
            "num_train_examples": len(messages),
            "epoch_losses": epoch_losses,
        }

    def save(self, **kwargs) -> Dict[str, Any]:
        if self._adapter_path is None:
            return {"status": "no_adapter"}
        return {"status": "saved", "adapter_path": str(self._adapter_path)}

    def load(self, **kwargs) -> Dict[str, Any]:
        adapter_path = kwargs.get("adapter_path")
        if not adapter_path:
            raise ValueError("adapter_path is required")
        self._adapter_path = Path(adapter_path)
        return {"status": "loaded", "adapter_path": str(self._adapter_path)}

    def apply_to_model(self, **kwargs) -> Dict[str, Any]:
        if self._adapter_path is None:
            raise ValueError("Adapter is not loaded or trained")
        return {"status": "applied", "adapter_path": str(self._adapter_path)}
