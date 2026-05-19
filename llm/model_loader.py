import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, torch_dtype: str = "float16",
                             device_map: str = "auto", **kwargs):
    """Загружает модель и токенизатор."""
    print(f"Загрузка модели {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = getattr(torch, torch_dtype) if torch_dtype else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        **kwargs
    )
    print(f"Модель загружена на устройство: {model.device}, тип: {model.dtype}")
    return tokenizer, model