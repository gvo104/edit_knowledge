"""
Функции для генерации ответов модели и оценки правильности.
Поддерживает одиночные запросы и пакетную обработку.
"""
import re
import torch
from typing import List, Optional

def get_model_answer(question: str, tokenizer, model,
                     max_new_tokens: int = 50,
                     temperature: float = 0.1,
                     do_sample: bool = False) -> str:
    """Одиночный запрос (для совместимости) — без лишних предупреждений."""
    messages = [
        {"role": "system", "content": "You are a helpful biomedical knowledge assistant. Answer concisely."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Явные параметры генерации — убираем предупреждения при жадном режиме
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        # top_p, top_k оставляем по умолчанию (или можно задать явно, если нужно)
    else:
        generate_kwargs["temperature"] = None
        generate_kwargs["top_p"] = None
        generate_kwargs["top_k"] = None

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    input_len = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def get_model_answers_batch(questions: List[str], tokenizer, model,
                            max_new_tokens: int = 50,
                            batch_size: int = 8,
                            do_sample: bool = False,
                            temperature: float = 0.1) -> List[str]:
    """
    Пакетная генерация ответов для списка вопросов.
    Значительно ускоряет обработку за счёт параллельного прогона нескольких примеров.
    Все предупреждения подавлены, параметры задаются явно.
    """
    responses = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        # Формируем сообщения для каждого вопроса
        messages = [
            [
                {"role": "system", "content": "You are a helpful biomedical knowledge assistant. Answer concisely."},
                {"role": "user", "content": q}
            ]
            for q in batch
        ]
        # Применяем chat template ко всем сразу (пакетно)
        texts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Токенизируем с padding и truncation
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # Генерационные параметры — аналогично одиночной функции
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["temperature"] = None
            generate_kwargs["top_p"] = None
            generate_kwargs["top_k"] = None

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        # Декодируем ответы, удаляя входные токены
        input_lengths = inputs['attention_mask'].sum(dim=1)
        for j, out in enumerate(outputs):
            new_tokens = out[input_lengths[j]:]
            ans = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(ans)
    return responses


# --- Функции оценки без изменений ---
def evaluate_answer_basic(model_answer: str, expected: str) -> float:
    model_norm = model_answer.lower().strip()
    expected_norm = expected.lower().strip()
    if model_norm == expected_norm:
        return 1.0
    if expected_norm in model_norm:
        return 0.8
    if model_norm in expected_norm:
        return 0.6
    return 0.0


def evaluate_answer_improved(model_answer: str, expected: str) -> float:
    model_norm = model_answer.lower().strip()
    expected_norm = expected.lower().strip()
    if model_norm == expected_norm:
        return 1.0
    if expected_norm in model_norm:
        return 0.9

    pmid_pattern = r'PMID:?\s*(\d+)'
    pmid_match = re.search(pmid_pattern, model_norm, re.IGNORECASE)
    expected_pmid = re.search(pmid_pattern, expected_norm, re.IGNORECASE)
    if pmid_match and expected_pmid and pmid_match.group(1) == expected_pmid.group(1):
        return 0.85

    gene_pattern = r'\b([A-Z0-9]+(?:[A-Z0-9]+)?)\b'
    gene_match = re.search(gene_pattern, model_norm)
    expected_gene = re.search(gene_pattern, expected_norm)
    if gene_match and expected_gene and gene_match.group(1) == expected_gene.group(1):
        return 0.7

    return 0.0


def evaluate_answer(model_answer: str, expected: str, strategy: str = "improved") -> float:
    if strategy == "basic":
        return evaluate_answer_basic(model_answer, expected)
    return evaluate_answer_improved(model_answer, expected)