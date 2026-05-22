"""
Генерация тестовых запросов (direct, inverse, paraphrase) и locality-вопросов.
"""
from typing import Dict, List, Set, Tuple

Triplet = Tuple[str, str, str]

# Шаблоны вопросов. paraphrase – список вариантов для каждого типа сущности
TEMPLATES = {
    "gene": {
        "direct": "In which PubMed article is {entity} mentioned?",
        "inverse": "Which gene is mentioned in {pmid}?",
        "paraphrase": [
            "What publication cites {entity}?",
            "Which paper mentions the gene {entity}?",
            "Give the PubMed ID of a study discussing {entity}.",
            "In which article does {entity} appear?"
        ]
    },
    "disease": {
        "direct": "In which PubMed article is the disease {entity} mentioned?",
        "inverse": "Which disease is mentioned in {pmid}?",
        "paraphrase": [
            "What publication discusses the disease {entity}?",
            "Which paper mentions the condition {entity}?",
            "Give the PubMed ID of a study about {entity}.",
            "In which article does the disease {entity} appear?"
        ]
    },
    "mutation": {
        "direct": "In which PubMed article is the mutation {entity} mentioned?",
        "inverse": "Which mutation is mentioned in {pmid}?",
        "paraphrase": [
            "What publication describes the mutation {entity}?",
            "Which paper mentions the variant {entity}?",
            "Give the PubMed ID of a study discussing the mutation {entity}.",
            "In which article does the mutation {entity} appear?"
        ]
    }
}


def generate_queries(triplets: Set[Triplet], entity_type: str = "gene",
                     limit: int = 100) -> Dict[str, List[Dict]]:
    """
    Генерирует direct, inverse и несколько paraphrase запросов на каждый факт.
    Возвращает словарь с ключами 'direct', 'inverse', 'paraphrase'.
    """
    tmpl = TEMPLATES.get(entity_type, TEMPLATES["gene"])
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}

    for subj, pred, obj in list(triplets)[:limit]:
        if pred == "mentioned in":
            # прямой вопрос
            queries['direct'].append({
                "question": tmpl["direct"].format(entity=subj),
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            # обратный вопрос
            queries['inverse'].append({
                "question": tmpl["inverse"].format(pmid=obj),
                "expected": subj,
                "triplet": (subj, pred, obj)
            })
            # несколько парафраз (по одному вопросу на каждый вариант)
            for paraphrase_template in tmpl["paraphrase"]:
                queries['paraphrase'].append({
                    "question": paraphrase_template.format(entity=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj)
                })
        elif pred == "co-occurs with":
            queries['direct'].append({
                "question": f"With which {entity_type} does {subj} co-occur?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            # парафразы для ко-оккуренции (2 варианта)
            queries['paraphrase'].append({
                "question": f"What {entity_type} is co-mentioned with {subj} in the same article?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            queries['paraphrase'].append({
                "question": f"Which {entity_type} appears together with {subj} in a PubMed article?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })

    return queries


def generate_locality_queries() -> List[Dict[str, str]]:
    """
    Создаёт набор простых общих вопросов для проверки сохранения знаний (locality).
    Возвращает список словарей с ключами 'question', 'expected'.
    """
    common_knowledge = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {"question": "Who wrote the play 'Romeo and Juliet'?", "expected": "William Shakespeare"},
        {"question": "What is the chemical symbol for water?", "expected": "H2O"},
        {"question": "In which year did the Titanic sink?", "expected": "1912"},
        {"question": "What is the largest planet in our solar system?", "expected": "Jupiter"},
        {"question": "Who painted the Mona Lisa?", "expected": "Leonardo da Vinci"},
        {"question": "What is the capital of Japan?", "expected": "Tokyo"},
        {"question": "How many continents are there on Earth?", "expected": "7"},
        {"question": "What is the speed of light in vacuum (km/s)?", "expected": "300000"},
        {"question": "What is the main language spoken in Brazil?", "expected": "Portuguese"},
    ]
    return common_knowledge