"""
Генерация вопросов для проверки извлечённых триплетов.
"""
from typing import List, Dict, Set, Tuple

Triplet = Tuple[str, str, str]


def generate_queries(triplets: Set[Triplet],
                     limit: int = 100) -> Dict[str, List[Dict]]:
    """
    Создаёт direct, inverse и paraphrase вопросы по первым `limit` триплетам.
    Возвращает словарь с ключами 'direct', 'inverse', 'paraphrase'.
    Каждый элемент – список словарей с 'question', 'expected', 'triplet'.
    """
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}

    for subj, pred, obj in list(triplets)[:limit]:
        if pred == "mentioned in":
            queries['direct'].append({
                "question": f"In which PubMed article is {subj} mentioned?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            queries['inverse'].append({
                "question": f"Which gene is mentioned in {obj}?",
                "expected": subj,
                "triplet": (subj, pred, obj)
            })
            queries['paraphrase'].append({
                "question": f"What publication cites {subj}?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
        elif pred == "co-occurs with":
            queries['direct'].append({
                "question": f"With which gene does {subj} co-occur?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })

    return queries