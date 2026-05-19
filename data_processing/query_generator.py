"""
Генерация вопросов для проверки извлечённых триплетов.
"""
from typing import List, Dict, Set, Tuple

Triplet = Tuple[str, str, str]


TEMPLATES = {
    "gene": {
        "direct": "In which PubMed article is {entity} mentioned?",
        "inverse": "Which gene is mentioned in {pmid}?",
        "paraphrase": "What publication cites {entity}?"
    },
    "disease": {
        "direct": "In which PubMed article is the disease {entity} mentioned?",
        "inverse": "Which disease is mentioned in {pmid}?",
        "paraphrase": "What publication discusses the disease {entity}?"
    },
    "mutation": {
        "direct": "In which PubMed article is the mutation {entity} mentioned?",
        "inverse": "Which mutation is mentioned in {pmid}?",
        "paraphrase": "What publication describes the mutation {entity}?"
    }
}

def generate_queries(triplets: Set[Triplet], entity_type: str = "gene",
                     limit: int = 100) -> Dict[str, List[Dict]]:
    tmpl = TEMPLATES.get(entity_type, TEMPLATES["gene"])
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}

    for subj, pred, obj in list(triplets)[:limit]:
        if pred == "mentioned in":
            queries['direct'].append({
                "question": tmpl["direct"].format(entity=subj),
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            queries['inverse'].append({
                "question": tmpl["inverse"].format(pmid=obj),
                "expected": subj,
                "triplet": (subj, pred, obj)
            })
            queries['paraphrase'].append({
                "question": tmpl["paraphrase"].format(entity=subj),
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
        elif pred == "co-occurs with":
            queries['direct'].append({
                "question": f"With which {entity_type} does {subj} co-occur?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
    return queries