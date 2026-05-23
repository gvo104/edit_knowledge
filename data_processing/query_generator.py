"""
Генерация тестовых запросов (direct, inverse, paraphrase) и locality-вопросов.
"""
from typing import Dict, List, Set, Tuple

Triplet = Tuple[str, str, str]

# Полные шаблоны (7 парафраз для каждого типа)
TEMPLATES = {
    "gene": {
        "direct": "In which PubMed article is {entity} mentioned?",
        "inverse": "Which gene is mentioned in {pmid}?",
        "paraphrase": [
            "What publication cites {entity}?",
            "Which paper mentions the gene {entity}?",
            "Give the PubMed ID of a study discussing {entity}.",
            "In which article does {entity} appear?",
            "What is the PubMed ID for the paper about {entity}?",
            "Which scientific article reports on {entity}?",
            "Provide the PMID of a study that examines {entity}.",
        ]
    },
    "disease": {
        "direct": "In which PubMed article is the disease {entity} mentioned?",
        "inverse": "Which disease is mentioned in {pmid}?",
        "paraphrase": [
            "What publication discusses the disease {entity}?",
            "Which paper mentions the condition {entity}?",
            "Give the PubMed ID of a study about {entity}.",
            "In which article does the disease {entity} appear?",
            "What is the PMID for the paper focusing on {entity}?",
            "Which scientific article investigates {entity}?",
            "Provide the PubMed ID of research concerning {entity}.",
        ]
    },
    "mutation": {
        "direct": "In which PubMed article is the mutation {entity} mentioned?",
        "inverse": "Which mutation is mentioned in {pmid}?",
        "paraphrase": [
            "What publication describes the mutation {entity}?",
            "Which paper mentions the variant {entity}?",
            "Give the PubMed ID of a study discussing the mutation {entity}.",
            "In which article does the mutation {entity} appear?",
            "What is the PMID for the paper about the mutation {entity}?",
            "Which scientific article reports on {entity}?",
            "Provide the PubMed ID of a study that examines the mutation {entity}.",
        ]
    }
}


def generate_queries(triplets: Set[Triplet], entity_type: str = "gene",
                     limit: int = 100, augment: bool = False) -> Dict[str, List[Dict]]:
    """
    Генерирует direct, inverse и paraphrase запросы.
    Если augment=True, используется расширенный набор парафраз (7 вместо 4).
    """
    tmpl = TEMPLATES.get(entity_type, TEMPLATES["gene"])
    queries = {'direct': [], 'inverse': [], 'paraphrase': []}

    # Определяем количество парафраз в зависимости от флага
    paraphrase_templates = tmpl["paraphrase"] if augment else tmpl["paraphrase"][:4]

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
            for pt in paraphrase_templates:
                queries['paraphrase'].append({
                    "question": pt.format(entity=subj),
                    "expected": obj,
                    "triplet": (subj, pred, obj)
                })
        elif pred == "co-occurs with":
            queries['direct'].append({
                "question": f"With which {entity_type} does {subj} co-occur?",
                "expected": obj,
                "triplet": (subj, pred, obj)
            })
            # Ко-оккуренционные парафразы (тоже зависят от augment)
            cooccur_templates = [
                f"What {entity_type} is co-mentioned with {subj} in the same article?",
                f"Which {entity_type} appears together with {subj} in a PubMed article?",
                f"Name a {entity_type} that co-occurs with {subj} in a publication.",
                f"With which {entity_type} does {subj} share a PubMed article?",
            ]
            if augment:
                cooccur_templates += [
                    f"Identify a {entity_type} that is mentioned alongside {subj} in a paper.",
                    f"List a {entity_type} that co-appears with {subj} in PubMed.",
                ]
            for pt in cooccur_templates:
                queries['paraphrase'].append({
                    "question": pt,
                    "expected": obj,
                    "triplet": (subj, pred, obj)
                })

    return queries


def generate_locality_queries() -> List[Dict[str, str]]:
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