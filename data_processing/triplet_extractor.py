"""
Извлечение триплетов (субъект-предикат-объект) из выборки статей.
"""
from collections import defaultdict
from typing import List, Dict, Set, Tuple

Triplet = Tuple[str, str, str]


def group_entities_by_pmid(records: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Группирует уникальные имена сущностей по PMID.
    Возвращает словарь {pmid: [entity1, entity2, ...]}
    """
    entities_by_pmid = defaultdict(list)
    for rec in records:
        pmid = rec['pmid']
        entity = rec['entity_name']
        if entity not in entities_by_pmid[pmid]:
            entities_by_pmid[pmid].append(entity)
    return dict(entities_by_pmid)


def build_triplets(entities_by_pmid: Dict[str, List[str]],
                   max_cooccur_entities: int = 5,
                   predicate: str = "mentioned in",
                   cooccur_predicate: str = "co-occurs with") -> Set[Triplet]:
    """
    По сгруппированным данным строит множество уникальных триплетов:
        - (entity, predicate, "PMID:...")
        - (entityA, cooccur_predicate, entityB) для первых max_cooccur_entities сущностей
    """
    triplets: Set[Triplet] = set()

    for pmid, entities in entities_by_pmid.items():
        pmid_tag = f"PMID:{pmid}"
        # Упоминание сущности в статье
        for entity in entities:
            triplets.add((entity, predicate, pmid_tag))

        # Ко-оккуренция сущностей внутри одной статьи
        if len(entities) >= 2:
            top_entities = entities[:max_cooccur_entities]
            for i in range(len(top_entities)):
                for j in range(i + 1, len(top_entities)):
                    triplets.add((top_entities[i], cooccur_predicate, top_entities[j]))

    return triplets


def extract_sample(records, sample_size: int):
    """
    Из итератора всех записей выбирает данные по первым sample_size уникальным PMID.
    Возвращает список записей (словарей) для этих статей.
    """
    sample = []
    seen_pmids = set()
    for rec in records:
        pmid = rec['pmid']
        if pmid not in seen_pmids:
            if len(seen_pmids) >= sample_size:
                break
            seen_pmids.add(pmid)
        if pmid in seen_pmids:
            sample.append(rec)
    return sample