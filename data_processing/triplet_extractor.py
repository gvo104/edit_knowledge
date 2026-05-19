"""
Извлечение триплетов (субъект-предикат-объект) из выборки статей.
"""
from collections import defaultdict
from typing import List, Dict, Tuple, Set

Triplet = Tuple[str, str, str]


def group_genes_by_pmid(records: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Группирует уникальные имена генов по PMID.
    Возвращает словарь {pmid: [gene1, gene2, ...]}
    """
    genes_by_pmid = defaultdict(list)
    for rec in records:
        pmid = rec['pmid']
        gene = rec['gene_name']
        if gene not in genes_by_pmid[pmid]:
            genes_by_pmid[pmid].append(gene)
    return dict(genes_by_pmid)


def build_triplets(genes_by_pmid: Dict[str, List[str]],
                   max_cooccur_genes: int = 5) -> Set[Triplet]:
    """
    По сгруппированным данным строит множество уникальных триплетов:
        - (gene, "mentioned in", "PMID:...")
        - (geneA, "co-occurs with", geneB) для первых max_cooccur_genes генов
    """
    triplets: Set[Triplet] = set()

    for pmid, genes in genes_by_pmid.items():
        pmid_tag = f"PMID:{pmid}"
        # Упоминание гена в статье
        for gene in genes:
            triplets.add((gene, "mentioned in", pmid_tag))

        # Ко-оккуренция генов внутри одной статьи
        if len(genes) >= 2:
            # Ограничиваем количество комбинаций, чтобы не плодить слишком много
            top_genes = genes[:max_cooccur_genes]
            for i in range(len(top_genes)):
                for j in range(i + 1, len(top_genes)):
                    triplets.add((top_genes[i], "co-occurs with", top_genes[j]))

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
                # Мы уже набрали нужное число PMID, дальше не идём
                break
            seen_pmids.add(pmid)
        if pmid in seen_pmids:  # добавляем все записи для отобранных PMID
            sample.append(rec)
    return sample