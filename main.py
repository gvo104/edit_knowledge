"""
Главный скрипт проекта edit_knowledge.
Режимы:
  1) Обычный – аргументы командной строки.
  2) --debug – работа по матрице экспериментов + сбор общего отчёта.
"""
import sys
import subprocess
import argparse
import glob
import json
from pathlib import Path

# Определяем корень папки edit_knowledge (где лежит этот main.py)
EDIT_KNOWLEDGE_DIR = Path(__file__).resolve().parent

# Импортируем config для получения списка датасетов
sys.path.insert(0, str(EDIT_KNOWLEDGE_DIR))
import config


# ══════════════════════════════════════════════════════════════
# Матрица экспериментов (единый словарь)
# Ключ   – имя датасета (gene, disease, mutation)
# Значение – словарь с настройками этапов:
#   processing        – True/False, запускать ли обработку PubTator → триплеты
#   choose_triplets   – "latest" (последний по времени файл),
#                       конкретный timestamp (строка, напр. "20250120_143022"),
#                       или None (если processing=True, возьмётся свежая версия)
#   probing           – True/False, запускать ли knowledge probing
#   train_lora        – True/False, запускать ли обучение LoRA
#   force_processing  – перезаписать файл триплетов (--force) [по умолчанию False]
#   force_probing     – перезаписать результаты probing (--force) [по умолчанию False]
#   force_retrain     – переобучить LoRA, даже если адаптер существует [по умолчанию False]
#   probe_after_train – если False, не запускать probing сразу после обучения [по умолчанию True]
#
# Все пути и версии формируются автоматически в соответствии с единой логикой.
# ══════════════════════════════════════════════════════════════
DEBUG_MATRIX = {
    "gene": {
        "processing": True,
        "choose_triplets": None,        # после процессинга берётся свежая версия
        "probing": True,
        "train_lora": True,
        "force_processing": True,       # пересоздать триплеты
        "force_probing": True,          # перезаписать baseline probing
        "force_retrain": True,          # переобучить LoRA
        "probe_after_train": True,
    },
    "disease": {
        "processing": True,
        "choose_triplets": None,
        "probing": True,
        "train_lora": True,
        "force_processing": True,
        "force_probing": True,
        "force_retrain": True,
        "probe_after_train": True,
    },
    "mutation": {
        "processing": True,
        "choose_triplets": None,
        "probing": True,
        "train_lora": True,
        "force_processing": True,
        "force_probing": True,
        "force_retrain": True,
        "probe_after_train": True,
    },
}


def run_script(script_path: Path, description: str, extra_args=None):
    """Запуск Python-скрипта и проверка кода возврата."""
    if extra_args is None:
        extra_args = []
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n{'='*60}")
    print(f"Запуск: {description}")
    print(f"Команда: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=EDIT_KNOWLEDGE_DIR, capture_output=False)
    if result.returncode != 0:
        print(f"ОШИБКА: скрипт {script_path.name} завершился с кодом {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"{description} успешно завершён.\n")


def _get_latest_version(dataset: str) -> str:
    """Возвращает timestamp последнего файла триплетов для датасета."""
    pattern = str(Path(config.OUTPUT_DIR) / f"{dataset}_*_triplets.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(f"Нет файлов триплетов для датасета {dataset}")
    # из имени файла извлекаем timestamp (между названием датасета и _triplets.json)
    name = Path(files[0]).stem            # gene_20250120_143022_triplets
    # удаляем префикс и суффикс
    timestamp = name[len(dataset)+1:].replace("_triplets", "")
    return timestamp


def collect_summary(datasets: list):
    """
    Собирает последние baseline- и LoRA-метрики по датасетам и сохраняет в data/summary.json.
    """
    summary_all = {}
    for dataset in datasets:
        # baseline
        baseline_dir = Path(config.PROBING_DIR)
        pattern_base = str(baseline_dir / f"*_{dataset}_*_probing_results.json")
        base_files = sorted(glob.glob(pattern_base), reverse=True)
        baseline_summary = None
        if base_files:
            with open(base_files[0], 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
            baseline_summary = baseline_data.get("summary")
        
        # lora
        lora_dir = Path(config.PROBING_DIR_LORA)
        pattern_lora = str(lora_dir / f"*_{dataset}_*_probing_results.json")
        lora_files = sorted(glob.glob(pattern_lora), reverse=True)
        lora_summary = None
        if lora_files:
            with open(lora_files[0], 'r', encoding='utf-8') as f:
                lora_data = json.load(f)
            lora_summary = lora_data.get("summary")
        
        summary_all[dataset] = {
            "baseline": baseline_summary,
            "lora": lora_summary
        }
    
    output_path = Path(config.BASE_DIR) / "data" / "summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_all, f, indent=2, ensure_ascii=False)
    print(f"Общий отчёт сохранён: {output_path}")


def run_debug_matrix():
    processing_script = EDIT_KNOWLEDGE_DIR / "data_processing" / "run_processing.py"
    probing_script = EDIT_KNOWLEDGE_DIR / "model_probing" / "run_probing.py"
    training_script = EDIT_KNOWLEDGE_DIR / "model_training" / "run_sft_lora.py"

    for dataset, params in DEBUG_MATRIX.items():
        print(f"\n{'#'*60}")
        print(f"# Эксперимент: датасет {dataset}")
        print(f"{'#'*60}")

        used_version = None   # версия триплетов, которую будем передавать явно

        # ------------------------------------------------------------
        # Этап 1: processing (если True)
        # ------------------------------------------------------------
        if params.get("processing"):
            proc_args = ['--dataset', dataset]
            if params.get("force_processing"):
                proc_args.append('--force')
            run_script(processing_script, f"1. Обработка данных (датасет {dataset})", proc_args)
            # После успешной обработки определяем только что созданную версию
            used_version = _get_latest_version(dataset)
        else:
            # Процессинг пропущен – нужна явная версия
            chosen = params.get("choose_triplets")
            if chosen is None or chosen == "latest":
                used_version = _get_latest_version(dataset)
            else:
                used_version = chosen

        # ------------------------------------------------------------
        # Этап 2: probing (если True)
        # ------------------------------------------------------------
        if params.get("probing"):
            probe_args = ['--dataset', dataset]
            if used_version:
                probe_args += ['--version', used_version]
            if params.get("force_probing"):
                probe_args.append('--force')
            run_script(probing_script, f"2. Baseline probing (датасет {dataset})", probe_args)

        # ------------------------------------------------------------
        # Этап 3: train_lora (если True)
        # ------------------------------------------------------------
        if params.get("train_lora"):
            train_args = ['--dataset', dataset]
            # Версия триплетов для обучения (если не было processing, нужно указать)
            if used_version:
                train_args += ['--version', used_version]
            if not params.get("probe_after_train", True):
                train_args.append('--no_probe')
            if params.get("force_retrain"):
                train_args.append('--force_retrain')
            run_script(training_script, f"3. LoRA обучение (датасет {dataset})", train_args)

    # После всех экспериментов собираем общий отчёт
    collect_summary(list(DEBUG_MATRIX.keys()))


def main():
    parser = argparse.ArgumentParser(description="Edit Knowledge Pipeline")
    parser.add_argument('--debug', action='store_true',
                        help='Запустить в режиме отладки согласно матрице DEBUG_MATRIX')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Датасет: gene, disease, mutation')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Запустить обработку и/или probing для всех датасетов')
    parser.add_argument('--skip_processing', action='store_true')
    parser.add_argument('--skip_probing', action='store_true')
    parser.add_argument('--version', type=str, default=None)
    args = parser.parse_args()

    if args.debug:
        print("=" * 60)
        print("РЕЖИМ ОТЛАДКИ (DEBUG MATRIX) – ПОЛНЫЙ ЦИКЛ С ПЕРЕЗАПИСЬЮ")
        print("=" * 60)
        run_debug_matrix()
        print("\n" + "=" * 60)
        print("ВСЕ ЭКСПЕРИМЕНТЫ ПО МАТРИЦЕ ЗАВЕРШЕНЫ")
        print("=" * 60)
        return

    # Обычный режим (без изменений)
    if args.all_datasets:
        datasets = list(config.DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [config.DATASET_NAME]

    print("="*60)
    print("PROJECT: Edit Knowledge Pipeline")
    print(f"Датасеты: {', '.join(datasets)}")
    if args.skip_processing:
        print(">>> Этап обработки данных ПРОПУЩЕН")
    if args.skip_probing:
        print(">>> Этап probing ПРОПУЩЕН")
    print("="*60)

    processing_script = EDIT_KNOWLEDGE_DIR / "data_processing" / "run_processing.py"
    probing_script = EDIT_KNOWLEDGE_DIR / "model_probing" / "run_probing.py"

    # Проверяем существование скриптов заранее
    if not args.skip_processing and not processing_script.exists():
        print(f"ОШИБКА: файл не найден {processing_script}")
        sys.exit(1)
    if not args.skip_probing and not probing_script.exists():
        print(f"ОШИБКА: файл не найден {probing_script}")
        sys.exit(1)

    # Последовательная обработка всех датасетов
    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# Датасет: {dataset_name}")
        print(f"{'#'*60}")

        # 1. Обработка данных
        if not args.skip_processing:
            run_script(processing_script,
                       f"1. Обработка данных (PubTator → триплеты, датасет {dataset_name})",
                       extra_args=['--dataset', dataset_name])

        # 2. Probing модели
        if not args.skip_probing:
            probing_args = ['--dataset', dataset_name]
            if args.version:
                probing_args += ['--version', args.version]
            run_script(probing_script,
                       f"2. Knowledge Probing модели Qwen2.5-0.5B (датасет {dataset_name})",
                       extra_args=probing_args)

    print("\n" + "="*60)
    print("ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print("="*60)


if __name__ == "__main__":
    main()