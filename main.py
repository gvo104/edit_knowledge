"""
Главный скрипт проекта edit_knowledge.
Последовательно запускает:
1. Обработку данных (извлечение триплетов из PubTator)
2. Knowledge probing модели на этих триплетах

Поддерживает запуск одного датасета или всех доступных (gene, disease, mutation).
"""
import sys
import subprocess
import argparse
from pathlib import Path

# Определяем корень папки edit_knowledge (где лежит этот main.py)
EDIT_KNOWLEDGE_DIR = Path(__file__).resolve().parent

# Импортируем config для получения списка датасетов
sys.path.insert(0, str(EDIT_KNOWLEDGE_DIR))
import config


def run_script(script_path: Path, description: str, extra_args=None):
    """Запуск Python-скрипта и проверка кода возврата."""
    if extra_args is None:
        extra_args = []
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n{'='*60}")
    print(f"Запуск: {description}")
    print(f"Команда: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(
        cmd,
        cwd=EDIT_KNOWLEDGE_DIR,   # чтобы относительные импорты внутри работали
        capture_output=False       # вывод сразу в консоль
    )
    if result.returncode != 0:
        print(f"ОШИБКА: скрипт {script_path.name} завершился с кодом {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"{description} успешно завершён.\n")


def main():
    parser = argparse.ArgumentParser(description="Edit Knowledge Pipeline")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Датасет: gene, disease, mutation. Если не указан и не задан --all_datasets, используется значение из config.DATASET_NAME')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Запустить обработку и/или probing для всех доступных датасетов (gene, disease, mutation)')
    parser.add_argument('--skip_processing', action='store_true',
                        help='Пропустить этап обработки данных')
    parser.add_argument('--skip_probing', action='store_true',
                        help='Пропустить этап probing')
    parser.add_argument('--version', type=str, default=None,
                        help='Версия триплетов для probing (если не указана, используется последняя)')
    args = parser.parse_args()

    # Определяем список датасетов
    if args.all_datasets:
        datasets = list(config.DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [config.DATASET_NAME]   # значение по умолчанию из config

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