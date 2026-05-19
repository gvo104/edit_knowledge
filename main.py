"""
Главный скрипт проекта edit_knowledge.
Последовательно запускает:
1. Обработку данных (извлечение триплетов из gene2pubtatorcentral)
2. Knowledge probing модели на этих триплетах
"""
import sys
import subprocess
import argparse
from pathlib import Path

# Определяем корень папки edit_knowledge (где лежит этот main.py)
EDIT_KNOWLEDGE_DIR = Path(__file__).resolve().parent

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
        cwd=EDIT_KNOWLEDGE_DIR,  # чтобы относительные импорты внутри работали
        capture_output=False      # вывод сразу в консоль
    )
    if result.returncode != 0:
        print(f"ОШИБКА: скрипт {script_path.name} завершился с кодом {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"{description} успешно завершён.\n")

def main():
    parser = argparse.ArgumentParser(description="Edit Knowledge Pipeline")
    parser.add_argument('--dataset', type=str, default='gene',
                        help='Датасет: gene, disease, mutation')
    parser.add_argument('--skip_processing', action='store_true',
                        help='Пропустить этап обработки данных (если уже выполнена)')
    parser.add_argument('--version', type=str, default=None,
                        help='Версия триплетов для probing (если не указана, последняя)')
    args = parser.parse_args()

    print("="*60)
    print("PROJECT: Edit Knowledge Pipeline")
    print("="*60)

    # 1. Обработка данных
    if not args.skip_processing:
        processing_script = EDIT_KNOWLEDGE_DIR / "data_processing" / "run_processing.py"
        if processing_script.exists():
            run_script(processing_script,
                       f"1. Обработка данных (PubTator → триплеты, датасет {args.dataset})",
                       extra_args=['--dataset', args.dataset])
        else:
            print(f"Файл не найден: {processing_script}")
            sys.exit(1)
    else:
        print(">>> Этап обработки данных пропущен (--skip_processing)")

    # 2. Probing модели
    probing_script = EDIT_KNOWLEDGE_DIR / "model_probing" / "run_probing.py"
    if not probing_script.exists():
        print(f"Файл не найден: {probing_script}")
        sys.exit(1)

    probing_args = ['--dataset', args.dataset]
    if args.version:
        probing_args += ['--version', args.version]

    run_script(probing_script,
               f"2. Knowledge Probing модели Qwen2.5-0.5B (датасет {args.dataset})",
               extra_args=probing_args)

    print("\n" + "="*60)
    print("ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print("="*60)

if __name__ == "__main__":
    main()