"""
Главный скрипт проекта edit_knowledge.
Последовательно запускает:
1. Обработку данных (извлечение триплетов из gene2pubtatorcentral)
2. Knowledge probing модели на этих триплетах
"""
import sys
import subprocess
from pathlib import Path

# Определяем корень папки edit_knowledge (где лежит этот main.py)
EDIT_KNOWLEDGE_DIR = Path(__file__).resolve().parent

def run_script(script_path: Path, description: str):
    """Запуск Python-скрипта и проверка кода возврата."""
    print(f"\n{'='*60}")
    print(f"Запуск: {description}")
    print(f"Скрипт: {script_path}")
    print('='*60)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=EDIT_KNOWLEDGE_DIR,  # чтобы относительные импорты внутри работали
        capture_output=False      # вывод сразу в консоль
    )
    if result.returncode != 0:
        print(f"ОШИБКА: скрипт {script_path.name} завершился с кодом {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"{description} успешно завершён.\n")

def main():
    print("="*60)
    print("PROJECT: Edit Knowledge Pipeline")
    print("="*60)

    # 1. Обработка данных
    processing_script = EDIT_KNOWLEDGE_DIR / "data_processing" / "run_processing.py"
    if processing_script.exists():
        run_script(processing_script, "1. Обработка данных (PubTator → триплеты)")
    else:
        print(f"Файл не найден: {processing_script}")
        sys.exit(1)

    # 2. Probing модели
    probing_script = EDIT_KNOWLEDGE_DIR / "model_probing" / "run_probing.py"
    if probing_script.exists():
        run_script(probing_script, "2. Knowledge Probing модели Qwen2.5-0.5B")
    else:
        print(f"Файл не найден: {probing_script}")
        sys.exit(1)

    print("\n" + "="*60)
    print("ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ")
    print("="*60)

if __name__ == "__main__":
    main()