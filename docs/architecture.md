# Архитектура проекта

## Принцип
Проект работает через единый модульный пайплайн в `src/` без legacy-entrypoint скриптов.

## Слои
- `src/core/`
  - `registry.py` — реестры датасетов, моделей, методов, метрик
  - `run_context.py` — контекст запуска и manifest
  - `artifacts.py` — структура артефактов по говорящему пути
  - `bootstrap.py` — регистрация всех компонентов
- `src/data/`
  - `base.py` — интерфейс датасета
  - `pubtator_dataset.py` — адаптер PubTator
- `src/editing/`
  - `base.py` — интерфейс метода редактирования
  - `lora_sft.py` — реализация LocFT-BF (LoRA SFT)
  - `locft_bf_aug.py` — LocFT-BF + augmentation (адаптер)
  - `rome.py` — заглушка ROME
  - `memit.py`, `mend.py`, `wise.py` — заглушки
- `src/probing/metrics/`
  - `schema.py` — единый формат метрик
  - `calculator.py` — расчёт метрик из сырого probing
  - `aggregator.py` — baseline vs edited сравнение
- `src/experiments/`
  - `pipeline.py` — независимые стадии
  - `experiment_manager.py` — полный прогон из одного конфига
- `src/cli/main.py`
  - единый entrypoint полного запуска

## Стадии
- `prepare_data`
- `baseline_probe`
- `train_edit_method`
- `post_edit_probe`
- `aggregate_results`

## Структура результатов
Для каждой стадии:

`outputs/runs/<experiment>/<dataset>/<model>/<method>/<stage>/<run_id>/`

Внутри:
- `config.yaml`
- `manifest.json`
- `logs/`
- `artifacts/`
- `metrics/`
- `checkpoints/`
- `reports/`

Итоговый отчёт полного прогона:

`outputs/reports/<experiment>_YYYYMMDD_HHMMSS_run_report.json`

## Нормализованные метрики
Единый формат хранится в `metrics.json`:
- `categories.direct`
- `categories.inverse`
- `categories.paraphrase`
- `categories.locality`

Для каждой категории:
- `total`
- `perfect_1_0`
- `accuracy_1_0`
- `good_enough_count`
- `good_enough_rate`
- `mean_score`
