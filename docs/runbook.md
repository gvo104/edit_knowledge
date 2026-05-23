# Runbook

## Полный запуск с нуля

```bash
python src/cli/main.py --config configs/experiment/full_pipeline.yaml
```

Это единственный основной способ запуска.

## Что произойдёт
Для каждого датасета из конфига:
1. `prepare_data`
2. `baseline_probe`
3. `train_edit_method` для каждого метода
4. `post_edit_probe` для каждого метода
5. Общий `aggregate_results`

## Где смотреть результаты
- стадии:
  - `outputs/runs/<experiment>/<dataset>/<model>/<method>/<stage>/<run_id>/`
- финальный отчёт:
  - `outputs/reports/<experiment>_YYYYMMDD_HHMMSS_run_report.json`

## Содержимое финального отчёта
- `metadata`
- `stages`
- `comparison.rows` (табличный формат)
- `comparison.summary_by_method`
- `comparison.summary_by_dataset`
- `comparison.best_method_per_dataset`

## Примечание
Старые способы запуска и старые форматы сохранения удалены.
