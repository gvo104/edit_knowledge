# Changelog

## 2026-05-23

### Финализация рефакторинга
- Введён единый full-pipeline entrypoint: `src/cli/main.py`.
- Удалены legacy entrypoint-скрипты и legacy-артефакты результатов.
- Оркестрация переведена на прямой вызов модулей без subprocess к старым CLI.
- Введена новая структура артефактов:
  - `outputs/runs/<experiment>/<dataset>/<model>/<method>/<stage>/<run_id>/`
- Добавлен слой нормализованных метрик:
  - `src/probing/metrics/schema.py`
  - `src/probing/metrics/calculator.py`
  - `src/probing/metrics/aggregator.py`
- Добавлен единый финальный отчёт:
  - `outputs/reports/<experiment>_YYYYMMDD_HHMMSS_run_report.json`
- Реализован единый интерфейс методов редактирования и подключены:
  - `locft_bf`
  - `locft_bf_aug`
  - `rome` (заглушка)
- Подготовлены заглушки для:
  - `memit`, `mend`, `wise`
