# Быстрый запуск: Speleo 15×10

В этом архиве — текущий pipeline с корутинами ролей и BlockHandle, а также
git-checkpoint движка `cuda_graphs_minimal/07-cuda-graphs`:
`bbe7abf55512342d56af61d4009dff0174d8896d` (файл `engine.bundle`).
GitHub-доступ и SSH-ключи для движка не нужны. Веса и venv в ZIP не включены.

Проверено на RTX PRO 6000 Blackwell Server Edition 96 GB: FP8-веса
Qwen3.6-35B-A3B, **FP32 GDN**, 15 параллельных pipeline по 10 действий.
Без профиля и картинок на диске. В одном прогоне: **191,6 tok/s**, 97,0 с,
150/150 действий. Это ориентир, не гарантия на другой машине.

Нужны Linux (Ubuntu/Debian для команды установки системных пакетов), интернет,
Python 3.11+ для setup, рабочий NVIDIA-драйвер и совместимый CUDA toolkit с `nvcc`.
Наш проверенный стек использует CUDA 13.0. Setup **не устанавливает драйвер/CUDA**.
Для этой конфигурации нужна свободная GPU примерно на 96 GB: в проверенном запуске
занималось около 90 GiB. Начинать установку следует со свободной GPU: в конце setup
проверяет CUDA и импорты (модель не загружает).

## 1. Распаковать

```bash
unzip speleo-15x10-cuda-graphs-07.zip
cd speleo-runner
nvidia-smi
nvcc --version
```

Все дальнейшие команды — из этой папки.

## 2. Один раз подготовить окружение и скачать модель

```bash
python3 -m environment.setup \
  --engine-bundle ./engine.bundle \
  --engine ./engine \
  --venv ./.venvs/minisgl \
  --craftium ./craftium \
  --download-model ./models/Qwen3.6-35B-A3B-FP8 \
  --jobs 8 \
  --system-deps
```

Команда создаёт venv, устанавливает зависимости по lock-файлу движка, собирает
Craftium закреплённой версии и скачивает закреплённые веса через `hf`.
Загрузка весов идёт параллельно сборке Craftium. `uv` устанавливается автоматически,
если его нет. `--system-deps` использует apt/sudo; если системные зависимости уже
установлены, этот флаг можно убрать.

Каталоги `engine` и `.venvs/minisgl` должны быть новыми. Не запускайте setup перед
каждым экспериментом. При наличии весов уберите `--download-model` и укажите их
путь в переменной `MODEL` файла эксперимента.

Проверка git-checkpoint:

```bash
git -C engine rev-parse HEAD
```

Ожидается `bbe7abf55512342d56af61d4009dff0174d8896d`.
Самостоятельно развернуть checkpoint без setup можно командой
`git clone engine.bundle engine`; затем setup запускается без `--engine-bundle`.

## 3. Запустить

```bash
python3 experiments/speleo_15x10.py
```

Активировать venv не нужно: Runner сам запускает GPU-worker через указанный
`VENV/bin/python`. Один GPU, одна загруженная модель, 15 pipeline × 10 действий.
Первый старт может несколько минут компилировать FlashInfer и захватывать графы.
Это отдельно от rollout TPS; первое действие и завершение фоновых ролей в TPS входят.

## 4. Забрать результаты

Результаты лежат в `results/speleo-15x10/`. Итоговый Summary печатается в терминале;
численные метрики также лежат в `analysis/summary.json`. Он содержит TPS, tokens/action,
средние decode/prefill batch и выборочное среднее GPU util.

Упаковать все логи с контрольными суммами:

```bash
python3 -m experiment_runner.artifacts results/speleo-15x10 results/speleo-15x10.zip
```

## Изменить запуск

Всё задаётся прямо в `experiments/speleo_15x10.py`: `VENV`, `MODEL`, `RESULTS`,
`GPUS`, `PIPELINES_PER_GPU`, `REPEATS`, `ACTIONS`, seeds и `ENGINE_PARAMS`.
Например, `GPUS = [0, 1]` запускает независимую модель и 15 pipeline **на каждой**
карте (не tensor parallelism). Для повторного запуска задайте новый `RESULTS`:
Runner не перезаписывает существующие результаты. Не меняйте файл во время рана.

Расширенное описание API — в README.md. Это FP32-GDN checkpoint; для BF16 GDN
нужна другая ветка движка, не просто изменение `dtype` модели.
