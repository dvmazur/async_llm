# Перенос проекта и окружение double_p

Основная ветка робототехнического проекта: rover в dvmazur/async_llm.
Личная копия Doub1e05/async_llm сохраняет историю первоначального переноса.
Исходный AsyncLLM сохраняется без переименования Python-пакета minisgl.

## Новый сервер

Репозиторий: git@github.com:dvmazur/async_llm.git
Целевой каталог: /home/double/data_scince_expedition/lets/async_llm
Окружение: /home/double/miniconda3/envs/double_p
GPU: две RTX 3090 по 24 ГБ, архитектура sm_86.

```bash
git clone --branch rover git@github.com:dvmazur/async_llm.git async_llm
cd async_llm
conda env create -f robot_sim/environment_double_p.yml
conda activate double_p
python -m pip install uv
uv pip install -c robot_sim/constraints.txt -e ".[dev]"
uv pip install gymnasium vizdoom websockify
```

В pyproject.toml уже заданы CUDA 13.0 индексы PyTorch и sglang-kernel.
Для компиляции выбрать установленный CUDA Toolkit 13.0:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
```

Исторический ros-linux-64.lock получен на Intel Cascadelake с AVX-512.
Не устанавливайте его явно на Ryzen: используйте environment_double_p.yml,
чтобы Conda выбрала совместимые с текущим CPU сборки.

## Настройки конкретной машины

Создать игнорируемый robot_sim/.runtime/config.sh:

```bash
ROBOT_CONDA_EXE=/home/double/miniconda3/bin/conda
ROBOT_CONDA_ENV=double_p
ROBOT_ROS_ENV=double_p
ROBOT_LLM_PYTHON=/home/double/miniconda3/envs/double_p/bin/python
ROBOT_HF_HOME=/home/double/.cache/huggingface
```

Не помещать в этот файл токены и пароли. Для моделей используется стандартный
Hugging Face cache; веса в Git не входят.

```bash
python robot_sim/scripts/download_models.py
bash robot_sim/scripts/setup_headless.sh
bash robot_sim/scripts/prepare_assets.sh
source robot_sim/scripts/robot_env.sh
python -m unittest discover -s robot_sim/tests -v
```

setup_headless.sh извлекает Xvfb и noVNC в .runtime, без sudo.
Текстуры сцен используют model://scene_materials/textures; URDF meshes разрешаются
при запуске. Исходники и параметры не зависят от имени пользователя старого сервера.

## Рабочие команды

```bash
# Физика мини-ровера:
bash robot_sim/scripts/run_rover.sh

# SLAM + IMU/одометрия + динамическая карта:
bash robot_sim/scripts/run_mapping.sh --duration 150 --record-bag

# Qwen reference backend, одна RTX 3090:
ROBOT_LLM_GPU=0 bash robot_sim/scripts/run_llm.sh \
  --backend transformers --gpu-memory-gib 21 --vision

# Явное распределение весов на обе карты:
ROBOT_LLM_GPU=0,1 ROBOT_LLM_MAX_GPUS=2 bash robot_sim/scripts/run_llm.sh \
  --backend transformers --gpu-memory-gib 18 --vision
```

Симулятор использует один логический CPU и CPU-рендеринг, 0 GPU.
LLM по умолчанию видит одну GPU; две требуют явного разрешения.
Двухкарточный reference backend использует консервативную передачу активаций через RAM,
сохранённую после диагностики старого сервера. AsyncLLM backend остаётся отдельным
режимом и не подменяется Transformers автоматически.

## Что хранится в Git

Исходники управления и картографии, готовые модели/текстуры, исходный rover_urdf,
MuJoCo-описание, конфигурации, задачи и инструкции.
Не входят: .runtime, Conda, CUDA caches, веса моделей, ROS bags, видео и logs из results.
Предыдущие экспериментальные записи остаются на старом сервере; их не нужно
скачивать для нового запуска. Установленные версии и проверки новой машины
фиксируются отдельно в results/server_setup.
