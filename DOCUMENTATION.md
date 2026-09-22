# Portable Speleo runner — single-world long runs

**Короткая инструкция установки и запуска: [README.md](README.md).**

## Single-world long runs

`experiments/speleo_1x500_r105_falsifier_async.py` runs one world at a time,
500 actions per episode, 105 repeats (seeds0–104). Model and graph buffers load
once per GPU. Repeats create fresh World/history/role RNGs and close their private
blocks; only the immutable shared system prefix stays cached in the engine.
Four async roles remain active inside that one pipeline. Their tested budgets
and temperatures are18/.35,60/.65,16/.45,18/.45 for observer/planner/executor/falsifier.
The original action readout remains argmax; role text is sampled. Planner starts
at most once per ten actions and owns an independent snapshot of text history.
History contains compact event records, not old images or every role answer.
`history_size` now logs the exact token count after every append, including action500.

Memory sizing for Qwen3.6-35B-A3B-FP8 with FP32 GDN on a free96-GB RTX PRO6000:

- KV is BF16:10 attention layers×2(K/V)×2 heads×256 dimensions×2 bytes =20 KiB/token.
- Completed5×500 histories at action490 were39382–42013 tokens each, about40–43k
  at500. One growing history plus planner copy is about1.6–1.7 GiB KV per pipeline.
- The new16384-page×16-token pool is262144 slots, exactly5 GiB, over three times
  that observed long-history requirement; remaining capacity covers temporary blocks.
  It is **not** a256k per-request context. Per-request limit is65536.
- Eight request slots allow concurrent roles and preparations even though world
  concurrency is1. Decode profiles[1,2,4,8], prefill[256,1024,4096], depth16.
  `max_prefill_rows=4096` is new query rows per batch, not context length.
-105 repeats do not multiply simultaneous KV/state memory. Do not retain previous
  pipelines, blocks or logits in an external callback.

The reference5×500 RTX run completed with a10-GiB pool,72.1 GiB allocated peak
and76.3 GiB reserved. The single-world configuration reduces pool and active-state
capacity; no fresh GPU measurement of this exact105-repeat configuration has been
performed. CPU tests cover500 actions and105 successive short episodes with
private-block cleanup. These are lifecycle/config tests, not model-quality tests
or a mathematical worst-case memory guarantee for other checkpoints/backends.

`SettledWorld` wraps reset: at least10 no-ops spaced0.1 seconds apart, require five
consecutive canonical positions(24.3,5.5,-36.3), velocity norm<0.001, and a nonblank
image. At120 no-ops or120 seconds it fails explicitly; it never starts from a known
unready spawn. No-ops are outside policy action counts; native max_steps includes
an extra120. Natural terminal events still end an episode early and are logged.
Position, velocity, pitch/yaw and `mt_dtime` are recorded, not passed to the model.
`mt_dtime` is the last native frame only, not the summed time of eight frames.
One world plus readiness checks does not prove fixed or identical simulation time.

One model per GPU, independently repeating async pipeline slots. The engine owns
batching. Pipeline code owns roles, history, its World and Recorder. No TP, image
replay, profiler, hidden warmup episodes, or runtime package installation.

The main entry point is **an ordinary Python file in `experiments/`**, not a
parameter-heavy launcher CLI. Runner/pipeline sources are read from this checkout;
they do not need to be installed in either the launching Python or the engine venv.

```text
experiments/speleo_15x5_fast_falsifer.py  # edit venv, model/config, GPUs, repeats, output path here
experiment_runner/        # execution and telemetry infrastructure
environment/              # explicit setup, dependency and Qwen/Craftium checks
pipelines/speleo.py        # role coroutines, token loop and all their communication
pipelines/prompts.py       # prompt text; no scheduling
pipelines/world.py         # async environment transport
tools/                    # optional checks, plotting and release packaging
pyproject.toml            # dependency groups and native build constraints
```

## Setup (explicit; never run by Runner)

Setup installs pinned `uv` locally if absent, without editing shell startup files.
Select an existing engine repository with `pyproject.toml` and `uv.lock`.
Choose its revision beforehand; setup never clones/pulls/checks out the engine:

```bash
python -m environment.setup \
  --engine /path/to/engine --venv /path/to/venv \
  --craftium /path/to/craftium --jobs 8 --system-deps \
  --download-model /path/to/Qwen3.6-35B-A3B-FP8
```

Run these commands from the extracted source directory. `--system-deps` explicitly
authorizes apt packages and sudo if needed; it **never installs GPU drivers**.
The machine needs a working CUDA toolkit compatible with its GPU/driver.
Model download uses `hf`, overlaps the native build, and pins the checkpoint
revision. Omit `--download-model` to use existing weights. Omit `--system-deps`
when system dependencies are already installed. Existing venv modification requires
`--update-existing`; existing dirty Craftium sources are not overwritten.

The selected engine's lock pins its Torch, FlashInfer, FLA and SGLang kernels.
This checkout declares runtime/test/plot dependencies and build constraints in
`pyproject.toml`; there are no separate requirements/constraints text files.
Setup first syncs the selected engine's lock, then resolves the runtime group and
Craftium together, constrained by that lock. The exported constraints are passed
through stdin, not maintained in another configuration file. A conflict fails
instead of silently upgrading the engine. Setup does **not** install this runner.
Preflight checks actual imports, not just metadata. RTX should not fall back due
to a missing CUDA library. On GB10 SM121 the engine's intentional MoE fallback is
reported explicitly. The NVIDIA cuSPARSELt 0.8.1 ARM wheel has a known `sbsa`
internal tag despite its `aarch64` filename; setup accepts only that exact tag
defect after validating the library's AArch64 ELF header, with a warning. All
other dependency-check failures remain fatal.

These checks run during explicit Setup, not before every experiment. To rerun them
manually from the checkout with the prepared venv: `python -m environment.check`
and `python -m environment.dependencies /path/to/uv`. They are specific to the
Qwen/Craftium environment, not a generic Runner contract.

Workers prepend the selected venv's `bin` directory to PATH: JIT build tools such
as ninja must come from that environment too. Preflight checks ninja before loading
weights. Craftium uses setuptools' pinned compatible editable mode so a neighbouring
checkout directory cannot shadow the actual package via a namespace import.

Setup installs Craftium revision `8eb8707cb756df47e76131a0058ab724d2383c76`.
The default model is `Qwen/Qwen3.6-35B-A3B-FP8`, revision
`61a5771f218894aaacf97551e24a25b866750fc2`. Old results with a different weight
revision must not be treated as exactly matched controls.

## Run

Open `experiments/speleo_15x5_fast_falsifer.py` and edit its settings:

```python
VENV = Path('/path/to/engine-venv')
MODEL = Path('/path/to/Qwen3.6-35B-A3B-FP8')
RESULTS = HERE / 'results' / 'experiment-001_fast_falsifer'
GPUS = [0, 1]
PIPELINES_PER_GPU = 15
REPEATS = 3
ACTIONS = 5
# ENGINE_PARAMS is an ordinary Python dict in the same file.
```

Then run the file, from any working directory:

```bash
python /path/to/repo/experiments/speleo_15x5_fast_falsifer.py
```

The launching Python only needs the standard library. The file adds its own source
root; Runner dispatches workers through the chosen `VENV/bin/python` with the same
source tree. It never installs packages on startup. Edit the file again for another
experiment, or copy it to another name inside `experiments/`.

If you prefer neighbouring JSON files, parse them explicitly in that Python file:

```python
import json
ENGINE_PARAMS = json.loads((HERE / 'engine.json').read_text())
```

The file contains the actual `Runner(...).set_engine_params(...).set_pipeline(...)
.set_concurrency(...).set_results_directory(...).run(gpus=...)` call. There is no
separate experiment format or hidden conversion into environment variables.
Keep the call under `if __name__ == '__main__'`. Each GPU worker re-executes this same
file with the same arguments under the selected Python. Its `Runner.run()` enters
worker mode and uses the locally recreated factory directly, then exits the child.
There is no module/function-name resolver or callable pickling. Factory closures and
lambdas are supported. Keep file startup declarative: code before `run()` executes
in every process. Use one `run()` per experiment file; do not edit it during a run.

The example above runs15slots ×3episodes ×2GPUs, each with5actions. Slots start their
next repeat independently; each GPU loads the model once. No SSH orchestration or TP.

`run()` blocks and returns `None`. It prints a numeric Summary and writes all results.
It never overwrites a nonempty result directory. Engine parameters are copied on
assignment. JSON is optional: parse it yourself and pass a dict. Values must be
serializable; use strings for paths and dtypes. Prefer absolute model paths.
Unknown engine/adapter parameters fail, not silently change the run.

`experiments/speleo_15x5_fast_falsifer.py` contains the fast validated layout: KV page
size 16, 8192 pages, max sequence 32768, decode profiles 4/16/48/64, prefill profiles
256/1024/4096, prefill cap 4096, depth 16. Attention workspace is managed by the
engine's existing backend; this runner does not override it. This is **FP8 weights
with FP32 GDN state**, not the experimental BF16 GDN variant. The cap is the engine's
segment-expanded query-row cap, not a number of bytes. Larger concurrency requires
an explicitly sized config and memory check; no automatic resizing/fallback.

## Настройки Python-эксперимента

Эксперимент — обычный Python-файл. Удобно скопировать готовый пример и менять
его, не затрагивая другие запуски:

```bash
cp experiments/speleo_15x10_fast_falsifer.py experiments/my_run_fast_falsifer.py
```

Все настройки ниже редактируются **в `experiments/my_run_fast_falsifer.py`**. Они не импортируются
из другого эксперимента. Файлы `rtx_*.py` — исторические контроли; для нового
эксперимента начните со `speleo_15x10_fast_falsifer.py`, а не с них.

### Пути, GPU и объём эксперимента

В начале файла уже определён `REPOSITORY`: это папка **запускалки**, вычисленная
от расположения файла, а не от текущего рабочего каталога. Например, для пяти
параллельных pipeline, двух эпизодов на каждый и десяти действий на эпизод:

```python
VENV = REPOSITORY / '.venvs' / 'minisgl'
MODEL = REPOSITORY / 'models' / 'Qwen3.6-35B-A3B-FP8'
RESULTS = REPOSITORY / 'results' / 'my-run-5x10-repeat2_fast_falsifer'

GPUS = [0]
PIPELINES_PER_GPU = 5
REPEATS = 2
ACTIONS = 10
MODEL_SEED_START = 0
WORLD_SEED_START = 0
DUMP_IMAGES = False
GIF_ON = False
```

| Переменная | За что отвечает |
|---|---|
| `VENV` | Окружение, созданное setup. Worker запускается через `VENV/bin/python`; оттуда импортируется установленный движок. Чтобы выбрать другую версию движка, укажите окружение, подготовленное с её checkout. |
| `MODEL` | Папка весов. Можно задать абсолютный путь: `Path('/data/models/Qwen3.6-35B-A3B-FP8')`. Он передаётся в `ENGINE_PARAMS['engine_config']['model_path']`. |
| `RESULTS` | Каталог именно этого запуска. Для следующего запуска задайте новое имя: непустой каталог Runner не перезаписывает. |
| `GPUS` | Список GPU для независимых копий эксперимента. `[0, 1]` загружает по модели на каждую карту. Это не tensor parallelism. |
| `PIPELINES_PER_GPU` | Число одновременно работающих pipeline **на каждой GPU**. Они делят один engine; сборкой model batches занимается engine. Это не размер decode-batch: один pipeline содержит несколько генерирующих ролей. |
| `REPEATS` | Сколько эпизодов последовательно выполняет каждый параллельный слот. Завершив свой эпизод, слот сразу начинает следующий, не ожидая остальных. Модель между повторами не перезагружается. |
| `ACTIONS` | Максимальное число действий агента в мире за эпизод, не число генерируемых токенов. Эпизод может закончиться раньше при завершении среды. |
| `MODEL_SEED_START` | Начало диапазона seed для генерации модели. Каждый эпизод получает свой seed, из него pipeline выводит seed ролей. |
| `WORLD_SEED_START` | Начало диапазона seed миров. Seed меняется между эпизодами, в том числе между повторами одного слота. |
| `DUMP_IMAGES` | Сохранять наблюдения в PNG. При `False` модель по-прежнему получает изображения; отключается только запись картинок на диск. |
| `GIF_ON` | Записывать GIF эпизода. Независим от PNG: можно включить GIF, оставив `DUMP_IMAGES=False`. |

Количество эпизодов: `len(GPUS) × PIPELINES_PER_GPU × REPEATS`. Пример выше —
10 эпизодов и до 100 действий суммарно, одновременно работают пять pipeline.
С `GPUS = [0, 1]` получится 20 эпизодов и до 200 действий, по пять pipeline на карту.

### Параметры движка: `ENGINE_PARAMS`

В файле уже есть полный словарь `ENGINE_PARAMS`. Для обычного изменения длины
эпизода или числа повторов его не нужно переписывать. Он разделён на две части:

- `engine_config` — аргументы конфигурации mini-sglang;
- `adapter_options` — настройки нашей обвязки, сейчас `cpu_threads`.

| Поля примера | Значение |
|---|---|
| `model_path`, `dtype`, `quantization` | Путь к весам, dtype вычислений и формат квантизации. В примере — FP8 checkpoint с `dtype='bfloat16'`, `quantization='fp8'`. Это не включает BF16-хранение GDN. |
| `max_running_req=64` | Вместимость движка по model requests, не количество игровых pipeline. |
| `page_size=16`, `num_page_override=8192` | Размер KV-страницы в токенах и явно заданное число страниц KV-пула. |
| `max_seq_len_override=32768` | Настройка максимальной длины последовательности движка. Не лимит действий эпизода. |
| `memory_ratio=.9` | Параметр бюджетирования памяти движка. При явном `num_page_override` число KV-страниц задаётся отдельно; это не общий предохранитель от OOM. |
| `attention_backend='fi'` | Использовать FlashInfer Attention. |
| `max_prefill_rows=4096` | Лимит работы одного prefill с учётом разворачивания сегментов shared-cache. Более длинная работа делится на части. Это не мегабайты и не лимит суммарного контекста. |
| `cuda_graph_bs=[4,16,48,64]`, `cuda_graph_max_bs=64` | Профили вместимости decode CUDA Graphs и максимальный размер. |
| `shared_cuda_graph_prefill_rows=[256,1024,4096]` | Профили вместимости prefill CUDA Graphs. |
| `shared_cuda_graph_max_depth=16` | Вместимость графового пути по глубине цепочки shared-блоков. |
| `generation_config` | Общие параметры генерации движка. Генерация конкретных ролей получает параметры из `ROLE_PARAMETERS` ниже. |
| `adapter_options['cpu_threads']=4` | Число CPU-потоков PyTorch в процессе одной модели. Не число pipeline и не `--jobs` сборки. |

При увеличении `PIPELINES_PER_GPU` нужно учитывать память и вместимости движка/
графов: Runner не увеличивает их автоматически.

### Генерация ролей: `ROLE_PARAMETERS`

Каждая запись задаёт параметры одного типа генерации. Например:

```python
'planner': RoleParams(
    budget=60, temperature=.65, seed_offset=2, top_k=20, top_p=.9,
),
```

`budget` — максимальное число генерируемых токенов за один вызов роли;
`temperature`, `top_k`, `top_p` — параметры sampling;
`seed_offset` — различающий роли компонент seed. В примере каждой роли задан
отдельный offset. Эти значения меняются прямо в словаре эксперимента.

| Ключ | Работа роли | Budget по умолчанию |
|---|---|---:|
| `observer` | Коротко описывает наблюдения | 18 |
| `planner` | Строит план, не чаще раза в 10 действий | 60 |
| `executor` | Формулирует намерение и решающее свидетельство | 16 |

Отдельного refine нет. Executor стартует после одного видимого токена observer.
Action-readout ждёт завершения executor. Planner может продолжать
генерацию между действиями. Интервал считается от старта предыдущего planner,
даже если тот вернул пустой ответ; `REPLAN` не обходит лимит.
Максимум за 100 действий — 4000 токенов ролей плюс 100 action-readout.
Seed роли: `1_000_003 * (context.model_seed + 1) + seed_offset`, как в V9;
offsets observer/planner/executor — 1/2/5, температуры — .35/.65/.45.

### Что делает код внизу файла

`make_pipeline(engine, context)` создаёт **новый эпизод**: его мир `SpeleoWorld`,
его `Recorder` и `SpeleoPipeline`. Engine уже создан и разделяется между pipeline
на одной GPU. Context передаёт seed эпизода и его отдельный каталог результатов.
`ACTIONS` задаёт предел и миру, и pipeline; `ROLE_PARAMETERS` передаётся pipeline.

```python
if __name__ == '__main__':
    (Runner(VENV, model_seed_start=MODEL_SEED_START, world_seed_start=WORLD_SEED_START)
        .set_engine_params(ENGINE_PARAMS)  # конфигурация одной модели
        .set_pipeline(RepeatedPipeline(make_pipeline, repeats=REPEATS))
        .set_concurrency(PIPELINES_PER_GPU)  # параллельные эпизоды на каждой GPU
        .set_results_directory(RESULTS)     # общий корень артефактов запуска
        .run(gpus=GPUS))                    # запустить и дождаться завершения
```

Эту цепочку обычно менять не нужно — достаточно переменных выше.
Сохраните файл и запустите **на GPU-машине**:

```bash
python3 experiments/my_run_fast_falsifer.py
```

Venv активировать не нужно. Запускающий Python использует стандартную библиотеку;
процесс каждой модели запускается через выбранный `VENV/bin/python`.
Для исходного примера без изменений команда — `python3 experiments/speleo_15x10_fast_falsifer.py`.
Не меняйте файл эксперимента во время выполнения: worker перечитывает тот же файл.

Первый запуск может компилировать FlashInfer и захватывать CUDA Graphs.
Инициализация отделена от rollout TPS. Первое действие и завершение фоновых ролей
входят в измеряемый rollout.


## Change a pipeline without changing the engine

Write a callable `make_pipeline(engine, context)`. It creates
a fresh `SpeleoPipeline(world, recorder, engine, context=context, max_actions=...)`.
Engine is shared; never close it inside a Pipeline. Context supplies episode ID,
world/model seeds, slot/repeat IDs and the episode's result directory. A custom
pipeline implements async `run()` with its own `finally` cleanup. It should check
the runner-provided `stop_requested()` between safe actions.

`RepeatedPipeline` stores a factory, not live worlds or CUDA tensors. The experiment
recreates it inside each worker; it is not serialized. Keep launch code in
the `if __name__ == '__main__'` guard. The model-facing adapter lives in
`experiment_runner/engine.py`; it has no role names, temperatures or game rules.

### The coroutine protocol is one file: `pipelines/speleo.py`

There are four persistent role tasks per episode:

- `observer` receives previous/current frames and the last action, prepares the image
  KV block, chooses its input chain and generates a live description.
- `planner` owns the ten-action start interval, freezes history, chooses its live inputs,
  and publishes completed plans at action boundaries. The executor reports observations;
  it does not construct planner prompts or decide when a plan is stale.
- `falsifier` reads the growing executor intention after four tokens and generates
  an objection. Executor awaits its completion before choosing the action.
- `executor` drives the episode, generates one intention without a refine stage,
  selects an action and calls `await world.pass_action(...)`.

`generate_tokens()` is the shared model-IO loop:
prefill its instruction block, sample one token, append through decode to its KV tail,
then notify readers. A `Generation` carries the input block chain, instruction block,
growing output block, text and completion/token notifications. Consumers read
live KV blocks rather than copying partially generated text.
Observer awaits this loop directly. Executor owns an `executor/tokens`
child task and drains that child even on failure. Planner owns one `planner/tokens` child task:
a long generation must not prevent its controller
from acknowledging later observations. It starts no second generation until the
first is finished/published, and drains its child before shutdown. This is local
policy concurrency, not a second scheduler or a new engine abstraction.

All queues, waits, read/write chains, planner publication, REPLAN decisions and
environment steps are visible in that file. Waiting for one observer token
exposes a live KV tail; planner can span environment actions.
The action query reads the executor and falsifier blocks exactly once.

The policy has `ROLE_PARAMS` defaults for direct use. Each supplied experiment
declares its own complete `ROLE_PARAMETERS` (budgets, temperatures, top-k/top-p,
seed offsets) and passes `role_params=...`; it imports neither another experiment's
config nor the policy's default dict. Engine sampling
receives explicit parameters and an RNG, with no role lookup. Token scoring is a
generic engine operation; action names and argmax selection belong to the executor.
`BlockHandle` in `experiment_runner/blocks.py` owns a native engine block.
Each consumer acquires its own handle with `share()` and releases it with
`await handle.aclose()` (or `async with`). Closing the last handle calls the engine's
normal `free_block`; there is no separate pool, allocator or block registry.
Closing a handle twice is harmless; reading/sharing a closed handle is an error.
Snapshot uses the native merge operation. Cached system prefixes have one engine
owner plus an independent handle per episode. All handles must be closed before
engine shutdown; Python assignment is only an alias, not an ownership acquisition.
We do not call asynchronous GPU cleanup from a Python destructor.
The old orchestration classes `Client`/`Stream` and `pipelines/core.py` are removed.

World returns an Observation from `await reset()` / `await pass_action(action)`.
The unmodified Craftium bridge can hold the GIL, so World uses a spawned helper
process and async IPC. Images are owned snapshots. Native rendering uses software
Mesa with two rendering threads and an Xvfb display when no DISPLAY is provided.
Position is logged for evaluation, **not added to model prompts**.

Prompts and policy rules come from `shared_control_t60_v9`, whose preserved BF16
7×100 repeat reached mean final height -13.6276 at 54.97 tokens/action. The original
Pool/Client were not copied: this port reuses the existing BlockHandle/Runner.
World, engine adapter and telemetry are unchanged. Async transport and engine
batching can change interleaving; that historical score is not a new quality result.

## Telemetry and optional media

```python
Recorder(context.results_directory, dump_images=False, gif_on=False)
```

Both flags default to false. No PNG/GIF/base64 images are stored in that mode;
model image inputs are unchanged. GIF-only mode streams frames without permanent
PNG files or retaining the entire episode in RAM. Image queues are bounded. Writer
errors are propagated. The agent's initial height and height after every action,
role text, rewards, actions and event times remain in numeric/text logs.

Summary runs automatically after workers/log writers finish. It is also reusable:

```bash
python -m experiment_runner summary /path/to/results/run
python -m tools.plot_height --run 'Minimal=/path/to/results/run' --output height.png
python -m experiment_runner.artifacts /path/to/results/run results.zip
```

Plotting is local-only and requires matplotlib (`uv pip install --project /path/to/repo
--python /path/to/analysis-venv/bin/python --group plot`). Plot data includes the number of available episodes per step;
missing heights/steps are not replaced by zero. Use `--descent` for height loss
relative to reset. Packing never removes source results. Partial results require
`--allow-partial`. ZIP checksums are stored and CRC is verified.

Metrics explicitly distinguish:

* Generated TPS: sampled **role** tokens, including terminal tokens, divided by
  full workload wall time, including the first action and background role drain.
* Tokens/action: the same role-token count / completed actions, an amortized cost.
* Mean decode batch: active workers per actual decode-forward, excluding padding.
* Mean prefill batch: both active requests and new token rows per prefill-forward.
* `output_tokens_including_readouts_tps`: additionally counts action readouts,
  matching the old launcher's numerator; it does not drop its first-action interval.
* `decode_rows_per_second`: input decode rows, **not** all generated output tokens.
* Sampled GPU utilization: nvidia-smi samples, **not** a trace-derived idle fraction.

No CUDA synchronization is added per forward. Per-client overlapping windows are
not summed as global token counts. Means use sums/counts, not means of means.
Partial/failed runs are marked, not silently treated as successful repeats.
Model initialization and episode/reset/flush boundaries remain separately recorded.
`experiment_runner/fi_warmup.py` owns the per-process sampler warmup. It makes one
synthetic call with its own RNG after Engine initialization, before any episodes.
This is not a Setup check or a discarded agent action.

There is no runner workspace option or custom engine workspace parameter.
The existing Attention backend allocates/binds its scratch; prefill graph setup
queries FlashInfer's required sizes and grows buffers when necessary. Runner does
not inspect or patch these wrappers.
Construction is timed as a whole; loader/capture methods are no longer patched just
to split their internal timings.

## Validation status

CPU tests cover independent repeats, seed
assignment, event-loop-safe recording, all media switches, feedback order and block
cleanup, fail-fast, arithmetic of overlapping episodes, plotting data, result packing,
and Setup ordering/download error handling.
Run them with `python -m pytest tests -q` in a test environment with NumPy/Pillow.
`tools.check_world` and `tools.small_check` retain the previous integration checks.
On the previous Falsifier strategy, before the experiment-file layout refactor,
they passed native reset/step and a real Qwen3.5-0.8B FP8 run on GB10:
2 slots × 2 repeats × 2 actions, one engine load, all 8 actions completed, role
and engine token counts reconciled, PNG/GIF switches exercised. Setup completed
in an isolated venv with the pinned native Craftium build.

Historical **previous Falsifier**, not this V9 port, measurement (including the old 2 GiB workspace override), with
real Qwen3.6-35B-A3B FP8 on RTX PRO6000,15pipelines ×5actions:

| Metric | New runner | Original portable control |
| --- | ---: | ---: |
| Actions completed |75|75|
| Full workload seconds |42.507|48.263|
| Role tokens/sec |222.03|198.70|
| Tokens/sec including action readouts (old numerator) |223.80|200.26|
| Mean decode batch |28.16|29.07|
| Mean prefill requests |6.75|9.86|
| Mean new prefill rows |422.08|511.10|

Same engine commit70942c4, checkpoint revision, layout and assigned role seeds;
no profiler and no first-action exclusion. This is a single paired experiment,
not a statistical speed guarantee. The old control writes PNGs and steps worlds
synchronously; new runner uses async World/Recorder with media disabled. Scheduling
and initial-world readiness differ. Prompts and role/core logic are preserved,
not bit-identical trajectories. Native imports passed; no OOM; all229result files
were downloaded and SHA256-verified before releasing the GPU. Real multi-GPU
execution was not available; GPU assignment/repeat scheduling are covered by CPU tests.

The stock-workspace/coroutine path was subsequently measured with 07-cuda-graphs
on RTX PRO6000, FP8 A3B weights, 15×10: 191.6 role tokens/s, 97.0 seconds, 150 actions.
This is a separate single-run reference, not a matched ablation of the table above.

Current V9 CPU tests control the live one-token wait, planner spanning actions
with an unchanged history snapshot, failure cleanup and independent episode RNGs.
Role/system/history prompt hashes are checked against the historical fast
falsifier policy. Falsifier and objection-related action instructions are restored.
A 100-action fake-model test checks budgets, planner starts (also for
EOS-only plans), exact readout block order, and historical RNG mapping. No GPU
inference-quality claim is made for the modified policy from these CPU tests.

Historical note (before the restoration in this revision): fast async without
falsifier was GPU-smoke-tested on RTX PRO 6000 with engine
`bbe7abf55512342d56af61d4009dff0174d8896d`: 15 pipelines x 2 actions completed,
including background-role drain. The full long-run engine settings were retained:
49,152 KV pages of 16 tokens, context 32,768, prefill cap 1024 and graph sizes
[4,16,48,64]. These settings match the archived successful H200 15x150x7 run,
apart from the deployment-specific model path. That older no-falsifier result
does not validate the new single-world parameters. Use the current entry point
and validation scope in the Single-world long runs section above.

## Release tools

`tools/build_release.py` builds a runner-source ZIP without engine, weights, venv
or Git bundle. The recipient supplies an existing engine checkout to setup:

```bash
python tools/build_release.py --output /path/to/release.zip
```

The generated `tools/release.json` contains source file checksums.
Neither Runner nor an experiment reads it. Results packaging is a separate helper
(`python -m experiment_runner.artifacts`), not the source-release builder.
