# Перенос на сервер с RTX 3090 — 29 октября 2026

Первоначально проект опубликован в ветке `rover` репозитория
https://github.com/Doub1e05/async_llm.
На этапе первоначального переноса в dvmazur/async_llm изменения не отправлялись.

## Установленная система

- Сервер: 5.187.77.37, пользователь double.
- Репозиторий: /home/double/data_scince_expedition/lets/async_llm.
- Conda: /home/double/miniconda3/envs/double_p, Python 3.12.16.
- ROS 2 Jazzy, Gazebo Sim 8, Nav2, slam_toolbox, robot_localization.
- PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA Toolkit 13.0.
- Transformers 5.14.1, Accelerate 1.13.0; исходный AsyncLLM установлен editable.
- Две RTX 3090, по 24 ГБ. Симулятор ограничен одним логическим CPU,
  использует программный рендеринг; LLM по умолчанию использует одну GPU.
- Xvfb/noVNC установлены в .runtime без sudo.

Conda решена заново для Ryzen. Старый Intel AVX-512 explicit lock не использован.
Пути к Python, Conda, HF cache вынесены в игнорируемый .runtime/config.sh.
На новом сервере origin использует публичный HTTPS для fetch/pull;
GitHub SSH для push с нового сервера не настраивался.

## Модели

В стандартном /home/double/.cache/huggingface/hub полностью скачаны:

| Модель | Snapshot |
| --- | --- |
| Qwen/Qwen3.5-0.8B | 2fc06364715b967f1860aea9cf38778875588b17 |
| Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a |

Веса, окружение, ROS bags и записи прежних экспериментов в Git не входят.
Предыдущие видео и результаты остаются на старом сервере.

## Проверки

| Проверка | Результат |
| --- | --- |
| Python unit tests после установки зависимостей | 38/38 PASS |
| pip check | Нет нарушенных зависимостей |
| CUDA BF16 matmul | PASS на каждой из двух карт |
| Прямое копирование тензоров между GPU | 6/6 PASS, оба направления |
| Физика ровера | PASS: вперёд, назад, повороты, дуга, торможение, watchdog, сенсоры, TF |
| Картография | Маршрут 4/4, 0 контактов; движущийся объект обнаружен и очищен с карты |
| Gazebo desktop/noVNC | HTTP 200; семь дочерних процессов живы при проверке |
| Qwen3.5-9B, Transformers BF16, одна GPU | Изображение и state_v2 приняты; forward, p=0.99886 |
| Qwen3.5-0.8B, исходный AsyncLLM BF16, одна GPU | Изображение и state_v2 приняты; forward, p=0.54709 |

Для обеих моделей проверены контрольная сумма переданного изображения,
допустимая команда и конечные нормированные вероятности.
Пик выделенной PyTorch памяти для 9B — 18 226.5 MiB.
Первый старт 0.8B с компиляцией CUDA kernels занял около 151 с.
Это проверки переноса и инференса на синтетическом кадре: они не измеряют
качество вождения. Полный проезд ровера под LLM на новом сервере ещё не оценивался.
9B через исходный AsyncLLM и распределение самой модели на две карты
в этой проверке не запускались. Nav2 установлен; отдельный новый Nav2 benchmark
не проводился. Браузерная проверка подтверждает запуск компонентов и HTTP,
но не заменяет визуальную оценку сцены.

Все проверочные процессы остановлены, GPU освобождены.

Подробные локальные свидетельства: robot_sim/results/server_setup —
physics/summary.json, mapping/summary.json, cuda.json, gui.json,
llm_smoke.json, unit_tests_after_install.log, pip-freeze.txt,
conda-linux-64-explicit.txt. Эта папка игнорируется Git.

## Запуск

```bash
ssh 5.187.77.37
cd /home/double/data_scince_expedition/lets/async_llm
conda activate double_p

# Qwen-9B, проверенный reference backend:
ROBOT_LLM_GPU=0 bash robot_sim/scripts/run_llm.sh \
  --backend transformers --gpu-memory-gib 21 --vision

# Qwen-0.8B, проверенный исходный AsyncLLM (вместо предыдущего процесса):
MODEL=Qwen/Qwen3.5-0.8B ROBOT_LLM_GPU=0 \
  bash robot_sim/scripts/run_llm.sh --backend async --memory-ratio 0.3 --vision
```

Симуляция и картография запускаются отдельными командами из DEPLOYMENT.md.
Для desktop-просмотра открыть SSH-туннель на своём компьютере:

```bash
ssh -L 17080:127.0.0.1:17080 5.187.77.37
cd /home/double/data_scince_expedition/lets/async_llm
bash robot_sim/scripts/run_gui.sh warehouse --robot rover
```

Затем открыть http://127.0.0.1:17080/vnc.html.
Пароль текущего сеанса хранится на сервере в
robot_sim/.runtime/viewer/session/password; прочитать его в другом терминале.
Ctrl+C завершает desktop и дочерние процессы.

Следующий эксперимент: совместить реальный кадр Gazebo, сенсорную карту и
Qwen-9B в замкнутом цикле; записать несколько задач и измерить успех,
контакты, время, длину пути и задержку решения.


## Публикация в основном репозитории

После предоставления SSH-доступа все пять коммитов робототехнической части
объединены с актуальным main (c03eed4) основного dvmazur/async_llm.
Рабочая ветка основного репозитория — rover. История исходных изменений сохранена.
Инструкции DEPLOYMENT.md теперь используют git@github.com:dvmazur/async_llm.git.
Результаты проверок переноса выше относятся к состоянию до этого объединения;
после объединения все 38 тестов robot_sim прошли. Лог проверки на старом
сервере: robot_sim/results/upstream_merge/robot_tests.log.
