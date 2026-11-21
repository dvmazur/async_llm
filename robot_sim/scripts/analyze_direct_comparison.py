"""Offline analysis of frozen direct trials; never imported by rover policies."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "robot_sim/results/candidate_navigation"
COLORS = {"geometric": "#218d68", "qwen": "#3984c8", "direct": "#dc6545"}
LABELS = {
    "geometric": "Geometric + Qwen + Nav2",
    "qwen": "Qwen coordinates + Nav2",
    "direct": "Qwen direct commands",
}


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot(destination, groups, tasks, roots):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(14, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.8, 1.25))
    ax = fig.add_subplot(grid[0, 0])
    counts = [sum(row["passed"] for row in group) for group in groups.values()]
    bars = ax.bar(range(3), counts, color=list(COLORS.values()), width=0.65)
    ax.set_xticks(range(3), ["Geometric\nwaypoints", "Qwen\nwaypoints", "Direct\ncommands"])
    ax.set_ylim(0, 15.5)
    ax.set_ylabel("Successful episodes / 14")
    ax.set_title("Complete systems, same scenes/seeds/budgets")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.25,
            f"{count}/14",
            ha="center",
            weight="bold",
        )
    ax = fig.add_subplot(grid[0, 1])
    matrix = np.array(
        [
            [sum(r["passed"] for r in group if r["task"] == task) for task in tasks]
            for group in groups.values()
        ]
    )
    ax.imshow(matrix, vmin=0, vmax=2, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(range(3), ["Geometric", "Qwen coords", "Direct"])
    ax.set_xticks(range(len(tasks)), [t.replace("_", "\n") for t in tasks], fontsize=9)
    ax.set_title("Successes per scene (two seeds)")
    for y in range(3):
        for x in range(len(tasks)):
            ax.text(
                x,
                y,
                f"{matrix[y, x]}/2",
                ha="center",
                va="center",
                color="white" if matrix[y, x] == 2 else "black",
                weight="bold",
            )
    for position, task in enumerate(("blue_occluded", "yellow_zigzag")):
        ax = fig.add_subplot(grid[1, position])
        scene = read(roots["direct"] / f"{task}__direct__s29/evaluation_scenario.json")
        xmin, xmax, ymin, ymax = scene["bounds"]
        for x, y, sx, sy in scene["partitions"]:
            ax.add_patch(Rectangle((x - sx / 2, y - sy / 2), sx, sy, color="#454545", zorder=3))
        for color, x, y in scene["distractors"]:
            ax.add_patch(
                Rectangle(
                    (x - 0.2, y - 0.2), 0.4, 0.4, facecolor=color, edgecolor="black", alpha=0.45
                )
            )
        for mode, root in roots.items():
            episode = f"{task}__{mode}__s29"
            trajectory = np.array(
                [r["truth"] for r in records(root / episode / "evaluation_trajectory.jsonl")]
            )
            row = next(r for r in groups[mode] if r["episode"] == episode)
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=COLORS[mode],
                lw=1.6,
                alpha=0.85,
                label=f"{mode}: {row['truth_path_m']:.1f} m, {'PASS' if row['passed'] else 'FAIL'}",
            )
            ax.plot(*trajectory[-1], "o", color=COLORS[mode], ms=5)
        ax.scatter(
            *scene["target_xy"],
            marker="X",
            s=100,
            color="#cf2a58",
            zorder=5,
            label="Target (evaluation only)",
        )
        ax.scatter(*scene["spawn"][:2], marker="*", s=100, color="black", zorder=5, label="Start")
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="World x, m", ylabel="World y, m")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_title(f"{task}, seed 29: measured truth trajectories")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Qwen 27B rover comparison: 42 primary episodes\nDirect: forward/left/right/stop with bounded steps and stopped inference",
        fontsize=15,
    )
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=BASE / "direct_comparison_20261106_v2")
    p.add_argument("--baseline", type=Path, default=BASE / "candidate_comparison_20261105_v1")
    args = p.parse_args()
    root, baseline = args.root.resolve(), args.baseline.resolve()
    result, old = read(root / "comparison.json"), read(baseline / "comparison.json")
    manifest = read(root / "experiment.json")
    errors = list(result["configuration_errors"])
    if (
        hashlib.sha256((baseline / "comparison.json").read_bytes()).hexdigest()
        != manifest["baseline_comparison_sha256"]
    ):
        errors.append("Baseline comparison changed")
    for path, digest in manifest["source_hashes"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            errors.append("Frozen source changed: " + path)
    groups = {
        mode: [r for r in old["episodes"] if r["mode"] == mode] for mode in ("geometric", "qwen")
    }
    groups["direct"] = result["episodes"]
    modes, scenes = {}, []
    for mode, rows in groups.items():
        present = [r for r in rows if not r["target_absent"]]
        absent = [r for r in rows if r["target_absent"]]
        modes[mode] = {
            "completed": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "present_passed": sum(r["passed"] for r in present),
            "present_total": len(present),
            "absent_passed": sum(r["passed"] for r in absent),
            "absent_total": len(absent),
            "mean_approximate_spl": statistics.mean(r["approximate_spl"] for r in present),
            "wall_seconds": sum(r["wall_seconds"] for r in rows),
            "completed_backend_seconds_per_decision": sum(
                r["agent"]["selection_seconds"] + r["agent"].get("generation_seconds", 0)
                for r in rows
            )
            / sum(r["agent"]["decisions"] for r in rows),
        }
    latency, client, durations = [], [], Counter()
    action_counts, outcomes = Counter(), Counter()
    for row in groups["direct"]:
        directory = root / row["episode"]
        audit = read(directory / "artifact_audit.json")
        if not audit["passed"]:
            errors.append("Audit failed: " + row["episode"])
        hashes = read(directory / "source_hashes.json")
        for path, digest in hashes.items():
            if path in manifest["source_hashes"] and manifest["source_hashes"][path] != digest:
                errors.append("Episode source differs: " + row["episode"] + " " + path)
        config = read(directory / "run_config.json")
        for other in old["episodes"]:
            if other["task"] == row["task"] and other["seed"] == row["seed"]:
                for key in (
                    "world_sha256",
                    "task_sha256",
                    "sensor_seed",
                    "sim_limit",
                    "wall_limit",
                ):
                    if config[key] != other["config"][key]:
                        errors.append("Unequal " + key + ": " + row["episode"])
        events, decisions = (
            records(directory / "events.jsonl"),
            records(directory / "decisions.jsonl"),
        )
        actions = [e for e in events if e["event"] == "command_finished"]
        forwards = [a for a in actions if a["action"] == "forward"]
        counts = Counter(a["action"] for a in actions)
        action_counts.update(counts)
        outcomes.update(a["outcome"] for a in actions)
        latency.extend(d["latency_s"] for d in decisions)
        client.extend(d["snapshot_wall_age"] for d in decisions)
        target_visible, near = [], []
        color = read(directory / "evaluation_scenario.json")["target_color"]
        for file in sorted((directory / "observations").glob("*/observation.json")):
            observation = read(file)
            objects = [o for o in observation["objects"] if o["visible"] and o["color"] == color]
            if objects:
                target_visible.append(observation["snapshot_id"])
                if min(o["range_m"] for o in objects) <= 0.8:
                    near.append(observation["snapshot_id"])
        end = next(e["sim_time"] for e in events if e["event"] == "finished")
        start = end - row["agent"]["sim_seconds"]
        frames = records(directory / "frames.jsonl")
        for frame, following in zip(frames, frames[1:]):
            durations[frame["state"]] += max(
                0, min(following["sim_time"], end) - max(frame["sim_time"], start)
            )
        trace = records(directory / "evaluation_trajectory.jsonl")
        scenes.append(
            {
                "episode": row["episode"],
                "task": row["task"],
                "seed": row["seed"],
                "passed": row["passed"],
                "reason": row["reason"],
                "path_m": row["truth_path_m"],
                "minimum_truth_distance_m": row["minimum_distance_to_target_m"],
                "actions": dict(counts),
                "forward_steps": len(forwards),
                "forward_brakes": sum(a["outcome"] == "lidar_brake" for a in forwards),
                "forward_under_5cm": sum(a["distance_m"] < 0.05 for a in forwards),
                "adjacent_opposite_turns": sum(
                    a["action"] in ("left", "right")
                    and b["action"] in ("left", "right")
                    and a["action"] != b["action"]
                    for a, b in zip(actions, actions[1:])
                ),
                "requested_color_visible_snapshots": len(target_visible),
                "requested_color_near_snapshots": near,
                "truth_max_x": max(t["truth"][0] for t in trace),
                "video_bytes": (directory / "replay.mp4").stat().st_size,
            }
        )
    analysis = {
        "verification_errors": errors,
        "modes": modes,
        "completed_actions": dict(action_counts),
        "completed_action_outcomes": dict(outcomes),
        "decisions": len(latency),
        "backend_latency_median_s": statistics.median(latency),
        "client_latency_median_s": statistics.median(client),
        "frame_sampled_state_sim_seconds": dict(durations),
        "frame_sampled_thinking_fraction": durations["thinking_stopped"] / sum(durations.values()),
        "episodes": scenes,
    }
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2))
    if errors:
        raise ValueError(errors)
    plot(
        root / "comparison_analysis.png",
        groups,
        manifest["tasks"],
        {"geometric": baseline, "qwen": baseline, "direct": root},
    )
    lines = [
        "# Прямое управление Qwen 27B: результаты",
        "",
        "Завершены все 14 основных эпизодов v2. Аудиты 14/14, контактов 0. Исходники замороженной серии не менялись, фактические сцены/задания/seed/бюджеты совпадают с baseline. Проверочных ошибок: 0.",
        "",
        "| Метод | Все задачи | Цель присутствует | Цель отсутствует | Прибл. SPL | Backend с/решение | Суммарно wall, ч |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, a in modes.items():
        lines.append(
            f"| {LABELS[mode]} | {a['passed']}/{a['completed']} | {a['present_passed']}/{a['present_total']} | {a['absent_passed']}/{a['absent_total']} | {a['mean_approximate_spl']:.3f} | {a['completed_backend_seconds_per_decision']:.2f} | {a['wall_seconds'] / 3600:.2f} |"
        )
    lines += [
        "",
        "| Сцена | Geometric + Nav2 | Qwen точки + Nav2 | Прямые команды |",
        "|---|---:|---:|---:|",
    ]
    for task in manifest["tasks"]:
        values = [sum(r["passed"] for r in rows if r["task"] == task) for rows in groups.values()]
        lines.append(f"| {task} | {values[0]}/2 | {values[1]}/2 | {values[2]}/2 |")
    lines += [
        "",
        "## Что показывают действия и траектории",
        "",
        f"Модель приняла {len(latency)} решений. Завершены {sum(action_counts.values())} примитивов: forward {action_counts['forward']}, left {action_counts['left']}, right {action_counts['right']}. Отдельных обычных stop модель не выбирала: два stop были подтверждением близкой цели. Остальные решения включают примитивы, прерванные концом бюджета; они не входят в счётчик завершённых действий.",
        "",
        f"Медиана backend-задержки {statistics.median(latency):.2f} с; медиана полной клиентской задержки {statistics.median(client):.2f} с. По интеграции состояний сохранённых кадров thinking_stopped занимает {analysis['frame_sampled_thinking_fraction']:.1%} времени симуляции. Это оценка по кадрам, а не аппаратный профиль GPU. Прямой ровер стоит во время инференса; waypoint-контур может двигаться.",
        "",
        f"Защита остановила {sum(s['forward_brakes'] for s in scenes)} из {action_counts['forward']} forward-шагов; {sum(s['forward_under_5cm'] for s in scenes)} шагов дали <5 см перемещения. Само торможение перед препятствием ожидаемо; показатель вместе с повторными поворотами указывает на трудности выбора дальнейшего движения, а не на доказанную ошибку защиты.",
        "",
        "На red_deadends seed43 ровер проехал 3,20 м за 500 с симуляции: 43 завершённых поворота и 29 соседних пар с противоположным направлением. Это измеримые колебания, но количество пар включает и возможные разумные корректировки направления.",
        "",
        "На yellow_zigzag seed29 прямой ровер проехал 29,83 м, однако максимальная world-x была −4,41 м: он не прошёл первую перегородку около x=−4. Цель находится в x=8. Geometric-режим дошёл до цели за 27,21 м, Qwen-waypoint за 50,54 м. Длина пути без пространственного продвижения не означает успешное исследование.",
        "",
        "В десяти сложных эпизодах сенсорные RGB/lidar-гипотезы не отметили искомый цвет как видимый ни в одном снимке, отправленном LLM. Это относится к сохранённым входным снимкам и данному детектору, не доказывает отсутствие цели во всех кадрах видео. На двух близких целях модель правильно подтвердила завершение при измеренном расстоянии 0,76/0,35 м. Наблюдаемый провал возникает преимущественно до получения близкого визуального подтверждения цели.",
        "",
        "## Интерпретация и ограничения",
        "",
        "Эта реализация прямых дискретных команд уступила обеим waypoint-схемам. Логи совместимы с сочетанием низкой частоты действий и недостаточной устойчивости исследования/обхода тупиков. Разделить причинный вклад задержки, памяти, подсказок и дискретизации без отдельных абляций нельзя. Это не вывод о невозможности прямого управления другими LLM или другими примитивами.",
        "",
        "Два успешных absent_yellow означают только отсутствие ложного заявления об успехе при минимально требуемом движении за 90 с. Это не доказательство полноценного обследования или отсутствия объекта.",
        "",
        "Сравниваются целые контуры при одинаковом бюджете, а не рассуждение при одинаковом числе действий. Режимы запускались в разное время, ROS/Gazebo не побитово детерминированны. На старте direct существовал посторонний GPU-процесс, которому не мешали; поэтому wall/backend-время нельзя считать чистым сравнением производительности алгоритмов. Backend с/решение включает завершённые генерационные запросы Qwen-waypoint; не включает всю очередь и анализ после раннего ответа. Приблизительный SPL использует прежнюю evaluator-only 10-см сетку, исключает отсутствующие цели и присваивает неудачам 0.",
        "",
        "Основная статистика — 28 baseline +14 direct v2. Не включены пилоты, старый direct v1 с невалидным стартовым временем и два дополнительных инфраструктурных повтора baseline. Единственный исходный Nav2 startup failure в baseline сохранён. По два seed на сцену дают инженерное сравнение конкретных конфигураций, не оценку общей надёжности.",
        "",
        "![Результаты и траектории](comparison_analysis.png)",
        "",
        "[Все 14 видео](index.html) · [Исходная таблица](COMPARISON.md) · [Численные диагностики](analysis.json)",
        "",
        "Полная геометрия и истинные траектории на рисунке используются только для этого офлайн-анализа и не подавались политикам.",
        "",
    ]
    (root / "ANALYSIS.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in analysis.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
