"""Measured-map rendering for model input and separately annotated review videos."""

import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def projection(grid, size=512):
    height, width = grid.values.shape
    scale = min((size - 32) / width, (size - 52) / height)
    return {
        "scale": scale,
        "offset_x": (size - width * scale) / 2,
        "offset_y": 34 + (size - 44 - height * scale) / 2,
        "height": height,
        "width": width,
        "resolution": grid.resolution,
        "origin": list(grid.origin),
        "yaw": grid.yaw,
    }


def pixel(xy, p):
    dx, dy = xy[0] - p["origin"][0], xy[1] - p["origin"][1]
    c, s = math.cos(p.get("yaw", 0)), math.sin(p.get("yaw", 0))
    x = (c * dx + s * dy) / p["resolution"]
    y = (-s * dx + c * dy) / p["resolution"]
    return (p["offset_x"] + x * p["scale"], p["offset_y"] + (p["height"] - y) * p["scale"])


def render_map(grid, pose, candidates=(), path=(), active=None, objects=(), size=512):
    p = projection(grid, size)
    pixels = np.full((*grid.values.shape, 3), 112, dtype=np.uint8)
    pixels[(grid.values >= 0) & (grid.values <= 35)] = (239, 240, 233)
    pixels[grid.values >= 65] = (30, 37, 45)
    pixels[(grid.values > 35) & (grid.values < 65)] = (156, 151, 132)
    base = Image.fromarray(np.flipud(pixels))
    base = base.resize(
        (max(1, round(p["width"] * p["scale"])), max(1, round(p["height"] * p["scale"]))),
        Image.Resampling.NEAREST,
    )
    image = Image.new("RGB", (size, size), (20, 28, 39))
    image.paste(base, (round(p["offset_x"]), round(p["offset_y"])))
    draw = ImageDraw.Draw(image)
    draw.text((12, 7), f"ONLINE SLAM | v{grid.version} | {grid.frame}", font=font(15), fill="white")
    if len(path) > 1:
        draw.line([pixel(xy, p) for xy in path], fill=(70, 163, 232), width=2)
    for obj in objects:
        x, y = pixel(obj["xy"], p)
        draw.rectangle((x - 3, y - 3, x + 3, y + 3), fill=(239, 147, 210))
    for candidate in candidates:
        x, y = pixel((candidate.x, candidate.y), p)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(32, 117, 87), outline="white", width=1)
        draw.text((x - 5, y - 8), candidate.id, font=font(12), fill="white")
    if active is not None:
        x, y = pixel((active.x, active.y), p)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=(255, 218, 64), width=3)
        draw.text((x + 12, y - 14), "WAYPOINT", font=font(11), fill=(255, 218, 64))
    x, y = pixel(pose[:2], p)
    angle = pose[2] - p.get("yaw", 0)
    points = [
        (x + 12 * math.cos(angle), y - 12 * math.sin(angle)),
        (x + 8 * math.cos(angle + 2.5), y - 8 * math.sin(angle + 2.5)),
        (x + 8 * math.cos(angle - 2.5), y - 8 * math.sin(angle - 2.5)),
    ]
    draw.polygon(points, fill=(0, 219, 231), outline="black")
    draw.text(
        (12, size - 17), "gray unknown | white free | black occupied", font=font(11), fill="white"
    )
    return image, p


def video(output, fps=5):
    """Truth target is added ONLY here, never to the policy's map image."""
    import cv2

    frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    if not frames:
        raise ValueError("No recorded frames")
    scenario = json.loads((output / "evaluation_scenario.json").read_text())
    config_file = output / "run_config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    title = scenario["name"]
    if config:
        method = (
            "direct commands"
            if config.get("control_mode") == "direct"
            else config["candidate_mode"]
        )
        title += f" | {method} | seed {config['sensor_seed']}"
    spawn = scenario["spawn"]
    target = scenario["target_xy"]
    c, s = math.cos(spawn[2]), math.sin(spawn[2])
    dx, dy = target[0] - spawn[0], target[1] - spawn[1]
    target_map = (c * dx + s * dy, -s * dx + c * dy)
    destination = output / "replay.mp4"
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Cannot create video")
    decisions = []
    if (output / "decisions.jsonl").exists():
        decisions = [
            json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()
        ]
    summary = (
        json.loads((output / "summary.json").read_text())
        if (output / "summary.json").exists()
        else {}
    )
    start, end = frames[0]["sim_time"], frames[-1]["sim_time"]
    index = decision_index = 0
    try:
        for stamp in np.arange(start, end + 1 / fps, 1 / fps):
            while index + 1 < len(frames) and frames[index + 1]["sim_time"] <= stamp:
                index += 1
            frame = frames[index]
            canvas = Image.new("RGB", (1280, 720), (14, 21, 31))
            draw = ImageDraw.Draw(canvas)
            camera = Image.open(output / frame["camera"]).convert("RGB").resize((640, 480))
            map_image = Image.open(output / frame["map"]).convert("RGB").resize((600, 600))
            # Coordinate projection initially matches the SLAM origin; the marker
            # is privileged evaluation metadata, not a localization measurement.
            tx, ty = pixel(target_map, frame["projection"])
            tx *= 600 / 512
            ty *= 600 / 512
            md = ImageDraw.Draw(map_image)
            if not scenario.get("absent") and 0 <= tx < 600 and 0 <= ty < 600:
                md.line((tx - 9, ty - 9, tx + 9, ty + 9), fill=(255, 65, 88), width=4)
                md.line((tx - 9, ty + 9, tx + 9, ty - 9), fill=(255, 65, 88), width=4)
                md.text(
                    (max(0, min(430, tx + 12)), max(30, ty - 22)),
                    "TARGET [evaluation]",
                    font=font(13),
                    fill=(255, 65, 88),
                )
            elif not scenario.get("absent"):
                # Keep the evaluator's target visible while SLAM has only
                # discovered a small patch. This overlay never enters render_map.
                vx, vy = tx - 300, ty - 300
                edge = min(275 / max(abs(vx), 1e-9), 255 / max(abs(vy), 1e-9))
                ex, ey = 300 + edge * vx, 300 + edge * vy
                norm = max(math.hypot(vx, vy), 1e-9)
                ux, uy = vx / norm, vy / norm
                md.line((ex - ux * 32, ey - uy * 32, ex, ey), fill=(255, 65, 88), width=4)
                md.polygon(
                    [
                        (ex, ey),
                        (ex - ux * 13 - uy * 7, ey - uy * 13 + ux * 7),
                        (ex - ux * 13 + uy * 7, ey - uy * 13 - ux * 7),
                    ],
                    fill=(255, 65, 88),
                )
                md.text(
                    (max(5, min(365, ex - 100)), max(35, min(555, ey + 12))),
                    "TARGET outside map [evaluation]",
                    font=font(12),
                    fill=(255, 65, 88),
                )
            canvas.paste(camera, (16, 60))
            canvas.paste(map_image, (668, 60))
            draw.text((16, 12), title, font=font(24), fill="white")
            draw.text(
                (16, 548),
                f"simulation {stamp - start:.1f}s | state: {frame['state']}"
                + (f" | {frame['action']}" if frame.get("action") else ""),
                font=font(17),
                fill=(110, 225, 211),
            )
            draw.text(
                (16, 578),
                f"distance travelled: {frame.get('path_m', 0):.2f} m | decisions: {frame.get('decisions', 0)}",
                font=font(16),
                fill="white",
            )
            while (
                decision_index + 1 < len(decisions)
                and decisions[decision_index + 1].get("sim_time", 0) <= stamp
            ):
                decision_index += 1
            decision = (
                decisions[decision_index]
                if decisions and decisions[decision_index].get("sim_time", 0) <= stamp
                else {}
            )
            text = decision.get("observer", "Waiting for model observations...").replace("\n", " ")
            for line_index, line in enumerate(textwrap.wrap(text, 68)[:3]):
                draw.text((16, 611 + line_index * 20), line, font=font(14), fill=(189, 203, 222))
            draw.text(
                (16, 692),
                "RGB + online SLAM + IMU/odom | "
                + (
                    "Qwen drives directions"
                    if config.get("control_mode") == "direct"
                    else "Qwen chooses waypoints"
                )
                + " | no preloaded map",
                font=font(14),
                fill="white",
            )
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        for _ in range(fps * 3):
            final = canvas.copy()
            draw = ImageDraw.Draw(final)
            draw.rectangle((200, 240, 1080, 450), fill=(18, 29, 43), outline="white", width=2)
            draw.text(
                (235, 265),
                "PASS" if summary.get("passed") else "NOT COMPLETED",
                font=font(32),
                fill=(100, 230, 175) if summary.get("passed") else (255, 160, 120),
            )
            draw.text(
                (235, 320),
                f"{summary.get('reason', 'unknown')} | path {summary.get('truth_path_m', 0):.2f} m",
                font=font(20),
                fill="white",
            )
            draw.text(
                (235, 365),
                "Red X: evaluator-only target. "
                + (
                    "Direct commands; no waypoints."
                    if config.get("control_mode") == "direct"
                    else "Yellow circle: model waypoint."
                ),
                font=font(18),
                fill="white",
            )
            writer.write(cv2.cvtColor(np.asarray(final), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    # H.264 for browser/app playback; preserve original until conversion succeeds.
    import subprocess

    h264 = output / "replay_h264.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(destination),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(h264),
        ],
        check=True,
    )
    h264.replace(destination)
    (output / "video_provenance.json").write_text(
        json.dumps(
            {
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fps": fps,
                "size": [1280, 720],
                "target_annotation": "offline_evaluator_only",
                "off_map_target_arrow": True,
                "source_frames": len(frames),
            },
            indent=2,
        )
    )
    return destination


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(video(args.output))


if __name__ == "__main__":
    main()
