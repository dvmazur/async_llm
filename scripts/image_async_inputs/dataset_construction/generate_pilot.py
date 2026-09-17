"""Render five controlled correction pairs; no model calls or external data.

Run: python3 scripts/image_async_inputs/dataset_construction/generate_pilot.py
"""

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pilot_v1"
SIZE = (960, 640)
NOTICE = (
    "The earlier picture contained an error. It has now been corrected. "
    "Recheck your reasoning using the current picture."
)
BLUE, ORANGE = "#2864b4", "#d56b24"


def font(size):
    return ImageFont.truetype("DejaVuSans.ttf", size)


def label(draw, xy, value, size=26, fill="#172337"):
    draw.text(xy, str(value), font=font(size), fill=fill, anchor="mm")


def canvas(title):
    image = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(image)
    label(draw, (480, 48), title, 32)
    return image, draw


def render(kind, state):
    image, draw = canvas(state["title"])
    if kind == "geometry":
        draw.rectangle((240, 170, 720, 430), outline="#172337", width=4)
        draw.line((240, 430, 720, 170), fill=BLUE, width=3)
        label(draw, (480, 475), "8")
        label(draw, (770, 300), state["height"])
        label(draw, (480, 275), "d")
        label(draw, (480, 560), "Rectangle; diagram not to scale", 22)
    elif kind in ("bar", "legend"):
        draw.line((150, 140, 150, 500, 840, 500), fill="#172337", width=3)
        for value in range(0, 21, 5):
            y = 500 - 16 * value
            draw.line((150, y, 840, y), fill="#d9dee5", width=1)
            label(draw, (110, y), value, 22)
        for value in range(21):
            y = 500 - 16 * value
            draw.line((143, y, 150, y), fill="#172337", width=2)
            if value % 5:
                draw.line((151, y, 840, y), fill="#edf0f4", width=1)
        if kind == "bar":
            for x, name, value in zip((320, 640), ("A", "B"), state["values"]):
                draw.rectangle((x - 65, 500 - 16 * value, x + 65, 499), fill=BLUE)
                label(draw, (x, 535), name)
            label(draw, (480, 590), "Output (units)", 22)
        else:
            for x, name, values in zip((330, 650), ("Period 1", "Period 2"),
                                       ((12, 8), (16, 10))):
                for offset, color, value in zip((-45, 45), (BLUE, ORANGE), values):
                    draw.rectangle((x + offset - 32, 500 - 16 * value,
                                    x + offset + 32, 499), fill=color)
                label(draw, (x, 535), name)
            for x, color, name in zip((290, 600), (BLUE, ORANGE), state["legend"]):
                draw.rectangle((x - 80, 92, x - 50, 122), fill=color)
                label(draw, (x + 35, 107), name, 23)
            label(draw, (480, 590), "Sales (units)", 22)
    elif kind == "receipt":
        for y in (170, 230, 310, 390):
            draw.line((130, y, 830, y), fill="#a0a9b5", width=2)
        for x, text in zip((270, 510, 730), ("Item", "Quantity", "Unit price")):
            label(draw, (x, 200), text)
        for y, row in zip((270, 350), (("Notebook", state["quantity"], "$4"),
                                      ("Pen", 3, "$2"))):
            for x, text in zip((270, 510, 730), row):
                label(draw, (x, y), text)
        label(draw, (480, 485), "Prices before discount; no tax", 24)
    elif kind == "graph":
        positions = {"S": (180, 310), "A": (480, 170),
                     "B": (480, 460), "T": (780, 310)}
        for a, b, weight in state["edges"]:
            p, q = positions[a], positions[b]
            draw.line((*p, *q), fill="#66758a", width=4)
            x, y = (p[0] + q[0]) // 2, (p[1] + q[1]) // 2
            draw.rectangle((x - 24, y - 24, x + 24, y + 24), fill="white")
            label(draw, (x, y), weight)
        for name, (x, y) in positions.items():
            draw.ellipse((x - 30, y - 30, x + 30, y + 30), fill="white",
                         outline=BLUE, width=4)
            label(draw, (x, y), name)
        label(draw, (480, 570), "Undirected graph; edge weights are travel times", 22)
    else:
        raise ValueError(kind)
    return image


def solve(kind, state):
    """Derive answers and observable intermediate facts from rendering data."""
    if kind == "geometry":
        diagonal = math.isqrt(8 ** 2 + state["height"] ** 2)
        assert diagonal ** 2 == 8 ** 2 + state["height"] ** 2
        return str(4 * diagonal), {"height": state["height"], "diagonal": diagonal}, (
            f"d = sqrt(8^2 + {state['height']}^2) = {diagonal}; "
            f"square perimeter = 4d = {4 * diagonal}."
        )
    if kind == "bar":
        a, b = state["values"]
        # Compare output per worker by cross multiplication, without float rounding.
        winner = "A" if a * 3 > b * 4 else "B"
        return winner, {"output_A": a, "output_B": b, "rate_A": a / 4,
                        "rate_B": b / 3}, f"Compare A: {a}/4 with B: {b}/3; choose {winner}."
    if kind == "legend":
        values = (12, 16) if state["legend"][0] == "North" else (8, 10)
        increase = values[1] - values[0]
        percent = Fraction(100 * increase, values[0])
        return str(percent), {"north_period_1": values[0], "north_period_2": values[1],
                              "increase": increase}, (
            f"North rises from {values[0]} to {values[1]}; "
            f"100 * {increase}/{values[0]} = {percent}%."
        )
    if kind == "receipt":
        subtotal = state["quantity"] * 4 + 3 * 2
        cents = subtotal * 90
        return f"{cents / 100:.2f}", {"notebook_quantity": state["quantity"],
                                     "subtotal": subtotal}, (
            f"Subtotal = {state['quantity']} * 4 + 3 * 2 = {subtotal}; "
            f"after 10% discount = ${cents / 100:.2f}."
        )
    weights = {(a, b): w for a, b, w in state["edges"]}
    via_a = weights["S", "A"] + weights["A", "T"]
    via_b = weights["S", "B"] + weights["B", "T"]
    return str(min(via_a, via_b)), {"via_A": via_a, "via_B": via_b,
                                   "route": "S-A-T" if via_a < via_b else "S-B-T"}, (
        f"The two simple routes cost {via_a} via A and {via_b} via B; "
        f"minimum = {min(via_a, via_b)}."
    )


# Correct states are specified first; each before-state overrides one semantic fact.
SPECS = [
    ("geometry", "length_label", {"title": "Rectangle", "height": 15},
     {"height": 6}, "Find the perimeter of a square whose side equals the diagonal "
     "of the rectangle in the picture.", "length units", "40", "68"),
    ("bar", "bar_height", {"title": "Team production", "values": [18, 12]},
     {"values": [12, 12]}, "Team A has 4 workers and team B has 3 workers. "
     "Which team has greater output per worker?", "team name", "B", "A"),
    ("legend", "legend_mapping", {"title": "Regional sales", "legend": ["South", "North"]},
     {"legend": ["North", "South"]}, "By what percentage did North's sales increase "
     "from Period 1 to Period 2? Give the exact percentage or fraction.", "percent", "100/3", "25"),
    ("receipt", "quantity_cell", {"title": "Stationery order", "quantity": 5},
     {"quantity": 2}, "A 10% discount applies to the entire order shown. "
     "How much must be paid? Give dollars to two decimal places.", "USD", "12.60", "23.40"),
    ("graph", "edge_weight", {"title": "Travel network",
     "edges": [["S", "A", 4], ["A", "T", 9], ["S", "B", 5], ["B", "T", 4]]},
     {"edges": [["S", "A", 4], ["A", "T", 2], ["S", "B", 5], ["B", "T", 4]]},
     "What is the minimum total travel time from S to T?", "time units", "6", "9"),
]


def main():
    OUT.mkdir(exist_ok=True)
    inputs, annotations, previews = [], [], []
    for index, (kind, edit, after, override, question, unit, expected_before,
                expected_after) in enumerate(SPECS, 1):
        sample_id = f"{index:02d}_{kind}"
        before = {**after, **override}
        states = {"before": before, "after": after}
        images = {name: render(kind, state) for name, state in states.items()}
        bbox = ImageChops.difference(images["before"], images["after"]).getbbox()
        assert bbox, f"No visible edit in {sample_id}"
        record = {"id": sample_id, "base_problem_id": sample_id, "category": kind,
                  "edit_type": edit, "unit": unit, "changed_pixel_bbox": bbox,
                  "rendering_spec": states, "answers": {}, "image_sha256": {},
                  "review_status": "pending_user_review", "source": "programmatic_original"}
        item = {"id": sample_id, "text_shard_1": question +
                " Explain your reasoning and put your final answer inside \\boxed{}.",
                "text_shard_2": NOTICE}
        for stage, state in states.items():
            path = OUT / f"{sample_id}_{stage}.png"
            images[stage].save(path)
            item[f"image_{stage}"] = path.name
            answer, facts, solution = solve(kind, state)
            assert answer == (expected_before if stage == "before" else expected_after)
            record["answers"][stage] = {"answer": answer, "facts": facts, "solution": solution}
            record["image_sha256"][stage] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["answers"]["before"]["answer"] != record["answers"]["after"]["answer"]
        inputs.append(item)
        annotations.append(record)
        previews.append((sample_id, images))
    for name, records in (("inputs.jsonl", inputs), ("annotations.jsonl", annotations)):
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    sheet = Image.new("RGB", (960, len(previews) * 360), "#e8ecf1")
    draw = ImageDraw.Draw(sheet)
    for row, (sample_id, images) in enumerate(previews):
        for column, stage in enumerate(("before", "after")):
            label(draw, (column * 480 + 240, row * 360 + 20), f"{sample_id}: {stage}", 20)
            sheet.paste(images[stage].resize((480, 320)), (column * 480, row * 360 + 40))
    sheet.save(OUT / "contact_sheet.png")
    print(f"Generated and checked {len(inputs)} pairs in {OUT}")


if __name__ == "__main__":
    main()
