"""Generate the small animated ASMC preview used by the repository README.

The deployed Pages application is the authoritative interactive explainer.  This
script produces a deterministic, dependency-light storyboard so the same idea is
visible from GitHub's Markdown renderer, which cannot execute the React page.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 760
HEIGHT = 420
FRAME_COUNT = 24
FRAME_DURATION_MS = 150

BG = "#fcfbf8"
INK = "#111827"
MUTED = "#64748b"
LINE = "#d8dee8"
BLUE = "#1f77b4"
PURPLE = "#6a3d9a"
GREEN = "#2ca02c"
ORANGE = "#d97706"
RED = "#c43c39"
SLATE = "#7f8a99"


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = _font(25, bold=True)
FONT_SUBTITLE = _font(12)
FONT_STAGE = _font(11, bold=True)
FONT_BODY = _font(12)
FONT_SMALL = _font(10)


def _centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    textbbox = getattr(draw, "textbbox", None)
    if textbbox is not None:
        box = textbbox((0, 0), text, font=font)
        width, height = box[2] - box[0], box[3] - box[1]
    else:
        width, height = draw.textsize(text, font=font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: str, width: int = 2) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 8
    left = (end[0] - length * math.cos(angle - math.pi / 6), end[1] - length * math.sin(angle - math.pi / 6))
    right = (end[0] - length * math.cos(angle + math.pi / 6), end[1] - length * math.sin(angle + math.pi / 6))
    draw.polygon((end, left, right), fill=fill)


def _rounded(draw: ImageDraw.ImageDraw, box, radius: int = 8, **kwargs) -> None:
    """Use rounded corners when supported, with an old-Pillow fallback."""
    rounded = getattr(draw, "rounded_rectangle", None)
    if rounded is not None:
        rounded(box, radius=radius, **kwargs)
    else:
        draw.rectangle(box, **kwargs)


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, accent: str) -> None:
    _rounded(draw, box, radius=12, fill="#ffffff", outline=LINE, width=1)
    _rounded(draw, (box[0], box[1], box[2], box[1] + 7), radius=12, fill=accent)
    draw.text((box[0] + 16, box[1] + 17), title, font=FONT_STAGE, fill=INK)


def _frame(frame_index: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((30, 22), "Cache-Coherent ASMC", font=FONT_TITLE, fill=INK)
    draw.text(
        (31, 56),
        "parallel particles  ->  ESS check  ->  ancestor map  ->  KV-cache gather",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )

    progress = frame_index / (FRAME_COUNT - 1)
    stage = min(5, int(progress * 6))
    stage_names = ["PROMPT", "DECODE", "WEIGHT", "RESAMPLE", "KV REORDER", "VOTE"]
    stage_x = [71, 193, 315, 437, 559, 681]
    for index, (x, name) in enumerate(zip(stage_x, stage_names)):
        active = index == stage
        completed = index < stage
        fill = BLUE if active else ("#e6eef7" if completed else "#f1f5f9")
        outline = BLUE if active else LINE
        _rounded(draw, (x - 49, 83, x + 49, 111), radius=8, fill=fill, outline=outline, width=2 if active else 1)
        text_color = "#ffffff" if active else (INK if completed else MUTED)
        _centered(draw, (x, 97), name, FONT_SMALL if index != 4 else _font(9, bold=True), text_color)
        if index < len(stage_x) - 1:
            _arrow(draw, (x + 51, 97), (stage_x[index + 1] - 53, 97), BLUE if completed else LINE, 2)

    _panel(draw, (28, 132, 371, 358), "PARTICLE POPULATION", BLUE)
    _panel(draw, (389, 132, 732, 358), "SHARED KV CACHE", PURPLE)

    # The particle traces move together during batched decoding.  At the
    # resampling boundary several traces inherit one high-weight ancestor.
    particle_colors = [BLUE, SLATE, PURPLE, GREEN, ORANGE, RED]
    ancestor = [2, 0, 2, 5, 2, 4]
    wave = min(1.0, max(0.0, progress * 6 - 1.0))
    for row in range(6):
        y = 181 + row * 25
        draw.text((46, y - 7), f"P{row + 1}", font=FONT_SMALL, fill=MUTED)
        # A small sinusoidal motion makes the decode stage visibly dynamic.
        x = 102 + 196 * wave + 5 * math.sin(frame_index * 0.55 + row)
        color = particle_colors[ancestor[row]] if stage >= 3 else particle_colors[row]
        if stage >= 3 and row != ancestor[row]:
            _arrow(draw, (112, y), (x - 8, y), "#cbd5e1", 1)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="#ffffff", width=2)
        draw.text((x + 14, y - 7), "tok", font=FONT_SMALL, fill=MUTED)
        weight = [0.29, 0.07, 0.41, 0.10, 0.08, 0.05][row]
        if stage >= 3:
            weight = [0.41, 0.29, 0.41, 0.05, 0.41, 0.08][row]
        _rounded(draw, (205, y - 5, 341, y + 5), radius=4, fill="#eef2f7")
        _rounded(draw, (205, y - 5, 205 + 136 * weight / 0.45, y + 5), radius=4, fill=color)

    draw.text((47, 333), "ESS / N", font=FONT_SMALL, fill=MUTED)
    ess = 0.86 - 0.45 * min(1.0, max(0.0, progress * 6 - 1.35))
    if stage >= 3:
        ess = 0.74
    draw.text((126, 330), f"{ess:.2f}", font=FONT_BODY, fill=ORANGE if ess < 0.5 else GREEN)
    draw.text((205, 333), "tau = 0.50", font=FONT_SMALL, fill=MUTED)
    status = ["initialize", "batched decode", "reweight + ESS", "ancestor sample", "gather cache", "weighted vote"][stage]
    draw.text((47, 337 + 16), status, font=FONT_SMALL, fill=INK)

    # Cache slots are deliberately drawn as rows: resampling changes the
    # ancestor order, while the token prefix itself is never replayed.
    cache_x = 430
    slot_w = 42
    cache_y = 181
    for row in range(6):
        y = cache_y + row * 25
        draw.text((406, y - 7), f"K{row + 1}", font=FONT_SMALL, fill=MUTED)
        source_row = ancestor[row] if stage >= 4 else row
        for slot in range(6):
            x = cache_x + slot * (slot_w + 3)
            slot_color = particle_colors[source_row] if slot <= 3 else "#e2e8f0"
            _rounded(draw, (x, y - 8, x + slot_w, y + 8), radius=4, fill=slot_color, outline="#ffffff", width=1)
            _centered(draw, (x + slot_w / 2, y), f"{slot + 1}", FONT_SMALL, "#ffffff" if slot <= 3 else MUTED)
        if stage >= 4 and source_row != row:
            draw.text((704, y - 7), "^", font=FONT_BODY, fill=PURPLE)

    cache_note = "reorder KV slices; do not replay the prefix" if stage >= 4 else "one cache row per particle"
    draw.text((406, 333), cache_note, font=FONT_SMALL, fill=PURPLE if stage >= 4 else MUTED)
    _rounded(draw, (406, 342, 716, 349), radius=4, fill="#ede9fe")
    used = 0.18 + 0.13 * stage
    _rounded(draw, (406, 342, 406 + 310 * min(1.0, used), 349), radius=4, fill=PURPLE)
    draw.text((406, 362), f"C_int budget used  {used:.2f} C*", font=FONT_SMALL, fill=INK)

    draw.text((30, 389), "ASMC demo storyboard - click the image to open the full interactive explainer", font=FONT_SMALL, fill=MUTED)
    return image


def main() -> None:
    output = Path(__file__).with_name("asmc_kv_cache.gif")
    # Keep the README asset small enough for GitHub while retaining readable
    # labels.  A 64-colour palette is more than sufficient for this flat UI.
    resample = getattr(Image, "Resampling", Image).LANCZOS
    frames = [
        _frame(index).resize((600, 332), resample=resample).quantize(colors=64, method=Image.MEDIANCUT)
        for index in range(FRAME_COUNT)
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
