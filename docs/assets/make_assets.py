"""Draw vardax's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the variational picture: over one assimilation window, the
analysis trajectory (solid) is pulled from the background (dashed) towards the
observations and their error bars, all at once. filterax draws the sequential
update (an ensemble pinching at each observation) and argminax the minimisation
itself (a descent to x̂), so the window is vardax's. Standard library plus
loguru; the tile and wordmark follow the other repos of the stack (radtrax,
gaussx, kernellib).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

import math
from pathlib import Path

from loguru import logger

HERE = Path(__file__).resolve().parent

NAME = "vardax"
WORDMARK = ("var", "dax")
LOGO_WIDTH = 460
# Rose: no other repo of the stack uses it.
GRADIENT = ("#FB7185", "#BE123C")
THEMES = {
    "light": {"ink": "#1F2937", "accent": "#E11D48"},
    "dark": {"ink": "#E5E7EB", "accent": "#FB7185"},
}
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
WHITE = "#FFFFFF"


def svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{NAME}">\n'
        f"<title>{NAME}</title>\n{body}\n</svg>\n"
    )


def tile(gradient_id: str) -> str:
    start, stop = GRADIENT
    return (
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{stop}"/></linearGradient></defs>'
        '<rect x="4" y="4" width="120" height="120" rx="26" '
        f'fill="url(#{gradient_id})"/>'
    )


def polyline(points: list[tuple[float, float]], *, width: float) -> str:
    commands = [
        f"{'M' if index == 0 else 'L'}{x:.1f} {y:.1f}"
        for index, (x, y) in enumerate(points)
    ]
    return (
        f'<path d="{" ".join(commands)}" fill="none" stroke="{WHITE}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def dot(x: float, y: float, radius: float) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{WHITE}"/>'


# (time, value, error half-width) of each observation in the window
OBSERVATIONS = (
    (32.0, 64.0, 9.0),
    (54.0, 52.0, 8.0),
    (76.0, 70.0, 9.0),
    (98.0, 56.0, 8.0),
)
# where the analysis passes: near each observation, not through it
ANALYSIS_KNOTS = (
    (22.0, 72.0),
    (32.0, 66.0),
    (54.0, 53.0),
    (76.0, 67.0),
    (98.0, 57.0),
    (106.0, 55.0),
)
WINDOW = (22.0, 106.0)


def background(time: float) -> float:
    """The first guess: a smooth, gently varying trajectory below the analysis."""
    start, stop = WINDOW
    # x_b(t) = 92 − 6 sin(π (t − t₀)/(t₁ − t₀))
    return 92 - 6 * math.sin(math.pi * (time - start) / (stop - start))


def catmull_rom(
    knots: tuple[tuple[float, float], ...], samples: int = 16
) -> list[tuple[float, float]]:
    """A smooth curve through the knots, with the end knots repeated."""
    padded = [knots[0], *knots, knots[-1]]
    points = []
    for index in range(1, len(padded) - 2):
        before, start, stop, after = padded[index - 1 : index + 3]
        for step in range(samples):
            s = step / samples
            point = []
            for axis in range(2):
                # the uniform Catmull-Rom spline between start and stop
                a = 2 * start[axis]
                b = (stop[axis] - before[axis]) * s
                c = (2 * before[axis] - 5 * start[axis] + 4 * stop[axis] - after[axis]) * s * s
                d = (3 * start[axis] - before[axis] - 3 * stop[axis] + after[axis]) * s**3
                point.append(0.5 * (a + b + c + d))
            points.append((point[0], point[1]))
    points.append(knots[-1])
    return points


def glyph() -> str:
    """The background, the observations with their errors, and the analysis through them."""
    start, stop = WINDOW
    times = [start + (stop - start) * step / 48 for step in range(49)]
    first_guess = " ".join(
        f"{'M' if index == 0 else 'L'}{time:.1f} {background(time):.1f}"
        for index, time in enumerate(times)
    )
    body = (
        f'<path d="{first_guess}" fill="none" stroke="{WHITE}" stroke-width="2.6" '
        'stroke-opacity="0.6" stroke-dasharray="4 5" stroke-linecap="round"/>'
    )
    for time, value, error in OBSERVATIONS:
        body += (
            f'<line x1="{time}" y1="{value - error}" x2="{time}" y2="{value + error}" '
            f'stroke="{WHITE}" stroke-width="2.2" stroke-opacity="0.75" '
            'stroke-linecap="round"/>'
        )
    body += polyline(catmull_rom(ANALYSIS_KNOTS), width=4.6)
    for time, value, _ in OBSERVATIONS:
        body += (
            f'<circle cx="{time}" cy="{value}" r="4.6" fill="none" '
            f'stroke="{WHITE}" stroke-width="2.4"/>'
        )
    return body


def icon() -> str:
    return svg(128, 128, tile(f"{NAME}-icon") + glyph())


def logo(theme: str) -> str:
    colours = THEMES[theme]
    head, tail = WORDMARK
    mark = f'<g transform="translate(6 6)">{tile(f"{NAME}-{theme}")}{glyph()}</g>'
    wordmark = (
        f'<text x="154" y="92" font-family="{SANS}" font-size="72" font-weight="700" '
        f'letter-spacing="-1.5" fill="{colours["ink"]}">{head}'
        f'<tspan fill="{colours["accent"]}">{tail}</tspan></text>'
    )
    return svg(LOGO_WIDTH, 140, mark + wordmark)


def main() -> None:
    outputs = {"icon.svg": icon()}
    for theme in THEMES:
        outputs[f"logo-{theme}.svg"] = logo(theme)
    for filename, content in outputs.items():
        (HERE / filename).write_text(content)
        logger.info("wrote {}", HERE / filename)


if __name__ == "__main__":
    main()
