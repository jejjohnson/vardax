"""Draw vardax's icon and logo: ``icon.svg`` and ``logo-{light,dark}.svg``.

The icon is the variational picture: level sets of the cost J and a descent
path ending at its minimiser x*. Standard library plus loguru; the tile and
wordmark follow the other repos of the stack (radtrax, gaussx, kernellib).
Run from the repository root::

    uv run --no-project --with loguru python docs/assets/make_assets.py
"""

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
SERIF = "'Times New Roman', Georgia, serif"
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


def glyph() -> str:
    """Level sets of J, rotated like a correlated B, and the descent to x*."""
    centre_x, centre_y = 74, 60
    rings = [(50, 34, 0.25), (37, 25, 0.38), (24, 16, 0.55), (11, 7.5, 0.8)]
    body = ""
    for radius_x, radius_y, opacity in rings:
        body += (
            f'<ellipse cx="{centre_x}" cy="{centre_y}" rx="{radius_x}" ry="{radius_y}" '
            f'fill="none" stroke="{WHITE}" stroke-width="2.4" stroke-opacity="{opacity}" '
            f'transform="rotate(-24 {centre_x} {centre_y})"/>'
        )
    descent = [(22, 100), (40, 90), (54, 74), (centre_x, centre_y)]
    body += polyline(descent, width=3.5)
    for x, y in descent[:-1]:
        body += dot(x, y, radius=3.6)
    body += dot(centre_x, centre_y, radius=7)
    body += (
        f'<text x="104" y="110" font-family="{SERIF}" font-size="20" font-style="italic" '
        f'font-weight="700" fill="{WHITE}" text-anchor="middle">x*</text>'
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
