"""Tiny SVG chart helpers for the docs figures. No dependencies.

Figures are drawn twice, once per theme, in the identity tokens
(docs/design/identity/tokens.md) and written to docs-site/public/figures/. A page shows the one
that matches Starlight's theme toggle with the `light:sl-hidden` / `dark:sl-hidden` classes.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "public" / "figures"

THEMES = {
    "light": {"ground": "#fbf9f4", "text": "#1d1a16", "muted": "#5c554b", "accent": "#97196a", "rule": "#cdc4b1"},
    "dark": {"ground": "#17140f", "text": "#ece6d8", "muted": "#aaa190", "accent": "#f29ad6", "rule": "#4c4536"},
}

SANS = "'IBM Plex Sans', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"


class Chart:
    """A plot area with linear scales. Coordinates are rounded so reruns are byte-identical."""

    def __init__(self, width: int, height: int, theme: dict[str, str], title: str):
        self.w, self.h, self.c = width, height, theme
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
            f'<rect width="{width}" height="{height}" fill="{theme["ground"]}"/>',
        ]

    @staticmethod
    def scale(lo: float, hi: float, a: float, b: float):
        return lambda v: round(a + (v - lo) / (hi - lo) * (b - a), 1)

    def line(self, pts, color="text", width=1.2):
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="{self.c[color]}" stroke-width="{width}" stroke-linejoin="round"/>')

    def rule(self, x1, y1, x2, y2, color="rule", width=1.0):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{self.c[color]}" stroke-width="{width}"/>')

    def band(self, x1, x2, y1, y2, color="rule", opacity=0.35):
        self.parts.append(f'<rect x="{x1}" y="{y1}" width="{round(x2 - x1, 1)}" height="{round(y2 - y1, 1)}" fill="{self.c[color]}" fill-opacity="{opacity}"/>')

    def dot(self, x, y, r=3.2, color="accent"):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{self.c[color]}"/>')

    def text(self, x, y, s, color="muted", size=12, anchor="start", family=SANS, weight=400):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{self.c[color]}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>'
        )

    def label(self, x, y, s, color="muted"):
        """Small-caps label, as used for table heads on the site."""
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{self.c[color]}" font-family="{SANS}" font-size="10.5" '
            f'font-weight="600" letter-spacing="1.2">{s.upper()}</text>'
        )

    def save(self, name: str, theme_name: str) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{name}-{theme_name}.svg"
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")
        return path
