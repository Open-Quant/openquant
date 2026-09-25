"""Figure helper for the runbooks: draw once per theme, show one, export both.

``figure(name, draw)`` calls ``draw(fig, colors)`` twice, once per docs-site
theme, and writes ``<name>-light.svg`` and ``<name>-dark.svg`` to
``docs-site/public/figures/notebooks/`` (or ``$OPENQUANT_FIGURE_DIR``). A docs
page shows the variant that matches Starlight's theme toggle, the same
convention as ``docs-site/scripts/figures/_svg.py``::

    <img class="dark:sl-hidden" src="/figures/notebooks/nb05-equity-drawdown-light.svg" alt="..." />
    <img class="light:sl-hidden" src="/figures/notebooks/nb05-equity-drawdown-dark.svg" alt="..." />

The light variant is also displayed inline, so the executed notebook carries
the plot as ``image/png``. The colours are the identity tokens in
``docs/design/identity/tokens.md``. SVGs are written without a date, creator or
random ids, so an unchanged figure is byte-identical after a re-run.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from IPython.display import Image, display  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(
    os.environ.get("OPENQUANT_FIGURE_DIR")
    or REPO_ROOT / "docs-site" / "public" / "figures" / "notebooks"
)

# docs/design/identity/tokens.md
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "ground": "#fbf9f4",
        "surface": "#f2eee4",
        "text": "#1d1a16",
        "muted": "#5c554b",
        "accent": "#97196a",
        "rule": "#cdc4b1",
    },
    "dark": {
        "ground": "#17140f",
        "surface": "#221e17",
        "text": "#ece6d8",
        "muted": "#aaa190",
        "accent": "#f29ad6",
        "rule": "#4c4536",
    },
}

SITE_FONT = "'IBM Plex Sans', system-ui, sans-serif"
INLINE_DPI = 90
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


# Axis formatter for fractions shown as percentages (drawdowns, returns).
percent = PercentFormatter(xmax=1.0, decimals=None)


def two_tone(colors: dict[str, str]) -> LinearSegmentedColormap:
    """Sequential map from the page ground to the accent, for matrices and counts."""
    return LinearSegmentedColormap.from_list("oq-seq", [colors["ground"], colors["accent"]])


def diverging(colors: dict[str, str]) -> LinearSegmentedColormap:
    """Diverging map: muted (negative), ground (zero), accent (positive)."""
    return LinearSegmentedColormap.from_list(
        "oq-div", [colors["muted"], colors["ground"], colors["accent"]]
    )


def _rc(colors: dict[str, str]) -> dict[str, Any]:
    return {
        "figure.facecolor": colors["ground"],
        "axes.facecolor": colors["ground"],
        "savefig.facecolor": colors["ground"],
        "axes.edgecolor": colors["rule"],
        "axes.labelcolor": colors["muted"],
        "axes.titlecolor": colors["text"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": colors["rule"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "xtick.color": colors["muted"],
        "ytick.color": colors["muted"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": colors["text"],
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": colors["muted"],
        "font.family": "DejaVu Sans",
        "lines.linewidth": 1.3,
        "svg.fonttype": "none",
        "svg.hashsalt": "openquant-notebooks",
        "path.simplify": True,
    }


def _clean_svg(text: str) -> str:
    # Text is kept as text (svg.fonttype none); render it in the site's face.
    return text.replace("'DejaVu Sans'", SITE_FONT).replace('"DejaVu Sans"', SITE_FONT)


def figure(
    name: str,
    draw: Callable[[Any, dict[str, str]], None],
    *,
    size: tuple[float, float] = (7.2, 3.2),
    alt: str | None = None,
) -> None:
    """Draw ``draw(fig, colors)`` in both themes, export SVGs, display the light one.

    ``name`` is the file stem (lower-case, digits, hyphens), conventionally
    prefixed with the notebook number, e.g. ``nb05-equity-drawdown``. ``alt``
    becomes the SVG title. Returns nothing, so a trailing call adds no
    machine-specific path to the notebook's outputs.
    """
    if not _NAME.match(name):
        raise ValueError(f"figure name must be lower-case letters, digits and hyphens: {name!r}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for theme, colors in THEMES.items():
        with plt.rc_context(_rc(colors)):
            fig = plt.figure(figsize=size, layout="constrained")
            draw(fig, colors)
            path = FIGURE_DIR / f"{name}-{theme}.svg"
            fig.savefig(
                path,
                format="svg",
                metadata={"Date": None, "Creator": None, "Title": alt or name},
            )
            path.write_text(_clean_svg(path.read_text(encoding="utf-8")), encoding="utf-8")
            if theme == "light":
                png = io.BytesIO()
                fig.savefig(png, format="png", dpi=INLINE_DPI, metadata={"Software": None})
                display(Image(data=png.getvalue(), format="png"))
            plt.close(fig)
