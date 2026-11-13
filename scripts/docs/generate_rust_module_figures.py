#!/usr/bin/env python3
"""Generate four explanatory Rust-module SVGs, using only Python's standard library.

Run from any directory:
    python3 scripts/docs/generate_rust_module_figures.py --write
    python3 scripts/docs/generate_rust_module_figures.py --check
    python3 -m unittest discover -s scripts/docs/tests -v

Sources: the equations, inputs and existing assertions in the four corresponding
module pages, checked against crates/openquant/src/{combinatorial_optimization,
etf_trick,fingerprint,hpc_parallel}.rs. These are independent explanatory
calculations, NOT captured Rust output or measurements. Inventory paths are
exhaustively scored using exact rational arithmetic, rather than Rust's trade DFS.
Roll adjustments use AFML Snippet 2.2. Fingerprint curves are population formulas
for independent uniform [-1, 1] features, NOT the page's finite-sample estimates.
HPC work uses the discrete arithmetic-series sum, not runtime benchmarks.

No external assets, imports of openquant, timestamps or random inputs are used.
The fixed light palette follows docs-site/src/styles/starlight.css (Monograph).
"""
import argparse
from fractions import Fraction
from html import escape
from itertools import product
from math import floor, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'docs-site/public/figures/modules'
INK, MUTED, ACCENT, RULE, PAPER = '#1d1a16', '#5c554b', '#97196a', '#cdc4b1', '#fbf9f4'


def inventory_optimum(ticket_cost):
    """Enumerate four intermediate holdings; initial and terminal holdings are zero."""
    returns = [Fraction(x, 100) for x in (2, 2, -1, 2, 0)]
    best, best_score, count = None, None, 0
    for middle in product(range(4), repeat=4):
        path = (0, *middle, 0)
        trades = [b - a for a, b in zip(path, path[1:])]
        score = sum(q * r - Fraction(q*q, 1000) - Fraction(abs(d), 500)
                    - Fraction(str(ticket_cost)) * (d != 0)
                    for q, r, d in zip(path[1:], returns, trades))
        count += 1
        if best_score is None or score > best_score:
            best, best_score = path, score
    return best, float(best_score), count


def roll_example():
    """The page's already sorted front chain, absolute backward adjustment."""
    raw = [70.4, 71.2, 71.1, 72.9, 73.0]
    gap = float(Fraction('72.0') - Fraction('71.1'))
    return raw, [p + gap if i < 3 else p for i, p in enumerate(raw)], gap


def partition_example():
    """N=1000, M=4: published boundary formulas and exact sum of row indices."""
    linear = [1000 * i // 4 for i in range(5)]
    nested = [floor(1000 * sqrt(i / 4) + .5) for i in range(5)]
    def with_work(bounds):
        return bounds, [(b * (b - 1) - a * (a - 1)) // 2
                        for a, b in zip(bounds, bounds[1:])]
    return with_work(linear), with_work(nested)


def ideal_effects(v, w):
    """Centered linear curve, quadratic curve, and joint interaction residual."""
    return 2 * v, v * v, v * w


class Svg:
    def __init__(self, title, description, height):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 {height}" '
            f'width="600" height="{height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            f'<rect width="600" height="{height}" fill="{PAPER}"/>',
            f'<g font-family="system-ui, sans-serif" font-size="20" fill="{INK}">',
        ]
        self.text(24, 36, title, size=24, weight='bold')

    def text(self, x, y, value, size=20, color=INK, anchor='start', weight='normal'):
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{color}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color=RULE, width=1, dash=''):
        self.parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
                          f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def curve(self, points, color=ACCENT, dash=''):
        coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        self.parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                          f'stroke-width="3" stroke-linejoin="round" stroke-dasharray="{dash}"/>')

    def rect(self, x, y, width, height, fill, stroke=PAPER):
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
                          f'fill="{fill}" stroke="{stroke}"/>')

    def finish(self):
        return '\n'.join([*self.parts, '</g>', '</svg>', ''])


def inventory_figure():
    s = Svg('Ticket costs change the inventory path',
            'Independent exhaustive reference over 256 feasible paths. Costs 0 and 0.01 '
            'avoid the dip; 0.03 holds through it; 0.10 stays flat. Initial and terminal '
            'inventory are zero. Each panel shows inventory after each trade.', 700)
    s.text(24, 67, 'Inventory units · same 256 feasible paths', color=MUTED)
    for row, (cost, label) in enumerate([(0, 'Avoid the dip'), (.03, 'Hold through the dip'), (.1, 'Do not trade')]):
        path, score, _ = inventory_optimum(cost)
        top = 105 + row * 185
        s.text(24, top, f'Ticket {cost:g} · {label}', weight='bold')
        s.text(24, top + 27, f'Net objective {score:.3f}', color=MUTED)
        for q in (0, 3):
            y = top + 118 - q * 23
            s.line(65, y, 555, y)
            s.text(50, y + 7, q, anchor='end')
        points = [(65, top + 118)]
        for i in range(1, len(path)):
            x = 65 + i * 98
            points.extend([(x, top + 118 - path[i-1] * 23), (x, top + 118 - path[i] * 23)])
        s.curve(points)
        for i in range(6):
            s.text(65 + i * 98, top + 147, i, anchor='middle')
    s.text(300, 661, 'Step (0 = initial inventory)', anchor='middle')
    s.text(24, 690, 'At ticket 0.01: same dip-avoiding path, score 0.089.', size=18, color=MUTED)
    return s.finish()


def roll_figure():
    raw, adjusted, gap = roll_example()
    s = Svg('Remove the roll gap, retain the move',
            'Raw and absolute backward-adjusted closes for January 2, 3, 4, 5 and 8, 2024. '
            'At the January 5 roll, the new open 72.00 exceeds the prior close 71.10 by 0.90. '
            'Earlier prices shift up by 0.90; the last two closes remain unchanged.', 560)
    s.text(24, 68, 'Absolute backward roll · price units', color=MUTED)
    s.line(25, 96, 65, 96, INK, 3, '8 5')
    s.text(75, 103, 'Raw close')
    s.line(285, 96, 325, 96, ACCENT, 3)
    s.text(335, 103, 'Adjusted close')
    x = lambda i: 75 + 118 * i
    y = lambda p: 375 - (p - 70) * 70
    for value in (70, 71, 72, 73):
        s.line(75, y(value), 547, y(value))
        s.text(58, y(value) + 7, value, anchor='end')
    s.line(x(3), 133, x(3), 375, MUTED, 1, '4 5')
    s.text(x(3), 132, 'Roll', anchor='middle')
    s.curve([(x(i), y(p)) for i, p in enumerate(raw)], INK, '8 5')
    s.curve([(x(i), y(p)) for i, p in enumerate(adjusted)])
    for i, day in enumerate(('02', '03', '04', '05', '08')):
        s.text(x(i), 406, day, anchor='middle')
    s.text(300, 439, 'January 2024 · sessions equally spaced', anchor='middle', size=18)
    s.text(24, 480, f'Roll gap: 72.00 open − 71.10 close = {gap:.2f}')
    s.text(24, 512, 'Raw close change: 1.80 = 0.90 gap + 0.90 move')
    s.text(24, 544, 'Adjusted close change: 0.90; history shifts up.', color=MUTED)
    return s.finish()


def fingerprint_figure():
    s = Svg('Three ways a model can use a feature',
            'Idealized analytic curves for independent uniform features on [-1,1], '
            'not the finite-sample Rust example. Linear centered dependence is 2v. '
            'Quadratic dependence is v squared with a constant fitted line at one third. '
            'The joint residual vw changes slope with w, although its average over w is zero.', 1015)
    s.text(24, 68, 'Idealized model: 2x₀ + x₁² + x₀x₂', color=MUTED)
    s.text(24, 98, 'Independent uniform features on [−1, 1].', color=MUTED)
    for row, (title, low, high, ticks) in enumerate([
        ('Linear: centered dependence = 2v', -2, 2, (-2, 0, 2)),
        ('Non-linear: dependence = v²', 0, 1, (0, 1)),
        ('Pairwise: joint residual = vw', -1, 1, (-1, 0, 1)),
    ]):
        top = 140 + row * 280
        s.text(24, top, title, weight='bold')
        x = lambda v: 80 + (v + 1) * 185
        y = lambda z: top + 175 - (z - low) / (high - low) * 130
        for tick in ticks:
            s.line(80, y(tick), 450, y(tick))
            s.text(65, y(tick) + 7, tick, anchor='end')
        for v in (-1, 0, 1):
            s.text(x(v), top + 203, v, anchor='middle')
        values = [i / 40 for i in range(-40, 41)]
        if row == 0:
            s.curve([(x(v), y(ideal_effects(v, 0)[0])) for v in values])
            s.text(24, top + 235, 'The fitted line explains the whole curve.', size=18, color=MUTED)
        elif row == 1:
            s.curve([(x(v), y(ideal_effects(v, 0)[1])) for v in values])
            s.line(80, y(1/3), 450, y(1/3), INK, 2, '7 5')
            s.text(465, y(1/3) + 7, '⅓')
            s.text(24, top + 235, 'Dashed: population least-squares line at ⅓.', size=18, color=MUTED)
        else:
            for w, color, dash in ((-1, INK, '8 5'), (0, MUTED, '2 5'), (1, ACCENT, '')):
                s.curve([(x(v), y(ideal_effects(v, w)[2])) for v in values], color, dash)
                s.text(465, y(w) + 7, f'w={w}', size=18, color=color)
            s.text(24, top + 235, 'Averaging over w hides the interaction.', size=18, color=MUTED)
    s.text(24, 1002, 'Horizontal: feature value v · vertical: prediction units', size=18, color=MUTED)
    return s.finish()


def hpc_figure():
    s = Svg('Equal atoms are not equal work',
            '1000 atoms with atom k costing k units, split into four contiguous molecules. '
            'Linear endpoints are 250, 500, 750, 1000. Nested endpoints are 500, 707, 866, '
            '1000. Discrete linear work rises from 31125 to 218625; nested work is about '
            '124875 per molecule. Bars show work counts, not measured execution times.', 850)
    s.text(24, 68, '1,000 atoms · 4 molecules · atom k costs k', color=MUTED)
    for row, (name, (bounds, work)) in enumerate(zip(('Linear', 'Nested'), partition_example())):
        top = 110 + row * 355
        s.text(24, top, name, weight='bold')
        for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
            s.rect(65 + a * .49, top + 25, (b-a)*.49, 40, ACCENT if i % 2 == 0 else INK)
            s.text(65 + (a+b)*.245, top + 53, f'M{i+1}', color=PAPER, anchor='middle', size=18)
        for b in bounds:
            s.text(65 + b * .49, top + 89, b, anchor='middle', size=18)
        s.text(24, top + 123, 'Work per molecule (same scale in both panels)', size=18, color=MUTED)
        for i, value in enumerate(work):
            y = top + 141 + 42 * i
            s.text(24, y + 23, f'M{i+1}', size=18)
            width = value / 220000 * 345
            s.rect(70, y, width, 30, ACCENT)
            s.text(80 + width, y + 23, f'{value:,}', size=18)
    s.text(24, 807, 'Work = sum of k for start ≤ k < end.', color=MUTED)
    s.text(24, 837, 'Discrete work counts, not a speedup benchmark.', size=18, color=MUTED)
    return s.finish()


def render_all():
    return {
        'combinatorial-optimization.svg': inventory_figure(),
        'etf-trick.svg': roll_figure(),
        'fingerprint.svg': fingerprint_figure(),
        'hpc-parallel.svg': hpc_figure(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stale = []
    for name, content in render_all().items():
        path = args.output_dir / name
        expected = content.encode('utf-8')
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
            print(f'Wrote {path}')
        elif not path.exists() or path.read_bytes() != expected:
            stale.append(str(path))
    if stale:
        print('Missing or stale figures; regenerate with --write:\n' + '\n'.join(stale))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
