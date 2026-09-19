"""Build the three identity candidates and their contrast report.

Run from the repository root:  python3 docs/design/identity/build_candidates.py
Every candidate renders the same content fixture; only tokens, type and
treatment differ. Contrast is computed from the tokens, never eyeballed.
"""
from pathlib import Path

OUT = Path(__file__).parent

# The pinwheel's three hues are the organisation's identity and are kept as is.
LOGO = {"magenta": "#e44de0", "cyan": "#52c8f5", "yellow": "#f0e050"}

CANDIDATES = {
    "a-monograph": {
        "name": "A · Monograph",
        "pitch": "The book this library implements, set properly: a serif reading column on warm paper, "
                 "magenta ink taken from the logo, ruled tables.",
        "first": "light",
        "accent_from": "magenta",
        "fonts": "family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400"
                 "&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;600",
        "light": dict(ground="#fbf9f4", surface="#f2eee4", text="#1d1a16", muted="#5c554b",
                      accent="#97196a", rule="#cdc4b1"),
        "dark": dict(ground="#17140f", surface="#221e17", text="#ece6d8", muted="#aaa190",
                     accent="#f29ad6", rule="#4c4536"),
        "css": """
:root{--body:'Newsreader',Georgia,serif;--ui:'IBM Plex Sans',sans-serif;--mono:'IBM Plex Mono',monospace;
  --body-size:1.19rem;--leading:1.62;--measure:52ch}
h1{font:600 3.1rem/1.08 var(--body);letter-spacing:-.015em}
h2{font:600 1.7rem/1.2 var(--body);border-top:1px solid var(--rule);padding-top:1.4rem}
.lede{font:italic 400 1.45rem/1.45 var(--body)}
.eyebrow{font:600 .74rem/1 var(--ui);letter-spacing:.14em;text-transform:uppercase}
th{font:600 .74rem/1.2 var(--ui);letter-spacing:.1em;text-transform:uppercase}
table{border-top:2px solid var(--text);border-bottom:2px solid var(--text)}
td,th{border-right:1px solid var(--rule)} td:last-child,th:last-child{border-right:0}
tbody tr:last-child td{border-bottom:0}
.formula{font:italic 400 1.3rem/1.5 var(--body)}
""",
    },
    "b-ledger": {
        "name": "B · Ledger",
        "pitch": "A trading terminal's ledger: dark first, monospaced headings and numerals, amber taken from "
                 "the logo, every surface divided by hairlines instead of cards.",
        "first": "dark",
        "accent_from": "yellow",
        "fonts": "family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500;600",
        "light": dict(ground="#f7f8f5", surface="#ebeee8", text="#131a16", muted="#536059",
                      accent="#735a00", rule="#c1c9c3"),
        "dark": dict(ground="#0b0e0d", surface="#131917", text="#dfe6e1", muted="#90a097",
                     accent="#e9c84a", rule="#2c3632"),
        "css": """
:root{--body:'IBM Plex Sans',sans-serif;--ui:'IBM Plex Mono',monospace;--mono:'IBM Plex Mono',monospace;
  --body-size:1.02rem;--leading:1.68;--measure:53ch}
h1{font:500 2.5rem/1.12 var(--mono);letter-spacing:-.03em}
h2{font:600 .95rem/1.3 var(--mono);letter-spacing:.12em;text-transform:uppercase;
  border-bottom:1px solid var(--rule);padding-bottom:.55rem}
h2::before{content:'// ';color:var(--accent)}
.lede{font:400 1.2rem/1.55 var(--body);color:var(--muted)}
.eyebrow{font:500 .74rem/1 var(--mono);letter-spacing:.1em;text-transform:uppercase}
th{font:500 .74rem/1.2 var(--mono);letter-spacing:.08em;text-transform:uppercase}
table{border:1px solid var(--rule)} td,th{border-right:1px solid var(--rule)} td:last-child,th:last-child{border-right:0}
.formula{font:400 1.05rem/1.5 var(--mono)}
header.site,aside.nav{background:var(--surface)}
""",
    },
    "c-blueprint": {
        "name": "C · Blueprint",
        "pitch": "An engineering drawing: condensed technical headings, a faint measured grid behind the title "
                 "block only, cyan-derived blue ink, light first.",
        "first": "light",
        "accent_from": "cyan",
        "fonts": "family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Sans+3:ital,wght@0,400;0,600;1,400"
                 "&family=IBM+Plex+Mono:wght@400;600",
        "light": dict(ground="#f6f9fb", surface="#e8eff4", text="#0e1f2b", muted="#4a6070",
                      accent="#065b88", rule="#b2c4d1"),
        "dark": dict(ground="#0a1822", surface="#11242f", text="#dbe9f2", muted="#8ea6b6",
                     accent="#62c6f1", rule="#2b4557"),
        "css": """
:root{--body:'Source Sans 3',sans-serif;--ui:'IBM Plex Sans Condensed',sans-serif;--mono:'IBM Plex Mono',monospace;
  --body-size:1.1rem;--leading:1.65;--measure:59ch}
h1{font:600 3rem/1.05 var(--ui);letter-spacing:-.005em}
h2{font:600 1.55rem/1.2 var(--ui);letter-spacing:.01em}
h2::after{content:'';display:block;width:2.5rem;height:3px;background:var(--accent);margin-top:.55rem}
.lede{font:400 1.3rem/1.5 var(--body)}
.eyebrow{font:600 .82rem/1 var(--ui);letter-spacing:.16em;text-transform:uppercase}
th{font:600 .82rem/1.2 var(--ui);letter-spacing:.1em;text-transform:uppercase}
table{border-top:1px solid var(--text)}
.formula{font:italic 400 1.2rem/1.5 var(--body)}
.titleblock{background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:24px 24px;border:1px solid var(--rule);padding:2rem 2rem 1.6rem;margin:0 -2rem 2rem}
""",
    },
}

BASE_CSS = """
*{box-sizing:border-box} html{-webkit-text-size-adjust:100%} :root[data-theme=dark]{color-scheme:dark} :root[data-theme=light]{color-scheme:light}
body{margin:0;background:var(--ground);color:var(--text);font:400 var(--body-size)/var(--leading) var(--body);
  font-feature-settings:'kern','liga'}
a{color:var(--accent);text-decoration:underline;text-underline-offset:.18em;text-decoration-thickness:1px}
header.site{display:flex;align-items:center;gap:.75rem;padding:.8rem 1.25rem;border-bottom:1px solid var(--rule);
  font:600 1.05rem/1 var(--ui)}
header.site svg{width:24px;height:24px;flex:none}
header.site .search{margin-left:1.5rem;flex:0 1 20rem;border:1px solid var(--rule);border-radius:3px;padding:.5rem .7rem;
  font:400 .85rem/1 var(--ui);color:var(--muted)}
header.site button{margin-left:auto;font:400 .8rem/1 var(--ui);color:var(--text);background:none;
  border:1px solid var(--rule);border-radius:3px;padding:.45rem .7rem;cursor:pointer}
.shell{display:grid;grid-template-columns:16rem minmax(0,1fr);min-height:100vh}
aside.nav{border-right:1px solid var(--rule);padding:1.5rem 1.25rem;font:400 .92rem/1.5 var(--ui)}
aside.nav h4{margin:1.4rem 0 .4rem;font:600 .72rem/1 var(--ui);letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
aside.nav h4:first-child{margin-top:0}
aside.nav a{display:block;padding:.22rem 0 .22rem .7rem;color:var(--text);text-decoration:none;border-left:2px solid transparent}
aside.nav a.on{border-left-color:var(--accent);color:var(--accent);font-weight:600}
main{padding:3rem 2rem 5rem;max-width:calc(var(--measure) + 4rem);margin:0 auto;width:100%}
.eyebrow{color:var(--accent);margin:0 0 1rem}
h1{margin:0 0 1rem} h2{margin:3rem 0 1rem} .lede{margin:0 0 1.5rem}
.pill{display:inline-block;font:600 .68rem/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  background:var(--accent);color:var(--ground);padding:.32rem .5rem;border-radius:2px;margin-right:.6rem;vertical-align:middle}
.status{font:400 .85rem/1.4 var(--ui);color:var(--muted);border-bottom:1px solid var(--rule);padding-bottom:1rem;margin-bottom:2rem}
code{font:400 .88em var(--mono);background:var(--surface);padding:.1em .3em;border-radius:2px}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:3px;padding:1rem 1.1rem;overflow-x:auto;
  font:400 .86rem/1.6 var(--mono);margin:1.25rem 0}
pre code{background:none;padding:0;font-size:inherit}
pre .c{color:var(--muted)} pre .k{color:var(--accent);font-weight:600}
pre.out{border-left:3px solid var(--accent)}
.formula{text-align:center;margin:1.6rem 0;overflow-x:auto;white-space:nowrap}
.tablewrap{overflow-x:auto;margin:1.25rem 0}
table{border-collapse:collapse;width:100%;font-size:.95em}
th,td{padding:.55rem .8rem;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
th{color:var(--muted)} th.n{text-align:right} td.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.9em}
aside.note{border-left:3px solid var(--accent);padding:.2rem 0 .2rem 1.1rem;margin:1.5rem 0;color:var(--muted)}
@media (max-width:760px){.shell{grid-template-columns:minmax(0,1fr)} aside.nav{display:none} header.site .search{display:none}
  main{padding:2rem 1rem 4rem} h1{font-size:2.1rem!important} .titleblock{margin:0 0 1.5rem!important;padding:1.25rem!important}}
"""

MARK = """<svg viewBox="-44 -44 88 88" aria-label="OpenQuant"><polygon points="-2,-10 -18,-38 16,-40 36,-20 14,0" fill="{magenta}"/>
<polygon points="-2,-10 -18,-38 16,-40 36,-20 14,0" fill="{cyan}" transform="rotate(120)"/>
<polygon points="-2,-10 -18,-38 16,-40 36,-20 14,0" fill="{yellow}" transform="rotate(240)"/></svg>""".format(**LOGO)

FIXTURE = """
<header class="site">@@MARK@@<span>OpenQuant</span><span class="search">Search&nbsp;&nbsp;Ctrl K</span>
<button onclick="var r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark'">Light / dark</button></header>
<div class="shell"><aside class="nav">
<h4>Getting started</h4><a>Overview</a><a>Quickstart</a>
<h4>Ch 3 · Labeling</h4><a class="on">labeling</a><a>bet_sizing</a>
<h4>Ch 4 · Sample weights</h4><a>sampling</a><a>sample_weights</a><a>sb_bagging</a>
<h4>Ch 7 · Validation</h4><a>cross_validation</a></aside>
<main><div class="titleblock"><p class="eyebrow">AFML chapter 3 · Module</p>
<h1>Triple-barrier labeling</h1>
<p class="lede">Label each event by the first barrier its price path touches, instead of by where the price happens to be after a fixed number of bars.</p></div>
<p class="status"><span class="pill">Reviewed</span>Read against AFML §3.4 and snippets 3.2–3.5 · validated 2026-09-19</p>
<p>Fixed-horizon labels ignore the path. A trade that reaches +5% and then reverses to −1% by the horizon is recorded as a loss, although any realistic exit rule would have taken the profit. The <a href="#">triple-barrier method</a> sets a profit-taking barrier, a stop-loss barrier and a vertical time barrier, and labels the observation by whichever is touched first. Barrier widths scale with a volatility estimate, usually from <code>get_daily_vol</code>.</p>
<h2>Definition</h2>
<p>For an event at <em>t</em><sub>0</sub> with target σ and multipliers <em>pt</em>, <em>sl</em>, the first-touch time is</p>
<p class="formula">τ = inf { t &gt; t₀ : r(t₀,t) ≥ pt·σ &nbsp;∨&nbsp; r(t₀,t) ≤ −sl·σ &nbsp;∨&nbsp; t = t₁ }</p>
<h2>Example</h2>
<pre><code><span class="k">from</span> openquant <span class="k">import</span> filters, labeling

<span class="c"># AFML ch. 2: emit an event only when cumulative drift exceeds 0.4%.</span>
events = filters.cusum_filter_timestamps(close, times, 0.004)
vertical = labeling.add_vertical_barrier(events, times, close, 0, 0, 30, 0)
labels = labeling.triple_barrier_labels(times, close, events, events, [0.005] * len(events), pt=1.0, sl=1.0, vertical_barrier_times=vertical)</code></pre>
<pre class="out"><code>500 bars -> 42 events -> 42 labels
label counts: {-1: 19, 1: 23}</code></pre>
<h2>Out-of-sample results</h2>
<div class="tablewrap"><table><thead><tr><th>Fold</th><th class="n">Events</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">Sharpe</th><th class="n">PSR</th></tr></thead><tbody>
<tr><td>2019 H1</td><td class="n">1,204</td><td class="n">0.581</td><td class="n">0.472</td><td class="n">1.14</td><td class="n">0.873</td></tr>
<tr><td>2019 H2</td><td class="n">988</td><td class="n">0.547</td><td class="n">0.510</td><td class="n">0.62</td><td class="n">0.701</td></tr>
<tr><td>2020 H1</td><td class="n">2,311</td><td class="n">0.533</td><td class="n">0.498</td><td class="n">−0.21</td><td class="n">0.412</td></tr>
<tr><td>2020 H2</td><td class="n">1,076</td><td class="n">0.569</td><td class="n">0.455</td><td class="n">0.97</td><td class="n">0.829</td></tr></tbody></table></div>
<aside class="note">Illustrative numbers for the design fixture only. They are not results.</aside>
</main></div>
"""


def luminance(hex_color):
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# (label, foreground token, background token, required minimum or None)
PAIRS = [
    ("body text on ground", "text", "ground", 4.5), ("body text on surface (code)", "text", "surface", 4.5),
    ("muted text on ground", "muted", "ground", 4.5), ("muted / comments on surface", "muted", "surface", 4.5),
    ("link and accent on ground", "accent", "ground", 4.5), ("keyword accent on surface", "accent", "surface", 4.5),
    ("status pill: ground on accent", "ground", "accent", 4.5), ("hairline rule on ground (informative)", "rule", "ground", None),
]


def tokens_css(tokens, grid_alpha):
    body = ";".join(f"--{k}:{v}" for k, v in tokens.items())
    r, g, b = (int(tokens["text"][i:i + 2], 16) for i in (1, 3, 5))
    return f"{body};--grid:rgba({r},{g},{b},{grid_alpha})"


def build():
    report = ["# Contrast report", "", "Computed by `build_candidates.py` from each candidate's tokens (WCAG 2.1 relative luminance).",
              "Required pairs need ≥ 4.5:1. The hairline row is informative: rules are not text.", ""]
    failures = []
    for cid, c in CANDIDATES.items():
        light, dark = tokens_css(c["light"], 0.07), tokens_css(c["dark"], 0.07)
        html = f"""<!doctype html><html lang="en" data-theme="{c['first']}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{c['name']} — OpenQuant identity candidate</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?{c['fonts']}&display=swap" rel="stylesheet">
<style>:root[data-theme=light]{{{light}}} :root[data-theme=dark]{{{dark}}}
{BASE_CSS}{c['css']}</style>
<script>function t(){{var h=location.hash.slice(1);if(h==='dark'||h==='light')document.documentElement.dataset.theme=h}}t();addEventListener('hashchange',t)</script>
</head><body>{FIXTURE.replace('@@MARK@@', MARK)}</body></html>"""
        (OUT / "candidates" / f"{cid}.html").write_text(html)

        report += [f"## {c['name']}", "", f"Accent derived from the logo's **{c['accent_from']}** `{LOGO[c['accent_from']]}`.", "",
                   "| Pair | Light | Dark | Required |", "| --- | ---: | ---: | ---: |"]
        for label, fg, bg, need in PAIRS:
            cells = []
            for theme in ("light", "dark"):
                ratio = contrast(c[theme][fg], c[theme][bg])
                cells.append(f"{ratio:.2f}")
                if need and ratio < need:
                    failures.append(f"{cid} {theme}: {label} = {ratio:.2f}")
            report.append(f"| {label} | {cells[0]} | {cells[1]} | {need or '—'} |")
        report.append("")
    report += ["## Result", "", "All required pairs pass." if not failures else "FAILURES:\n" + "\n".join(f"- {f}" for f in failures), ""]
    (OUT / "contrast.md").write_text("\n".join(report))
    return failures


if __name__ == "__main__":
    bad = build()
    print("contrast failures:", bad or "none")
    raise SystemExit(1 if bad else 0)
