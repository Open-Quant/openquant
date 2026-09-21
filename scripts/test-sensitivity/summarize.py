#!/usr/bin/env python3
"""Deduplicate docs/test-sensitivity-results.tsv (last run of each id/phase wins) and print a
per-module summary of the mutation audit. Run after mutate.py."""
import collections
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "docs", "test-sensitivity-results.tsv")

rows = collections.OrderedDict()
for line in open(PATH):
    c = line.rstrip("\n").split("\t")
    if c[0] != "id":
        rows[(c[0], c[1])] = c
with open(PATH, "w") as f:
    f.write("id\tphase\tfile\tmutation\ttest_targets\tresult\n")
    for c in rows.values():
        f.write("\t".join(c) + "\n")

groups = collections.OrderedDict()
for (mid, phase), c in rows.items():
    control = mid.startswith("CTL-")
    module = mid[4:] if control else mid.rsplit("-", 1)[0]
    g = groups.setdefault(module, {"ctl": {}, "before": [], "after": []})
    verdict = c[5].split(" ")[0]
    if control:
        g["ctl"][phase] = verdict
    else:
        g[phase].append((mid.rsplit("-", 1)[1], verdict))

for module, g in groups.items():
    kb = sum(v == "KILLED" for _, v in g["before"])
    ka = sum(v == "KILLED" for _, v in g["after"])
    survivors = ",".join(n for n, v in g["before"] if v != "KILLED") or "-"
    after = f"{ka}/{len(g['after'])}" if g["after"] else "-"
    ctl = g["ctl"].get("before", "?") + ("->" + g["ctl"]["after"] if "after" in g["ctl"] else "")
    print(f"{module:28s} control {ctl:20s} before {kb}/{len(g['before'])}  after {after:5s} survivors(before) {survivors}")

real = [(k, c) for k, c in rows.items() if not k[0].startswith("CTL-")]
before = [c for k, c in real if k[1] == "before"]
print("realistic mutations killed before:", sum(c[5].startswith("KILLED") for c in before), "/", len(before))
redone = {k[0] for k, _ in real if k[1] == "after"}
print(
    "modules worked on: before",
    sum(rows[(i, "before")][5].startswith("KILLED") for i in redone),
    "after",
    sum(rows[(i, "after")][5].startswith("KILLED") for i in redone),
    "of",
    len(redone),
)
