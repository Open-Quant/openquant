"""Correlated feature rejections must have a surviving witness."""

from openquant.feature_diagnostics import feature_screen_report


def test_rejected_intermediate_cannot_remove_an_unrelated_feature():
    u = [-1.0, -1.0, 1.0, 1.0]
    v = [-1.0, 1.0, -1.0, 1.0]
    # Corr(A, B) = 1/sqrt(5), Corr(B, C) = 2/sqrt(5), Corr(A, C) = 0.
    # All are complete and nonconstant, with std(A) > std(B) > std(C).
    rows = [[10 * x, 2 * (x + 2 * y), y] for x, y in zip(u, v)]
    report = feature_screen_report(rows, feature_names=["A", "B", "C"], max_corr=0.4)
    assert report["selected_features"] == ["A", "C"]
    assert report["rejection_reasons"] == {"B": ["high_corr>0.40_with:A"]}


def test_globally_constant_rejection_cannot_eliminate_eligible_feature():
    rows = [[-1.2e-12, -1.0], [1.2e-12, 1.0], *([[0.0, None]] * 6)]
    # f0 has usable pair-only dispersion but fails the full-column std gate.
    report = feature_screen_report(rows, min_coverage=0.25)
    assert report["selected_features"] == ["f1"]
    assert report["rejection_reasons"] == {"f0": ["constant_or_near_constant"]}
    assert [row["max_abs_corr"] for row in report["records"]] == [1.0, 1.0]


def test_seeded_screen_selects_a_maximal_set_against_exact_pearson():
    import random
    from fractions import Fraction

    rng = random.Random(1911)
    threshold = 0.45
    threshold_sq = Fraction(threshold) ** 2
    for _ in range(200):
        latent = [[rng.randrange(-20, 21) for _ in range(3)] for _ in range(12)]
        weights = [[rng.randrange(-3, 4) for _ in range(3)] for _ in range(9)]
        rows = [[sum(x * w for x, w in zip(r, f)) for f in weights] for r in latent]
        cols = list(zip(*rows))
        centered = [[len(col) * x - sum(col) for x in col] for col in cols]
        ss = [sum(x * x for x in col) for col in centered]
        report = feature_screen_report(rows, max_corr=threshold)
        kept = {int(name[1:]) for name in report["selected_features"]}

        def correlated(i, j):
            cross = sum(a * b for a, b in zip(centered[i], centered[j]))
            return cross**2 > threshold_sq * ss[i] * ss[j]

        assert all(ss[i] > 0 for i in kept)
        assert all(not correlated(i, j) for i in kept for j in kept if i != j)
        for i, spread_sq in enumerate(ss):
            if i in kept or spread_sq == 0:
                continue
            [reason] = report["rejection_reasons"][f"f{i}"]
            witness = int(reason.rsplit(":f", 1)[1])
            assert witness in kept and correlated(i, witness)
            assert (ss[witness], -witness) >= (spread_sq, -i)
