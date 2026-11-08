import random
from fractions import Fraction

import pytest
from openquant.feature_diagnostics import _solve_fraction_system


def test_fraction_solver_recovers_known_rational_solutions():
    rng = random.Random(904)
    for n in range(1, 7):
        for case in range(12):
            a = [
                [Fraction(rng.randrange(-9, 10), rng.randrange(1, 20)) for _ in range(n)]
                for _ in range(n)
            ]
            for i in range(n):
                a[i][i] = 1 + sum(abs(v) for j, v in enumerate(a[i]) if j != i)
            # Permuting rows forces pivots without changing the exact solution.
            rng.shuffle(a)
            x = [
                Fraction(rng.randrange(-20, 21), rng.randrange(1, 30))
                * Fraction(2) ** ((case - 6) * 150)
                for _ in range(n)
            ]
            b = [sum((v * z for v, z in zip(row, x)), Fraction()) for row in a]
            original = [row[:] for row in a]
            assert [v.hex() for v in _solve_fraction_system(a, b)] == [float(v).hex() for v in x]
            assert a == original


def test_fraction_solver_swaps_and_singular_systems():
    q = Fraction
    a = [[q(0), q(1, 3)], [q(2, 7), q(1, 11)]]
    assert _solve_fraction_system(a, [q(1), q(2)]) == [float(q(133, 22)), 3.0]
    for a, b in [([[q(0)]], [q(1)]), ([[q(1), q(2)], [q(2), q(4)]], [q(3), q(6)])]:
        with pytest.raises(ValueError, match="Singular linear system"):
            _solve_fraction_system(a, b)
    assert _solve_fraction_system([], []) == []
