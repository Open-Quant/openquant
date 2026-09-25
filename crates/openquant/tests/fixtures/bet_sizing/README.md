# Bet sizing fixtures

The fixtures live in `tests/fixtures/bet_sizing/` at the repository root, each with its generator
beside it (neither imports openquant or mlfinlab):

- `reserve_fixture.json`: AFML section 10.2 "reserve" sizing on 500 synthetic events
  (`numpy default_rng(138)`): concurrent long/short counts, a two-Gaussian fit and the bet sizes.
  `uv run --with pandas --with scipy python tests/fixtures/bet_sizing/generate_reserve.py`
- `prob_dynamic_budget.json`: AFML snippets 10.1-10.4 and the budget approach of section 10.2.
  `uv run --with pandas --with scipy python tests/fixtures/bet_sizing/generate_prob_dynamic_budget.py`
