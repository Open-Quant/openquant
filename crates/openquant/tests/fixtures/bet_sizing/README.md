# Bet sizing fixtures

- `reserve_fixture.json`: Generated via Python `.venv_x86` using `scripts/gen_bet_sizing_fixtures.py`. Contains EF3M fit params and resulting `bet_size` for a 500-sample synthetic dataset (same seed as mlfinlab tests).

Generation command:
```bash
cd mlfinlab
.venv_x86/bin/python -c "import sys; sys.path.insert(0,''); import scripts.gen_bet_sizing_fixtures as m; m.main()"
```
