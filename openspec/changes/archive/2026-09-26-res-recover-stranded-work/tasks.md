## 1. Port

- [x] 1.1 Add `PurgedSplitDiagnostics`, `PurgedSplit`, `CpcvSplit` and `CpcvPath` with rustdoc
- [x] 1.2 Rebuild `split` on one split builder and add `split_with_diagnostics`
- [x] 1.3 Add `cpcv_splits`, `cpcv_paths`, `naive_kfold_splits` and `count_train_test_overlaps`
- [x] 1.4 Add typed error variants and validation in `PurgedKFold::new`

## 2. Tests

- [x] 2.1 Port the four `59ac6fa` tests (`cpcv_paths` count checked on `cpcv_splits`)
- [x] 2.2 Regression test against the previous `split` on random labels and embargoes
- [x] 2.3 AFML §12.4 split and path counts for N ≤ 8, and the N = 6, k = 2 assignment
- [x] 2.4 Leakage property test for CPCV, and invalid-input tests

## 3. Docs

- [x] 3.1 Document the new functions on the cross_validation page and regenerate the API inventory
