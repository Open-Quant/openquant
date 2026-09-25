## ADDED Requirements

### Requirement: Sharpe-ratio moments from a returns series
`openquant.evaluation.return_moments(returns)` SHALL return the number of observations, the
per-period Sharpe ratio (mean over the sample standard deviation with n − 1, not annualised),
the population skewness m3/m2^1.5 and the raw population kurtosis m4/m2^2 (3 for a normal
distribution). Every statistic in the module SHALL use these conventions.

#### Scenario: Committed fixture
- **WHEN** `return_moments` is called on `python/tests/fixtures/evaluation_returns.csv`
- **THEN** it returns n_obs 120, Sharpe 0.122705, skewness −0.475417 and kurtosis 3.697184, matching a standard-library hand computation to 1e-12 relative

### Requirement: Probabilistic Sharpe ratio
`probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)` SHALL return
Z[(SR − SR*)·√(T − 1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²)] (Bailey and López de Prado 2012;
AFML §14.7.2), with Z the standard normal CDF and SR, γ3, γ4, T from `return_moments`.

#### Scenario: PSR against zero and against a positive benchmark
- **WHEN** PSR is computed for the fixture with benchmark 0 and with benchmark 0.05
- **THEN** it returns 0.902329 and 0.778541 respectively, to 1e-9

### Requirement: Deflated Sharpe ratio
`expected_max_sharpe(n_trials, sharpe_std)` SHALL return
SR0 = σ_SR·[(1 − γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e))], γ the Euler–Mascheroni constant
(Bailey and López de Prado 2014; AFML §14.7.3). `deflated_sharpe_ratio(returns, ...)` SHALL
return the PSR of the returns against SR0, taking the trials either as a list of every
trial's per-period Sharpe ratio (N its length, σ_SR its population standard deviation) or as
`n_trials` and `sharpe_std`.

#### Scenario: Eight trials given as Sharpe ratios
- **WHEN** DSR is computed for the fixture with trial Sharpe ratios 0.05, −0.02, 0.11, 0.03, 0.08, −0.04, 0.01 and the fixture's own
- **THEN** σ_SR is 0.055199, SR0 is 0.080536 and DSR is 0.671845, to 1e-9

#### Scenario: Trials given as a count and a dispersion
- **WHEN** DSR is computed for the fixture with 100 trials and σ_SR 0.05
- **THEN** SR0 is 0.126530 and DSR is 0.483898, to 1e-9

### Requirement: Minimum track record length
`minimum_track_record_length(returns, benchmark_sr=0.0, alpha=0.05)` SHALL return
1 + (1 − γ3·SR + (γ4 − 1)/4·SR²)·(Z⁻¹(1 − α) / (SR − SR*))² (Bailey and López de Prado 2012),
and SHALL return infinity when SR is at or below SR*.

#### Scenario: MinTRL for the fixture
- **WHEN** MinTRL is computed for the fixture at (SR* 0, α 0.05), (SR* 0.05, α 0.05) and (SR* 0, α 0.10)
- **THEN** it returns 193.0008, 547.8914 and 117.5522 observations, to 1e-9 relative

#### Scenario: Underperforming track record
- **WHEN** the benchmark is at or above the observed Sharpe ratio
- **THEN** MinTRL is `math.inf`

### Requirement: Persistent trial registry
`TrialRegistry(path)` SHALL persist every recorded trial to a JSON file at the given path,
with the configuration's SHA-256 hash of canonical JSON, a UTC timestamp, the per-period
Sharpe ratio, the number of observations, skewness, kurtosis and the configuration itself.
Writes SHALL be atomic (temporary file in the same directory, then rename) and SHALL re-read
the file first so that registries opened on the same path share one count. Recording a
configuration whose hash is already registered SHALL replace its entry rather than add a
trial.

#### Scenario: Two runs share a registry
- **WHEN** one `TrialRegistry` records two configurations and a second `TrialRegistry` opened later on the same path records a third
- **THEN** the second registry reports 3 trials, a third registry opened on the path reports 3, and the first registry reports 3 after `reload()`

#### Scenario: Re-running a configuration
- **WHEN** a configuration already in the registry is recorded again
- **THEN** the trial count is unchanged and the entry holds the latest result

#### Scenario: Invalid registry file or configuration
- **WHEN** the path holds a file that is not a schema-1 registry, or a configuration is not JSON-serialisable
- **THEN** a `ValueError` or `TypeError` is raised and nothing is written

### Requirement: DSR deflated by the registry
`TrialRegistry.deflated_sharpe_ratio(returns)` SHALL deflate using N = the registry's trial
count and σ_SR = the population standard deviation of the registered Sharpe ratios, and
SHALL raise `ValueError` when the registry holds fewer than two trials.

#### Scenario: Registry of eight trials recorded over two runs
- **WHEN** seven configurations with Sharpe ratios 0.05, −0.02, 0.11, 0.03, 0.08, −0.04, 0.01 are recorded in one run and the fixture in a second
- **THEN** the registry DSR of the fixture equals the eight-trial DSR, 0.671845, and recording a ninth trial changes it to the nine-trial hand-computed value

### Requirement: Meta-labeling overlay metrics
`meta_label_metrics(meta_labels, meta_predictions, threshold=0.5)` SHALL return precision,
recall, F1, accuracy and the confusion counts of acting on bets whose prediction is at least
`threshold`, together with the precision, recall and F1 of the primary model that acts on
every bet (AFML §3.6). A ratio with a zero denominator SHALL be reported as 0.0.

#### Scenario: Ten bets
- **WHEN** labels 1,1,1,1,0,0,0,0,0,1 are scored against predictions 0.9,0.8,0.3,0.6,0.7,0.2,0.1,0.4,0.55,0.45
- **THEN** TP 3, FP 2, FN 2, TN 3, precision, recall and F1 are 0.6, and the primary model has precision 0.5, recall 1 and F1 2/3

### Requirement: Strategy-failure probability
`strategy_failure_probability` SHALL return the report of
`strategy_risk.estimate_strategy_failure_probability` (AFML §15.4) with an added
`failure_probability` equal to its `empirical_failure_probability`.

#### Scenario: Same answer as the binding
- **WHEN** it is called with the same bets, horizon and seed as the binding
- **THEN** `failure_probability` equals the binding's `empirical_failure_probability`

### Requirement: Input validation
Functions taking returns SHALL raise `ValueError` for fewer than 3 observations, non-finite
values or constant returns, and `TypeError` for non-numeric input. `alpha` SHALL be in (0, 1),
`n_trials` an integer of at least 2, `sharpe_std` non-negative, and meta-labels 0 or 1 with
predictions in [0, 1] and of equal length.

#### Scenario: Invalid inputs
- **WHEN** any of these constraints is violated
- **THEN** the call raises before computing, with a message naming the argument
