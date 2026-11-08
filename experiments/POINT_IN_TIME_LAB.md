# Point-in-time feature revisions

This original synthetic example separates an observation's event time from the
time its value became usable. It adds no production data API and diagnoses no bug
in the existing OHLCV cleaner. That cleaner explicitly keeps the first/last
surviving **input row** for each symbol/time key; a final-data snapshot and a
historical availability query are different contracts.

## Lookup contract

`PointInTimeIndex(records).lookup(entity, feature, decision_time)` snapshots six
fields: entity, feature, event_time, available_time, value and revision_id.
Every timestamp must be timezone-aware; offsets normalize to UTC. Resolution is
microseconds. Strings use `YYYY-MM-DDTHH:MM:SS[.ffffff]` followed by `Z` or a
known `±HH:MM` offset. Finer fractions, offset seconds/fractions, and the unknown
`-00:00` offset are rejected before Python datetime parsing. This prevents
silent truncation or offset rounding at a cutoff. Aware datetime objects normalize
their represented offsets directly. Values must be
finite real numbers exactly representable as binary64, not booleans. Integers or
other real inputs that would lose precision during conversion are rejected, so
distinct large integers cannot silently collapse into a nonconflicting tie. Missing timestamps and values are rejected,
not imputed. The caller must supply truthful actual usable availability times;
this utility cannot establish them from a provider's scheduled release date.

Both eligibility comparisons are inclusive:

`event_time <= decision_time AND available_time <= decision_time`.

Among eligible records, the latest event wins; within that event, the latest
available revision wins. A late revision of an older event does not displace a
newer eligible event. This is an explicit policy, not a universal feature-store
rule. No age limit is imposed: an eligible value may be stale. The returned
selection includes both timestamps, value and sorted revision IDs. Missing
results have an explicit status and no fabricated value.

Exact duplicate rows collapse. Tied publications with the same value combine
revision IDs deterministically; signed zero is canonicalized. Conflicting values
at the **selected** event/publication key raise an error. Conflicts at ineligible
future keys do not affect an earlier query; superseded conflicts are not audited.
Malformed rows still fail index construction even when future-dated. The utility
is therefore a query reference, not a global data-quality audit. Mutating input
or returned dictionaries cannot rewrite the index's immutable row snapshots.

The deliberately contaminated `event_only_final_revision` mode ignores the
availability cutoff, while retaining event_time <= decision_time. It represents
looking back through today's revisions. Its name is explicit so it cannot be
mistaken for the default availability-aware lookup.

## Runnable lookup example

From the repository root:

```python
from experiments.point_in_time_lab import PointInTimeIndex

key = dict(entity="A", feature="measurement", event_time="2020-01-01T00:00:00Z")
index = PointInTimeIndex([
    dict(key, available_time="2020-01-02T09:00:00Z", value=10, revision_id="initial"),
    dict(key, available_time="2020-01-05T09:00:00Z", value=12, revision_id="revised"),
])
result = index.lookup("A", "measurement", "2020-01-03T12:00:00Z")
assert result["selected"]["value"] == 10
assert result["selected"]["revision_ids"] == ["initial"]
```

The later revision remains stored but cannot enter that earlier decision.

## Primary grounding

[The Federal Reserve Bank of St. Louis real-time API documentation](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
distinguishes today's revised history from information known in earlier periods.
[Its release-date documentation](https://fred.stlouisfed.org/docs/api/fred/releases_dates.html)
also warns that a publisher's release date need not equal when the data becomes
available on FRED or ALFRED. Neither date is automatically an intraday trading
availability timestamp.

[Feast's current point-in-time documentation](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
explicitly distinguishes its event-time default from an opt-in created-time
cutoff. That distinction motivates this original reference API; no Feast code is
copied or tested here, and no claim is made about every feature store's defaults.

## Frozen synthetic protocol

Before retained runs: 200 independent replications, 64 daily decisions per
replication, seed 97, four fixed families. Labels are independent fair binary
latent states, revealed at midnight the following day. The fixed predictor maps
a positive feature to label 1 and a negative feature to label 0; it fits no
parameters and accesses no labels at decision time. Initial features are noisy
proxies jointly generated with latent states; predictive correlation is a model
assumption, not evidence of access to revealed future labels.

All event times are midnight, decisions are noon:

- **On time:** a 75%-accurate proxy arrives at 06:00; no revisions. Both lookups
  should agree. A constant-positive predictor has population accuracy .5.
- **Delayed release:** the same proxy arrives at 18:00. The safe lookup is missing
  on the first day and then carries an older event. Current labels are independent
  of those older proxies, so conditional accuracy is .5; unconditional correct
  frequency is `(days-1)/(2*days)`. Event-only lookup improperly sees the current
  proxy before publication and has population accuracy .75.
- **Late revision:** an uninformative initial proxy arrives at 06:00. A revision
  exactly matching the now-revealed label arrives two days after the event.
  Safe accuracy is .5; event-only revision accuracy is 1 by construction.
- **Signal then revision:** the initial 06:00 proxy is 75% accurate, then the same
  perfect delayed revision arrives. Safe accuracy is .75; contaminated accuracy
  is 1. Leakage can inflate an already useful signal as well as a useless one.

The constant-positive baseline is evaluated on the same labels. Missing decisions
remain explicit. Accuracy conditional on a selected record is descriptive; the
unconditional correct fraction counts all decisions, so abstention is not hidden.
Monte Carlo error for that fraction uses independent replication means, not an
assumption that all within-replication decisions are independent. Every replication
retains counts and an input hash; the first replication additionally retains all
source rows and decision-level provenance. Remaining inputs regenerate from the
saved configuration, seed and family identity.

## Cost, use and limits

Index construction groups and sorts records; lookup scans one entity/feature
group until the first eligible row, with worst-case linear work in that group.
This small reference does not promise database-scale performance. Reports record
row/query counts and elapsed time. NumPy generates the experiment; lookup uses
Python datetimes and immutable tuples rather than OpenQuant's native extension.
There are no external data calls, monetary costs, trades or investment claims.

```bash
python experiments/point_in_time_lab.py --out /tmp/point-in-time.json
python -m pytest python/tests/test_point_in_time_lab.py -q
```

Real integration still requires provider-specific availability semantics, clock
precision, processing latency, retention/versioning policy, stale-feature policy
and business-specific event precedence. Filtering by timestamps cannot repair
incorrect metadata or retrospective edits to the recorded availability clock.

## Retained findings

The declared 51,200-decision protocol completed in about 1.81 seconds in the
research environment (76,800 generated feature rows and 102,400 lookups). This
single machine timing is not a database throughput benchmark. Source SHA256 was
`61d5ee94c8acd6b9ebbfab389c3cf7ff50ad3120fafeb9587f0671479ab81ad6`.

On-time accuracy was .751484 for both lookups, retaining the neutral control.
Delayed release gave .494609 correct decisions overall for the safe lookup
(.502460 conditional on a selection), versus .750469 for the contaminated lookup.
The safe run retained 200 missing decisions and 12,600 older-event selections.
The uninformative-initial/late-revision control produced .500234 safe accuracy
versus 1.0 contaminated accuracy; informative initial proxies gave .744609 versus
1.0. Perfect contaminated scores are constructed leakage controls, not empirical
performance discoveries. Every safe lookup used zero future publications.
