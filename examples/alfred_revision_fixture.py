"""Public two-point date-domain fixture, not a FRED adapter or complete vintage history.

Facts are attributed in fixtures/alfred_unemployment_revision.json. Selecting only
records with dates strictly before a cutoff illustrates a date-domain policy on
this tiny known subset; it does NOT establish what FRED served at a past instant.
No network calls, fabricated UTC times, or production API integration occur.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

FIXTURE = Path(__file__).with_name("fixtures") / "alfred_unemployment_revision.json"


def load_fixture():
    return json.loads(FIXTURE.read_text())


def latest_documented_before(fixture, cutoff):
    eligible = [r for r in fixture["records"] if date.fromisoformat(r["release_date"]) < cutoff]
    if not eligible:
        return None
    latest = max(date.fromisoformat(r["release_date"]) for r in eligible)
    winners = [r for r in eligible if date.fromisoformat(r["release_date"]) == latest]
    if len(winners) != 1:
        raise ValueError("ambiguous fixture release date")
    return dict(winners[0])


def check_selected_association(selected, candidate):
    """Exact documented tuple identity; not an authenticity or completeness proof."""
    fields = ("fixture_id", "release_date", "value")
    if any(selected[key] != candidate[key] for key in fields):
        raise ValueError("candidate differs from selected documented revision")


def run():
    fixture = load_fixture()
    earlier = latest_documented_before(fixture, date(1990, 2, 3))
    later = latest_documented_before(fixture, date(1996, 3, 9))
    naive_latest = max(fixture["records"], key=lambda r: r["release_date"])
    changed_value = dict(earlier, value=naive_latest["value"])
    try:
        check_selected_association(earlier, changed_value)
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    return dict(
        source=fixture["source"],
        observation=fixture["observation"],
        unit=fixture["unit"],
        earlier_known_subset_selection=earlier,
        later_known_subset_selection=later,
        naive_latest_for_earlier_cutoff=naive_latest,
        difference_percentage_points=str(Decimal(later["value"]) - Decimal(earlier["value"])),
        same_date_record_excluded=latest_documented_before(fixture, date(1990, 2, 2)) is None,
        old_identity_with_later_value_rejected=mismatch_rejected,
        nanosecond_binding="not attempted: exact event/publication clocks are not supplied",
        limitations=fixture["limitations"],
    )


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
