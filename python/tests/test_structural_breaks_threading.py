"""Long structural-break calls must let other Python threads make progress."""

import sys
import threading

import pytest
from openquant import structural_breaks


@pytest.mark.parametrize(
    ("name", "size", "args"),
    [
        ("get_chow_type_stat", 512, (20,)),
        ("get_chu_stinchcombe_white_statistics", 4096, ("one_sided",)),
        ("get_sadf", 120, ("linear", True, 20, 1)),
    ],
)
def test_structural_breaks_release_gil_during_computation(name, size, args):
    # These inputs keep even an optimized build busy long enough to schedule a waiting
    # observer. This is a progress check, not a wall-clock speed threshold.
    series = [4.0 + 0.001 * i + 0.002 * (i % 2) for i in range(size)]
    function = getattr(structural_breaks, name)
    ready = threading.Event()
    start = threading.Event()
    finished = threading.Event()
    progressed = threading.Event()

    def observe():
        ready.set()
        start.wait()
        if not finished.is_set():
            progressed.set()

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    assert ready.wait(timeout=5), "observer did not start"
    previous_interval = sys.getswitchinterval()
    try:
        # Prevent an ordinary bytecode timeslice immediately before/after the native call
        # from making a GIL-holding implementation look correct. The observer waits on an
        # Event and only runs a few instructions, so it cannot monopolize the interpreter.
        sys.setswitchinterval(1000.0)
        start.set()
        function(series, *args)
        finished.set()
    finally:
        finished.set()
        start.set()
        sys.setswitchinterval(previous_interval)
        observer.join(timeout=5)
    assert not observer.is_alive(), "observer did not finish"
    assert progressed.is_set(), f"{name} held the GIL throughout the computation"
