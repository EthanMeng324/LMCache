"""Unit tests for serving-priority CXL admission."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import threading


_MODULE_PATH = (
    Path(__file__).parents[3]
    / "lmcache"
    / "v1"
    / "storage_backend"
    / "cxl_resource_governor.py"
)
_SPEC = spec_from_file_location("cxl_resource_governor_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CxlResourceGovernor = _MODULE.CxlResourceGovernor


def test_background_is_deferred_while_serving_is_active():
    governor = CxlResourceGovernor()

    with governor.serving():
        with governor.try_background() as admitted:
            assert not admitted

    snapshot = governor.snapshot()
    assert snapshot.serving_active == 0
    assert snapshot.background_active is False
    assert snapshot.background_deferred == 1


def test_waiting_serving_blocks_a_new_background_chunk():
    governor = CxlResourceGovernor()
    serving_entered = threading.Event()
    release_serving = threading.Event()

    def serve():
        with governor.serving():
            serving_entered.set()
            release_serving.wait(timeout=2)

    with governor.try_background() as admitted:
        assert admitted
        thread = threading.Thread(target=serve)
        thread.start()

        assert not serving_entered.wait(timeout=0.05)
        with governor.try_background() as second_admitted:
            assert not second_admitted

    assert serving_entered.wait(timeout=1)
    release_serving.set()
    thread.join(timeout=1)
    assert not thread.is_alive()

    snapshot = governor.snapshot()
    assert snapshot.background_admitted == 1
    assert snapshot.background_deferred == 1
    assert snapshot.serving_waited == 1
