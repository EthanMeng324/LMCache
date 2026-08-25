# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import asyncio


from lmcache.v1.cache_engine import LMCacheEngine
import lmcache.v1.cache_engine as cache_engine_module


class _Tensor:
    def __init__(self):
        self.copied_from = None

    def copy_(self, other, non_blocking=False):
        self.copied_from = other


class _MemoryObj:
    def __init__(self, size=128):
        self.tensor = _Tensor()
        self.meta = SimpleNamespace(fmt="kv")
        self.refs = 1
        self.size = size

    def get_shape(self):
        return (1,)

    def get_dtype(self):
        return "dtype"

    def get_size(self):
        return self.size

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1


class _Source:
    def __init__(self, objects):
        self.objects = objects
        self.pins = {key: 0 for key in objects}

    def contains(self, key, pin=False):
        if key not in self.objects:
            return False
        return True

    def pin(self, key):
        if key not in self.objects:
            return False
        self.pins[key] += 1
        return True

    def unpin(self, key):
        self.pins[key] -= 1


class _Target:
    def __init__(self, existing=(), fail_allocate=False):
        self.objects = {key: _MemoryObj() for key in existing}
        self.fail_allocate = fail_allocate

    def contains(self, key, pin=False):
        return key in self.objects

    def allocate(self, shape, dtype, fmt, eviction, busy_loop):
        if self.fail_allocate:
            return None
        return _MemoryObj()

    def submit_put_task(self, key, memory_obj):
        memory_obj.ref_count_up()
        self.objects[key] = memory_obj


class _StorageManager:
    def __init__(self, source, target):
        self.storage_backends = {
            "CxlBackend": source,
            "LocalCPUBackend": target,
        }
        self.removed = []

    def batched_get(self, keys, location):
        return [self.storage_backends[location].objects[key] for key in keys]

    def batched_remove(self, keys, locations):
        self.removed.extend(keys)


def _engine(source, target):
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.storage_manager = _StorageManager(source, target)
    return engine


class _CrossNodeFormat:
    def token_dim(self):
        return 0


class _CrossNodeMemoryObj:
    def __init__(self, tokens=16):
        self.meta = SimpleNamespace(shape=(tokens,), fmt=_CrossNodeFormat())
        self.refs = 1

    def ref_count_down(self):
        self.refs -= 1


class _CompletedFuture:
    def result(self):
        return None


def test_cross_node_move_releases_pins_and_staging_objects(monkeypatch):
    """A P2P move owns neither the source pin nor its staging references."""

    memory_obj = _CrossNodeMemoryObj()
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.lookup_pins = {"move-1": {"LocalCPUBackend": ["key-1"]}}
    engine.lookup = lambda *args, **kwargs: 16
    unpinned = []
    engine.lookup_unpin = lambda lookup_id: unpinned.append(lookup_id)

    class _P2P:
        async def async_batched_submit_put_task(self, *args, **kwargs):
            return None

    class _StorageManager:
        loop = object()
        storage_backends = {"P2PBackend": _P2P()}

        def batched_get(self, keys, location):
            assert keys == ["key-1"]
            assert location == "LocalCPUBackend"
            return [memory_obj]

        def batched_remove(self, keys, locations):
            raise AssertionError("copy=True must not remove the source")

    engine.storage_manager = _StorageManager()

    def _submit(coro, loop):
        coro.close()
        return _CompletedFuture()

    monkeypatch.setattr(
        cache_engine_module.asyncio, "run_coroutine_threadsafe", _submit
    )

    assert engine.move(
        [1, 2], "LocalCPUBackend", ("peer:1", "LocalCPUBackend"), "move-1"
    ) == 16
    assert memory_obj.refs == 0
    assert unpinned == ["move-1"]


def test_cross_node_move_unpins_when_lookup_has_no_complete_prefix():
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.lookup_pins = {}
    engine.lookup = lambda *args, **kwargs: 0
    unpinned = []
    engine.lookup_unpin = lambda lookup_id: unpinned.append(lookup_id)
    engine.storage_manager = SimpleNamespace()

    assert engine.move(
        [1, 2], "LocalCPUBackend", ("peer:1", "LocalCPUBackend"), "move-none"
    ) == 0
    assert unpinned == ["move-none"]


def test_promotes_non_contiguous_arbitrary_keys_and_balances_ownership():
    source_objects = {"A": _MemoryObj(), "C": _MemoryObj()}
    source = _Source(source_objects)
    target = _Target(existing=["B"])
    engine = _engine(source, target)

    result = engine.promote_keys_intra_node(
        ["A", "B", "missing", "C"],
        "CxlBackend",
        "LocalCPUBackend",
        "test-event",
    )

    assert result.promoted == ("A", "C")
    assert result.already_present == ("B",)
    assert result.source_missing == ("missing",)
    assert result.bytes_promoted == 256
    assert source.pins == {"A": 0, "C": 0}
    assert source_objects["A"].refs == 0
    assert source_objects["C"].refs == 0
    assert target.objects["A"].refs == 1
    assert target.objects["C"].refs == 1


def test_allocation_failure_releases_staging_and_source_pin():
    staging = _MemoryObj()
    source = _Source({"A": staging})
    target = _Target(fail_allocate=True)
    engine = _engine(source, target)

    result = engine.promote_keys_intra_node(
        ["A"], "CxlBackend", "LocalCPUBackend", "test-event"
    )

    assert result.failed == ("A",)
    assert source.pins["A"] == 0
    assert staging.refs == 0

