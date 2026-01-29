# SPDX-License-Identifier: Apache-2.0
"""
Standalone stress test for cxl_shm slot exhaustion.

This script is intentionally NOT a pytest test. Run it directly, e.g.:

  sudo -E env LMCACHE_CXL_DAX_DEVICE=/dev/dax1.0
    ./venv/bin/python tests/cxl_shm_slot_stress.py --max-objs 2097152 --obj-size 14680064

What it tests:
- Creates many CXL shared objects via libcxl_shm.so and stops at first failure.
- Helps distinguish:
  - "slot/hash table" exhaustion (No free slots available for shared objects)
  - DAX mapped size exhaustion (cxl_shm.c currently aborts on this)
  - name collision (object already exists)

Notes:
- cxl_shm.c hardcodes CXL_SHM_DAX_SIZE = 32GB.
- The hash placement uses mem_hash_init(mh, 10, 200000) and tries at most 10
  candidate indices per key; it does NOT probe the whole table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from lmcache.v1.storage_backend.cxl_shm_binding import CxlShmWrapper


DEFAULT_BUCKETS_10_200000 = [
    199999,
    199967,
    199961,
    199933,
    199931,
    199921,
    199909,
    199889,
    199877,
    199873,
]


def djb2_u64(s: str) -> int:
    h = 5381
    for ch in s:
        h = ((h << 5) + h) + ord(ch)  # h*33 + c
        h &= (1 << 64) - 1
    return h


def simulate_slots(max_objs: int, prefix: str) -> tuple[int, int]:
    """Simulate find_avail_hash_idx() placement without touching the DAX device.

    Returns:
        (placed, max_node)
    """
    buckets = DEFAULT_BUCKETS_10_200000
    max_node = sum(buckets)
    used = bytearray(max_node)  # 0/1 occupancy of each slot

    placed = 0
    for i in range(max_objs):
        name = make_name(prefix, i)
        key = djb2_u64(name)
        base = 0
        placed_here = False
        for lvl, b in enumerate(buckets):
            if lvl > 0:
                base += buckets[lvl - 1]
            idx = base + (key % b)
            if used[idx] == 0:
                used[idx] = 1
                placed_here = True
                break
        if not placed_here:
            return placed, max_node
        placed += 1
    return placed, max_node


def make_name(prefix: str, i: int) -> str:
    # Ensure <= 20 bytes (CXL_SHM_ONAME_LEN).
    # We'll use a fixed-width base36 counter.
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    n = i
    chars = []
    if n == 0:
        chars.append("0")
    else:
        while n:
            n, r = divmod(n, 36)
            chars.append(alphabet[r])
    suffix = "".join(reversed(chars))
    name = f"{prefix}{suffix}"
    # Truncate defensively; for stable distribution, keep the rightmost chars.
    if len(name) > 20:
        name = name[-20:]
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-objs", type=int, default=200_000, help="Max objects to create")
    ap.add_argument(
        "--obj-size",
        type=int,
        default=64,
        help="Size in bytes for each created object",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="T",
        help="Object name prefix (keep short)",
    )
    ap.add_argument("--num-procs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument(
        "--reset",
        action="store_true",
        help="Call cxl_shm_finalize() before creating objects (rank 0 only)",
    )
    ap.add_argument(
        "--reset-metadata",
        action="store_true",
        help="Reset only metadata slots/cursor (debug; faster than full finalize)",
    )
    ap.add_argument(
        "--simulate-only",
        action="store_true",
        help="Only simulate hash-slot placement (no DAX access)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N successful creates",
    )
    ap.add_argument(
        "--reuse-after-destroy",
        action="store_true",
        help="Test that destroy() makes space reusable (same name create/destroy loop).",
    )
    ap.add_argument(
        "--reuse-iters",
        type=int,
        default=100_000,
        help="Iterations for --reuse-after-destroy loop",
    )
    ap.add_argument(
        "--reuse-keys",
        type=int,
        default=1,
        help="Number of distinct keys for --reuse-after-destroy (1 => single-key; >1 => multi-key).",
    )

    args = ap.parse_args()

    if args.simulate_only:
        placed, max_node = simulate_slots(args.max_objs, args.prefix)
        print(
            f"[simulate] placed={placed} max_node={max_node} "
            f"load_factor={placed / max_node:.6f}"
        )
        if placed < args.max_objs:
            print("[simulate] FAILED due to candidate-set exhaustion (bucket_levels too small)")
            return 2
        print("[simulate] OK")
        return 0

    dax = os.environ.get("LMCACHE_CXL_DAX_DEVICE", "<unset>")
    print(f"[info] LMCACHE_CXL_DAX_DEVICE={dax}")
    print("[info] cxl_shm.c currently hardcodes CXL_SHM_DAX_SIZE=32GB")
    print(f"[info] will create up to {args.max_objs} objects, obj_size={args.obj_size} bytes")

    cxl = CxlShmWrapper(num_procs=args.num_procs, rank=args.rank)
    rc = cxl.init()
    if rc != 0:
        print(f"[fatal] cxl_shm_init failed rc={rc}", file=sys.stderr)
        return 1

    if (counts := cxl.debug_count_in_use()) is not None:
        rin, rmax, tin, tmax, curr_off, levels = counts
        print(
            f"[info] debug_count_in_use: reachable_in_use={rin}/{rmax} "
            f"total_in_use={tin}/{tmax} bucket_levels={levels} curr_offset={curr_off}"
        )

    if args.reset:
        if args.rank != 0:
            print("[fatal] --reset requires --rank 0", file=sys.stderr)
            return 1
        print("[info] resetting dax metadata/data via cxl_shm_finalize() ...")
        cxl.finalize()
        rc = cxl.init()
        if rc != 0:
            print(f"[fatal] cxl_shm_init failed after reset rc={rc}", file=sys.stderr)
            return 1

    if args.reset_metadata:
        print("[info] resetting metadata via cxl_shm_reset_metadata() ...")
        rc = cxl.reset_metadata()
        if rc != 0:
            print(f"[fatal] cxl_shm_reset_metadata failed rc={rc}", file=sys.stderr)
            return 1
        if (counts := cxl.debug_count_in_use()) is not None:
            rin, rmax, tin, tmax, curr_off, levels = counts
            print(
                f"[info] after reset_metadata: reachable_in_use={rin}/{rmax} "
                f"total_in_use={tin}/{tmax} bucket_levels={levels} curr_offset={curr_off}"
            )

    if args.reuse_after_destroy:
        if args.reuse_keys <= 1:
            name = make_name(args.prefix, 0)
            print(f"[reuse] name={name!r} iters={args.reuse_iters} obj_size={args.obj_size}")
            counts0 = cxl.debug_count_in_use()
            if counts0 is not None:
                _, _, _, _, curr0, _ = counts0
                print(f"[reuse] curr_offset(start)={curr0}")

            # First create: establishes an offset and advances curr_offset once.
            rc, hnd = cxl.create(name, args.obj_size)
            if rc != 0 or hnd is None:
                print(f"[reuse] FATAL: initial create failed rc={rc}", file=sys.stderr)
                return 3
            first_off = int(hnd.obj_contents.offset) if hnd.obj else -1
            cxl.destroy(hnd)

            counts1 = cxl.debug_count_in_use()
            curr1 = None
            if counts1 is not None:
                _, _, _, _, curr1, _ = counts1
                print(f"[reuse] curr_offset(after first destroy)={curr1}")

            # Subsequent create/destroy cycles should reuse the same offset and NOT
            # keep increasing curr_offset.
            for i in range(1, args.reuse_iters + 1):
                rc, hnd = cxl.create(name, args.obj_size)
                if rc != 0 or hnd is None:
                    print(f"[reuse] FAILED: create failed at iter={i} rc={rc}", file=sys.stderr)
                    return 3
                off = int(hnd.obj_contents.offset) if hnd.obj else -1
                if off != first_off:
                    print(
                        f"[reuse] FAILED: offset changed at iter={i} off={off} first_off={first_off}",
                        file=sys.stderr,
                    )
                    return 3
                cxl.destroy(hnd)

                if args.progress_every and i % args.progress_every == 0:
                    counts = cxl.debug_count_in_use()
                    if counts is not None:
                        _, _, _, _, curr, _ = counts
                        print(f"[reuse] progress iter={i} curr_offset={curr}")
                        if curr1 is not None and curr != curr1:
                            print(
                                f"[reuse] FAILED: curr_offset grew (curr={curr} expected={curr1})",
                                file=sys.stderr,
                            )
                            return 3

            print("[reuse] OK: destroy() reuse works (offset stable; curr_offset stable)")
            return 0

        # Multi-key variant: create/destroy across a pool of distinct keys.
        names = [make_name(args.prefix, i) for i in range(args.reuse_keys)]
        print(
            f"[reuse-multi] keys={args.reuse_keys} iters={args.reuse_iters} "
            f"(total ops={args.reuse_keys * args.reuse_iters}) obj_size={args.obj_size}"
        )

        counts0 = cxl.debug_count_in_use()
        if counts0 is not None:
            _, _, _, _, curr0, _ = counts0
            print(f"[reuse-multi] curr_offset(start)={curr0}")

        # Round 1: allocate once per key, record offset, then destroy.
        first_off = {}
        for nm in names:
            rc, hnd = cxl.create(nm, args.obj_size)
            if rc != 0 or hnd is None:
                print(f"[reuse-multi] FATAL: initial create failed name={nm!r} rc={rc}", file=sys.stderr)
                return 3
            first_off[nm] = int(hnd.obj_contents.offset) if hnd.obj else -1
            cxl.destroy(hnd)

        counts1 = cxl.debug_count_in_use()
        curr1 = None
        if counts1 is not None:
            _, _, _, _, curr1, _ = counts1
            print(f"[reuse-multi] curr_offset(after round1)={curr1}")

        # Subsequent rounds: for each key, create/destroy and verify offset stability.
        ops = 0
        for r in range(1, args.reuse_iters + 1):
            for nm in names:
                rc, hnd = cxl.create(nm, args.obj_size)
                if rc != 0 or hnd is None:
                    print(
                        f"[reuse-multi] FAILED: create failed round={r} name={nm!r} rc={rc}",
                        file=sys.stderr,
                    )
                    return 3
                off = int(hnd.obj_contents.offset) if hnd.obj else -1
                if off != first_off[nm]:
                    print(
                        f"[reuse-multi] FAILED: offset changed round={r} name={nm!r} "
                        f"off={off} first_off={first_off[nm]}",
                        file=sys.stderr,
                    )
                    return 3
                cxl.destroy(hnd)
                ops += 1

                if args.progress_every and ops % args.progress_every == 0:
                    counts = cxl.debug_count_in_use()
                    if counts is not None:
                        _, _, _, _, curr, _ = counts
                        print(f"[reuse-multi] progress ops={ops} curr_offset={curr}")
                        if curr1 is not None and curr != curr1:
                            print(
                                f"[reuse-multi] FAILED: curr_offset grew (curr={curr} expected={curr1})",
                                file=sys.stderr,
                            )
                            return 3

        print("[reuse-multi] OK: destroy() reuse works for multiple keys")
        return 0

    start = time.time()
    created = 0
    total_bytes = 0
    for i in range(args.max_objs):
        name = make_name(args.prefix, i)
        rc, hnd = cxl.create(name, args.obj_size)
        if rc != 0 or hnd is None:
            elapsed = time.time() - start
            print()
            print("[result] CREATE FAILED")
            print(f"[result] i={i} name={name!r} rc={rc} elapsed={elapsed:.2f}s")
            print(f"[result] created={created} total_bytes={total_bytes}")
            if (cands := cxl.debug_candidate_slots(name, max_out=64)) is not None:
                used = sum(u for _, u in cands)
                print(f"[debug] candidate_slots={len(cands)} used={used}")
                print("[debug] idx,in_use:", " ".join(f"{idx}:{u}" for idx, u in cands))
            print(
                "[hint] If stderr shows 'No free slots available for shared objects', "
                "this is slot/hash-table exhaustion (not DAX size)."
            )
            print(
                "[hint] If stderr shows 'Object with name ... already exists', "
                "you likely didn't reset and are reusing names."
            )
            print(
                "[hint] If the process aborts with 'Not enough space in DAX device', "
                "you hit the hardcoded 32GB mapping limit."
            )
            return 3

        # Keep handles open? Not necessary for slot accounting; but close anyway.
        cxl.close(hnd)
        created += 1
        total_bytes += args.obj_size

        if args.progress_every and created % args.progress_every == 0:
            elapsed = time.time() - start
            rate = created / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] created={created} total_bytes={total_bytes} "
                f"rate={rate:.1f} objs/s"
            )

    elapsed = time.time() - start
    print()
    print("[result] OK")
    print(f"[result] created={created} total_bytes={total_bytes} elapsed={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

