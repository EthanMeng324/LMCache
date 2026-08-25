#!/usr/bin/env bash
# Verify the GPU->CPU->CXL tier-aware routing + prefetch pipeline.
#
# Three stages:
#   A) automated  - LMCache emits tier-tagged KV events (runs a pytest here)
#   B) manual     - Dynamo's router ingests them (grep the scheduler log)
#   C) manual     - multi-turn stickiness + prefetch (grep the worker logs)
#
# Stages B/C need a LIVE Dynamo deployment, so this script runs A and prints
# ready-to-paste commands / checks for B and C.
#
# Usage:
#   ./verify_cxl_strata.sh                 # run stage A, print B/C checklist
#   ./verify_cxl_strata.sh --scheduler-log /path/to/frontend.log   # also grep B
#   ./verify_cxl_strata.sh --worker-log    /path/to/workerA.log     # also grep C
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHED_LOG=""
WORKER_LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scheduler-log) SCHED_LOG="$2"; shift 2 ;;
    --worker-log)    WORKER_LOG="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; }

# --------------------------------------------------------------------------- #
bold "== Preflight: required switches =="

# PYTHONHASHSEED — required for multi-node + shared CXL hash consistency.
if [[ -n "${PYTHONHASHSEED:-}" ]]; then
  ok "PYTHONHASHSEED=$PYTHONHASHSEED"
else
  warn "PYTHONHASHSEED is NOT set. REQUIRED on every node for multi-node + shared CXL."
  warn "  export PYTHONHASHSEED=0   (use the SAME value on all nodes)"
fi

cat <<'EOF'

  Switches this pipeline needs (set these in your deployment):
    LMCache config:   enable_kv_events: true          (default false)
    Dynamo frontend:  --router-mode kv-strata         (default round-robin)
                      [--strata-cpu-overlap-weight 0.9]
                      [--strata-cxl-overlap-weight 0.8]
                      [--router-temperature 0]         (0 = deterministic/sticky)
    Prefetch (LMCache extra_config):
                      cxl_prefetch_enabled: true
EOF

# --------------------------------------------------------------------------- #
bold ""
bold "== Stage A (automated): LMCache emits tier-tagged KV events =="

PYTEST="${PYTEST:-pytest}"
if command -v "$PYTEST" >/dev/null 2>&1; then
  ( cd "$REPO_DIR" && PYTHONHASHSEED="${PYTHONHASHSEED:-0}" \
      "$PYTEST" -q tests/v1/test_cxl_strata_pipeline.py -s )
  A_RC=$?
  if [[ $A_RC -eq 0 ]]; then
    ok "Stage A passed: medium=CPU and medium=CXL events emitted with token_ids."
  else
    fail "Stage A failed (rc=$A_RC). LMCache is not emitting healthy tier events;"
    fail "fix this before looking at the router (B/C would fail downstream)."
  fi
else
  warn "pytest not found; skipping stage A. Run it manually:"
  warn "  PYTHONHASHSEED=0 pytest -q tests/v1/test_cxl_strata_pipeline.py -s"
fi

# --------------------------------------------------------------------------- #
bold ""
bold "== Stage B (needs live Dynamo): router ingests per-tier scores =="
cat <<'EOF'
  The scheduler logs its logit formula per worker. In kv-strata mode a worker
  whose prefix is only in CPU/CXL must get a NON-ZERO effective overlap.

  Look for lines like:
    "Formula for worker_id=... = <w> * prefill_blocks + decode_blocks = ..."
  and confirm workers with no GPU hit still get credited (cpu/cxl scores > 0).
EOF
if [[ -n "$SCHED_LOG" ]]; then
  if [[ -f "$SCHED_LOG" ]]; then
    echo "  --- grep '$SCHED_LOG' ---"
    grep -E "Formula for worker_id|CPU scores|cxl_scores|final GPU scores" "$SCHED_LOG" | tail -20
    if grep -qE "CPU scores=\{[^}]|cxl_scores" "$SCHED_LOG"; then
      ok "Found non-empty CPU/CXL score lines -> router is seeing lower tiers."
    else
      warn "No CPU/CXL score lines found. Is --router-mode kv-strata set AND"
      warn "enable_kv_events=true on the workers?"
    fi
  else
    warn "scheduler log not found: $SCHED_LOG"
  fi
else
  echo "  (pass --scheduler-log <file> to auto-grep this)"
fi

# --------------------------------------------------------------------------- #
bold ""
bold "== Stage C (needs live Dynamo): multi-turn stickiness + prefetch =="
cat <<'EOF'
  Drive a multi-turn conversation (same prefix across turns) and confirm:
   1. turn N+1 is routed BACK to the node that served turn N
      (its cpu/cxl overlap keeps it "sticky");
   2. that node serves the prefix from CPU, not CXL, thanks to the prefetch.

  On the prefix-holding worker's log, the prefetch should fire:
    "CXL predictive prefetch enabled ..."     (at startup)
    "CXL predictive prefetch promoted N chunk(s) to LocalCPUBackend."
  and the follow-up retrieve should be the fast CPU path (compare the
  "Retrieved ... cost X ms" lines: CXL turn ~200ms vs CPU turn ~10ms in our
  single-node measurement).
EOF
if [[ -n "$WORKER_LOG" ]]; then
  if [[ -f "$WORKER_LOG" ]]; then
    echo "  --- grep '$WORKER_LOG' ---"
    grep -E "predictive prefetch (enabled|promoted)|Retrieved [0-9]+ out of" "$WORKER_LOG" | tail -20
    if grep -qE "predictive prefetch promoted" "$WORKER_LOG"; then
      ok "Prefetch is firing on this worker."
    else
      warn "No 'promoted' lines. Is cxl_prefetch_enabled=true, and did a"
      warn "continuation (hit >= min_hit_tokens) actually occur on this node?"
    fi
  else
    warn "worker log not found: $WORKER_LOG"
  fi
else
  echo "  (pass --worker-log <file> to auto-grep this)"
fi

bold ""
bold "Done. Stage A is decisive for the LMCache side; B/C confirm the router side"
bold "on a live cluster. See CXL_PREFETCH_STRATA_SETUP.md for the full rationale."
