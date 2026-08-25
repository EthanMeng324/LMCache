#!/usr/bin/env bash
# Verify the router-owned global CXL prefetch implementation.
#
# The default run is safe for a live machine: it does not initialize/reset a
# CXL mapping and does not start Dynamo or LMCache services.  Hardware-backed
# checks are opt-in with --cxl-functional or --gpu-e2e.
#
# Usage:
#   ./LMCache/verify_global_prefetch.sh
#   ./LMCache/verify_global_prefetch.sh --cxl-functional
#   ./LMCache/verify_global_prefetch.sh --gpu-e2e
#
# Environment overrides:
#   PYTHON=/path/to/python       Python from the Dynamo uv environment
#   CXL_SIZE_GB=0.05             size for the opt-in tmpfs CXL test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LM_DIR="$ROOT_DIR/LMCache"
DYNAMO_DIR="$ROOT_DIR/dynamo"
PYTHON_BIN="${PYTHON:-$DYNAMO_DIR/.venv/bin/python}"
RUN_CXL=0
RUN_GPU=0

for arg in "$@"; do
  case "$arg" in
    --cxl-functional) RUN_CXL=1 ;;
    --gpu-e2e) RUN_GPU=1 ;;
    -h|--help)
      sed -n '1,22p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

green() { printf '\033[32m✓\033[0m %s\n' "$*"; }
yellow() { printf '\033[33m!\033[0m %s\n' "$*"; }
header() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# The project convention prefers rg, but keep this verifier usable on minimal
# service images where only POSIX grep is installed.
contains_text() {
  local pattern="$1" file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -q -- "$pattern" "$file"
  else
    grep -Eq -- "$pattern" "$file"
  fi
}

print_matches() {
  local pattern="$1" file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n -- "$pattern" "$file"
  else
    grep -nE -- "$pattern" "$file"
  fi
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "No usable Python found. Set PYTHON=/path/to/dynamo/.venv/bin/python" >&2
  exit 1
fi

header "Preflight"
echo "workspace: $ROOT_DIR"
echo "python:    $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import lmcache
print("lmcache:", lmcache.__file__)
try:
    import torch
    print("torch:", torch.__version__, "cuda_available=", torch.cuda.is_available())
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")
PY

if [[ -n "${PYTHONHASHSEED:-}" && "${PYTHONHASHSEED}" != 0 ]]; then
  yellow "PYTHONHASHSEED=${PYTHONHASHSEED}; multi-node deployments should use 0"
else
  export PYTHONHASHSEED=0
  green "PYTHONHASHSEED=0"
fi

header "Static architecture checks"
if print_matches '_CxlAssociationPrefetcher|_CxlPrefetcher' \
    "$LM_DIR/lmcache/v1/cache_engine.py"; then
  echo "worker-local predictor symbols are still present" >&2
  exit 1
fi
green "LMCache cache engine contains no worker-local predictor"
contains_text 'cxl_global_prefetch_enabled' \
  "$LM_DIR/benchmarks/multi_round_qa/lmcache-realtrace.yaml"
green "real-trace config uses the global prefetch switch"

header "Python syntax and control-message contract"
export PYTHONPATH="$LM_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m py_compile \
  "$LM_DIR/lmcache/v1/cache_engine.py" \
  "$LM_DIR/lmcache/v1/storage_backend/cxl_backend.py" \
  "$LM_DIR/lmcache/v1/storage_backend/local_cpu_backend.py" \
  "$LM_DIR/lmcache/v1/cache_controller/message.py" \
  "$LM_DIR/lmcache/v1/cache_controller/executor.py" \
  "$LM_DIR/lmcache/v1/cache_controller/controllers/kv_controller.py" \
  "$LM_DIR/lmcache/v1/cache_controller/controller_manager.py" \
  "$LM_DIR/lmcache/v1/cache_controller/worker.py" \
  "$LM_DIR/lmcache/v1/api_server/__main__.py"
"$PYTHON_BIN" - <<'PY'
import msgspec
from lmcache.v1.cache_controller.message import (
    Msg, PrefetchHintMsg, PrefetchStatusMsg, PrefetchStatusRetMsg,
    PrefetchStatusWorkerMsg, PrefetchStatusWorkerRetMsg,
)

messages = [
    PrefetchHintMsg(event_id="e", instance_id="i", tokens=[1, 2],
                    model_id="model", session_id="session", task_id="task"),
    PrefetchStatusMsg(event_id="e", instance_id="i", task_id="task"),
    PrefetchStatusRetMsg(event_id="e", task_id="task", state="READY"),
    PrefetchStatusWorkerMsg(worker_event_id="w", task_id="task"),
    PrefetchStatusWorkerRetMsg(worker_event_id="w", task_id="task", state="READY"),
]
for message in messages:
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(message), type=Msg)
    assert type(decoded) is type(message), (type(decoded), type(message))
print("message round-trip: ok")
PY
green "Python syntax and message round-trip"

header "Dynamo Rust checks"
(cd "$DYNAMO_DIR" && cargo check -p dynamo-llm --quiet)
(cd "$DYNAMO_DIR" && cargo test -p dynamo-llm global_prefetch --lib --quiet)
green "Dynamo global prefetch compiles and its unit tests pass"

header "LMCache predictor/control tests"
(cd "$ROOT_DIR" && "$PYTHON_BIN" -m pytest -q \
  LMCache/tests/v1/prefetch/test_association_predictor.py \
  LMCache/tests/v1/prefetch/test_prefetch_design.py)
green "LMCache predictor utilities and design contracts pass"

if [[ "$RUN_CXL" -eq 1 ]]; then
  header "Opt-in CXL functional test"
  echo "This test creates and resets its own /dev/shm mapping; size=${CXL_SIZE_GB:-0.05} GB"
  (cd "$LM_DIR" && "$PYTHON_BIN" tests/v1/storage_backend/test_cxl_prefetch.py \
    --size-gb "${CXL_SIZE_GB:-0.05}")
  green "CXL->CPU functional test"
else
  yellow "CXL functional test not run; pass --cxl-functional to enable it"
fi

if [[ "$RUN_GPU" -eq 1 ]]; then
  header "Opt-in GPU end-to-end test"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" -m pytest -q \
    LMCache/tests/v1/test_cxl_prefetch_e2e.py -s)
  green "GPU CXL promotion end-to-end test"
else
  yellow "GPU E2E test not run; pass --gpu-e2e to enable it"
fi

header "Result"
green "Global prefetch validation completed"
cat <<'EOF'

For a live service/ShareGPT validation, keep this script separate from the
deployment process. Start the Dynamo router and LMCache workers with:
  DYN_GLOBAL_PREFETCH_ENABLED=true
  DYN_PREFETCH_HINTS_ENABLED=true
  DYN_PREFETCH_HINT_ENDPOINT=http://<lmcache-api-host>:<port>/v1/cxl/prefetch
  # Single worker: use a template; multiple workers: use an explicit map.
  DYN_PREFETCH_HINT_INSTANCE_TEMPLATE=<instance-template>
  DYN_PREFETCH_HINT_INSTANCE_MAP=<dynamo-worker-id>=<lmcache-instance-id>,...
  DYN_GLOBAL_PREFETCH_CHUNK_SIZE=256
and use the migrated config:
  LMCache/benchmarks/multi_round_qa/lmcache-realtrace.yaml
EOF
