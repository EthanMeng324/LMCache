#!/usr/bin/env bash
# Run a real ShareGPT-format service trace through a live OpenAI-compatible
# Dynamo/vLLM endpoint with LMCache global prefetch enabled.
#
# This script intentionally does not start or stop the model service: service
# startup is deployment-specific.  It verifies the endpoint, preserves a
# stable LMCache session id, runs the trace, and writes a reproducible report.
#
# Example:
#   ./run_global_prefetch_realtrace.sh \
#     --base-url http://127.0.0.1:8000/v1 \
#     --model Qwen2.5-3B-Instruct \
#     --trace ShareGPT.json \
#     --trace-limit 20 --num-rounds 3 --answer-len 128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DYNAMO_DIR="$ROOT_DIR/dynamo"
PYTHON_BIN="${PYTHON:-$DYNAMO_DIR/.venv/bin/python}"

BASE_URL=""
MODEL=""
TRACE="$SCRIPT_DIR/ShareGPT.json"
TRACE_LIMIT=""
NUM_USERS=1
TOTAL_USERS=1
NUM_ROUNDS=3
QPS=0.05
ANSWER_LEN=128
OUTPUT_DIR="/tmp/lmcache-global-prefetch-realtrace-$(date +%Y%m%d-%H%M%S)"
ROUTER_LOG=""
WORKER_LOG=""
SKIP_HEALTH=0
DRY_RUN=0

usage() {
  sed -n '1,24p' "$0"
  cat <<'EOF'

Required:
  --base-url URL       OpenAI-compatible URL, usually http://host:8000/v1
  --model NAME         Served model name

Trace/workload:
  --trace FILE         ShareGPT-format JSON (default: ShareGPT.json)
  --trace-limit N      Maximum conversations loaded from FILE
  --num-users N        Concurrent sessions (default: 1)
  --total-users N      Sessions before exit (default: 1)
  --num-rounds N       Conversation rounds (default: 3)
  --qps QPS            Aggregate request rate (default: 0.05)
  --answer-len N       Maximum generated tokens (default: 128)

Diagnostics:
  --output-dir DIR     Report directory
  --router-log FILE    Optional Dynamo frontend log to summarize
  --worker-log FILE    Optional LMCache worker log to summarize
  --skip-health        Do not call GET /models before replay
  --dry-run            Validate trace/workload flow without sending requests
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --trace) TRACE="$2"; shift 2 ;;
    --trace-limit) TRACE_LIMIT="$2"; shift 2 ;;
    --num-users) NUM_USERS="$2"; shift 2 ;;
    --total-users) TOTAL_USERS="$2"; shift 2 ;;
    --num-rounds) NUM_ROUNDS="$2"; shift 2 ;;
    --qps) QPS="$2"; shift 2 ;;
    --answer-len) ANSWER_LEN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --router-log) ROUTER_LOG="$2"; shift 2 ;;
    --worker-log) WORKER_LOG="$2"; shift 2 ;;
    --skip-health) SKIP_HEALTH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$MODEL" ]]; then
  echo "--base-url and --model are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "No usable Python found; set PYTHON=/path/to/dynamo/.venv/bin/python" >&2
  exit 1
fi
if [[ ! -f "$TRACE" ]]; then
  echo "trace file not found: $TRACE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
BASE_URL="${BASE_URL%/}"
export PYTHONHASHSEED=0
export PYTHONPATH="$ROOT_DIR/LMCache${PYTHONPATH:+:$PYTHONPATH}"

echo "== Global prefetch real-trace run =="
echo "python:       $PYTHON_BIN"
echo "base URL:     $BASE_URL"
echo "model:        $MODEL"
echo "trace:        $TRACE"
echo "trace limit:  ${TRACE_LIMIT:-unlimited}"
echo "output dir:   $OUTPUT_DIR"
echo "session IDs:  enabled"

if [[ "${DYN_GLOBAL_PREFETCH_ENABLED:-}" != "true" ]]; then
  echo "WARNING: DYN_GLOBAL_PREFETCH_ENABLED is not true in this shell." >&2
  echo "         Make sure the Dynamo router was started with global prefetch enabled." >&2
fi
if [[ "${DYN_PREFETCH_HINT_ENDPOINT:-}" != *"/v1/cxl/prefetch"* ]]; then
  echo "WARNING: DYN_PREFETCH_HINT_ENDPOINT is not set to /v1/cxl/prefetch." >&2
fi

if [[ "$SKIP_HEALTH" -eq 0 ]]; then
  echo "Checking $BASE_URL/models ..."
  "$PYTHON_BIN" - "$BASE_URL/models" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read())
    models = payload.get("data", [])
    print(f"health: ok ({len(models)} model(s))")
except Exception as exc:
    raise SystemExit(f"health check failed for {url}: {exc}")
PY
fi

CSV_PATH="$OUTPUT_DIR/summary.csv"
RUN_LOG="$OUTPUT_DIR/benchmark.log"
META_PATH="$OUTPUT_DIR/run_metadata.txt"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "base_url=$BASE_URL"
  echo "model=$MODEL"
  echo "trace=$TRACE"
  echo "trace_limit=${TRACE_LIMIT:-}"
  echo "num_users=$NUM_USERS"
  echo "total_users=$TOTAL_USERS"
  echo "num_rounds=$NUM_ROUNDS"
  echo "qps=$QPS"
  echo "answer_len=$ANSWER_LEN"
  echo "PYTHONHASHSEED=$PYTHONHASHSEED"
} >"$META_PATH"

CMD=(
  "$PYTHON_BIN" "$SCRIPT_DIR/multi-round-qa.py"
  --num-users "$NUM_USERS"
  --total-users "$TOTAL_USERS"
  --shared-system-prompt 0
  --user-history-prompt 0
  --answer-len "$ANSWER_LEN"
  --num-rounds "$NUM_ROUNDS"
  --qps "$QPS"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --output "$CSV_PATH"
  --sharegpt
  --sharegpt-trace "$TRACE"
  --send-lmcache-session-id
  --request-with-user-id
  --enforce-strict-concurrent-users
)
if [[ -n "$TRACE_LIMIT" ]]; then
  CMD+=(--trace-limit "$TRACE_LIMIT")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run --disable-ramp-up)
fi

echo "Starting trace replay; log=$RUN_LOG"
(cd "$SCRIPT_DIR" && "${CMD[@]}" 2>&1 | tee "$RUN_LOG")

if [[ ! -s "$CSV_PATH" ]]; then
  echo "benchmark completed without a non-empty CSV: $CSV_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - "$CSV_PATH" "$OUTPUT_DIR/metrics.txt" <<'PY'
import math
import sys
import pandas as pd

csv_path, report_path = sys.argv[1:]
df = pd.read_csv(csv_path)
ttft = pd.to_numeric(df.get("ttft"), errors="coerce").dropna()
prompt = pd.to_numeric(df.get("prompt_tokens"), errors="coerce").dropna()
generation = pd.to_numeric(df.get("generation_tokens"), errors="coerce").dropna()
lines = [
    f"completed_requests={len(df)}",
    f"avg_ttft_seconds={ttft.mean() if len(ttft) else math.nan}",
    f"p50_ttft_seconds={ttft.quantile(0.50) if len(ttft) else math.nan}",
    f"p95_ttft_seconds={ttft.quantile(0.95) if len(ttft) else math.nan}",
    f"prompt_tokens={int(prompt.sum()) if len(prompt) else 0}",
    f"generation_tokens={int(generation.sum()) if len(generation) else 0}",
]
with open(report_path, "w", encoding="utf-8") as output:
    output.write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

for pair in "router:$ROUTER_LOG" "worker:$WORKER_LOG"; do
  name="${pair%%:*}"
  file="${pair#*:}"
  [[ -z "$file" ]] && continue
  if [[ ! -f "$file" ]]; then
    echo "WARNING: $name log not found: $file" >&2
    continue
  fi
  grep -Ei 'global prefetch|prefetch|cxl|cpu overlap|blockstored|used_on_time|unused' \
    "$file" >"$OUTPUT_DIR/${name}_prefetch_lines.log" || true
done

echo ""
echo "Reports:"
echo "  metadata: $META_PATH"
echo "  CSV:      $CSV_PATH"
echo "  metrics:  $OUTPUT_DIR/metrics.txt"
echo "  log:      $RUN_LOG"
echo "The CSV/metrics show request performance; router/worker logs show whether"
echo "global prefetch commands and CPU/CXL residency events were actually used."
