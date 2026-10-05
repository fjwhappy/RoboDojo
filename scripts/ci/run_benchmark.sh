#!/usr/bin/env bash
# Full RoboDojo benchmark orchestrator for one policy on one multi-GPU runner.
#
# One "worker" per GPU: a dedicated policy-server container (scripts/ci/policy_server.sh)
# plus a sequence of eval-client containers (scripts/ci/robodojo_ci.sh run-task) on the
# same GPU. Tasks are spread over workers by the embedded runtime weights (longest first).
# Everything is resumable: a stable run id per benchmark makes the eval client resume
# partially finished tasks, and tasks whose status.json is PASS with the full episode
# count are skipped on re-launch.
#
# Commands:
#   launch   start the orchestrator detached (no-op + status if it is already running)
#   status   print progress (exit 0 = finished, 10 = running, 11 = not running/incomplete)
#   run      orchestrate in the foreground (used by `launch`)
#   stop     stop the orchestrator and all its client containers (servers are kept)
#   summary  (re)build the summary from the results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m[WARNING] $*\e[0m" >&2; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m" >&2; exit 1; }

POLICY_NAME="${POLICY_NAME:-Pi_05}"
CKPT="${CKPT:-RoboDojo-sim-arx_x5-joint-0}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
ENV_CFG="${ENV_CFG:-arx_x5}"
SEED="${SEED:-0}"
EVAL_NUM="${EVAL_NUM:-native}"
BENCH_GPUS="${BENCH_GPUS:-1,2,3,4,5,6,7}"
BENCH_BASE_PORT="${BENCH_BASE_PORT:-10000}"
BENCH_TASKS="${BENCH_TASKS:-}"              # comma list; empty = all runnable tasks
BENCH_DIMENSION="${BENCH_DIMENSION:-}"
BENCH_TAG="${BENCH_TAG:-${POLICY_NAME}-${CKPT}-seed${SEED}}"
BENCH_ROOT="${BENCH_ROOT:-${HOME}/robodojo-bench}/${BENCH_TAG}"
TASK_ATTEMPTS="${TASK_ATTEMPTS:-4}"
BOOT_GAP="${BOOT_GAP:-90}"                 # seconds between eval-client boots across workers
export SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"

for v in POLICY_NAME CKPT ACTION_TYPE ENV_CFG BENCH_TAG; do
  [[ "${!v}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || error "invalid ${v}: '${!v}'"
done
[[ "${SEED}" =~ ^[0-9]+$ ]] || error "invalid SEED"
[[ "${EVAL_NUM}" == native || "${EVAL_NUM}" =~ ^[0-9]+$ ]] || error "invalid EVAL_NUM"
[[ "${BENCH_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || error "invalid BENCH_GPUS"

RESULTS="${BENCH_ROOT}/eval_result"
STATE="${BENCH_ROOT}/state"
PIDFILE="${STATE}/orchestrator.pid"
LOG="${BENCH_ROOT}/orchestrator.log"
RUN_ID="bench-${BENCH_TAG}"

orchestrator_running() {
  [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null
}

# Expected episodes per task (native counts from _task.yml, or EVAL_NUM).
expected_json() {
  python3 - "${REPO_DIR}" "${EVAL_NUM}" "$@" <<'PY'
import json, sys
from pathlib import Path
import yaml
repo, eval_num, *tasks = sys.argv[1:]
cfg = yaml.safe_load(Path(repo, "task/RoboDojo/config/_task.yml").read_text())
common = int(cfg["common"].get("eval_nums", 50))
out = {}
for t in tasks:
    native = int((cfg.get("tasks") or {}).get(t, {}).get("eval_nums", common))
    out[t] = native if eval_num == "native" else min(native, int(eval_num))
print(json.dumps(out))
PY
}

task_list() {
  local args=(--format plain --only-runnable)
  [[ -z "${BENCH_DIMENSION}" ]] || args+=(--dimension "${BENCH_DIMENSION}")
  if [[ -n "${BENCH_TASKS}" ]]; then
    tr ',' '\n' <<< "${BENCH_TASKS}" | sed '/^$/d'
  else
    python3 "${REPO_DIR}/scripts/internal/task_inventory.py" "${args[@]}"
  fi
}

# Greedy longest-processing-time split using RUNTIME_WEIGHTS from smoke_all_tasks.sh.
partition() {
  local n="$1"; shift
  python3 - "${REPO_DIR}/scripts/internal/smoke_all_tasks.sh" "${ENV_CFG}" "${n}" "$@" <<'PY'
import re, sys
src, env_cfg, n, *tasks = sys.argv[1:]
n = int(n)
weights = {k: int(v) for k, v in re.findall(r'"(\w+)/' + re.escape(env_cfg) + r'": (\d+)', open(src).read())}
default = sorted(weights.values())[len(weights) // 2] if weights else 1
loads, groups = [0] * n, [[] for _ in range(n)]
for t in sorted(tasks, key=lambda t: -weights.get(t, default)):
    i = min(range(n), key=lambda i: loads[i])
    groups[i].append(t)
    loads[i] += weights.get(t, default)
for g in groups:
    print(",".join(g))
PY
}

task_done() {  # task_done TASK EXPECTED -> 0 if PASS with the full episode count
  local f="${RESULTS}/_ci/$1/status.json"
  [[ -f "${f}" ]] || return 1
  python3 - "${f}" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if d.get("status") == "PASS" and int(d.get("eval_time") or 0) >= int(sys.argv[2]) else 1)
PY
}

common_env() {
  export ROBODOJO_OUTPUT_DIR="${RESULTS}"
  export ROBODOJO_RUN_ID="${RUN_ID}"
  export ROBODOJO_MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-5}"
  export POLICY_NAME CKPT ACTION_TYPE ENV_CFG SEED
}

worker() {  # worker GPU PORT TASKS_CSV EXPECTED_JSON STAGGER_SECONDS
  local gpu="$1" port="$2" tasks_csv="$3" expected="$4" stagger="${5:-0}" task exp attempt
  common_env
  # Stagger Isaac Sim / model start-up across workers (concurrent Kit boots are racy).
  sleep "${stagger}"
  IFS=',' read -r -a tasks <<< "${tasks_csv}"
  for task in "${tasks[@]}"; do
    [[ -n "${task}" ]] || continue
    exp="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "${expected}" "${task}")"
    if task_done "${task}" "${exp}"; then
      echo "[worker gpu${gpu}] SKIP ${task} (already complete)"; continue
    fi
    for (( attempt = 1; attempt <= TASK_ATTEMPTS; attempt++ )); do
      if [[ -f "${STATE}/stop" ]]; then echo "[worker gpu${gpu}] stop requested"; return 0; fi
      # Make sure this worker's policy server is up (restart it if it died).
      POLICY_CONTAINER="robodojo-policy-${POLICY_NAME}-g${gpu}" POLICY_PORT="${port}" POLICY_GPU="${gpu}" \
        TASK="${task}" bash "${SCRIPT_DIR}/policy_server.sh" start > "${STATE}/server-g${gpu}.log" 2>&1 || {
          echo "[worker gpu${gpu}] policy server failed to start (attempt ${attempt})"; tail -20 "${STATE}/server-g${gpu}.log"; sleep 30; continue; }
      # Serialize Isaac Sim boots across workers: concurrent Kit start-ups race
      # ("Accessed invalid null prim" / "Simulation context already exists").
      exec 9>"${STATE}/boot.lock"; flock 9
      echo "[worker gpu${gpu}] RUN ${task} (expected ${exp} episodes, attempt ${attempt}) $(date -Is)"
      TASK="${task}" POLICY_HOST=127.0.0.1 POLICY_PORT="${port}" ENV_GPU="${gpu}" EVAL_NUM="${exp}" \
        ROBODOJO_CONTAINER_NAME="robodojo-bench-${POLICY_NAME}-${task}" \
        bash "${SCRIPT_DIR}/robodojo_ci.sh" run-task > "${STATE}/task-${task}.log" 2>&1 9>&- &
      local run_pid=$! run_rc=0
      sleep "${BOOT_GAP}"; flock -u 9; exec 9>&-
      wait "${run_pid}" || run_rc=$?
      if (( run_rc == 0 )); then
        cp -f "${RESULTS}/_ci/${task}/client.log" "${STATE}/client-${task}-a${attempt}.log" 2>/dev/null || true
        if task_done "${task}" "${exp}"; then
          echo "[worker gpu${gpu}] PASS ${task} $(date -Is)"; break
        fi
        echo "[worker gpu${gpu}] PARTIAL ${task}: fewer than ${exp} episodes; resuming"
      else
        cp -f "${RESULTS}/_ci/${task}/client.log" "${STATE}/client-${task}-a${attempt}.log" 2>/dev/null || true
        echo "[worker gpu${gpu}] FAIL ${task} (attempt ${attempt}): $(grep -E '\[ERROR\]' "${STATE}/task-${task}.log" | tail -1)"
        sleep 20
      fi
    done
  done
  echo "[worker gpu${gpu}] finished $(date -Is)"
}

cmd_summary() {
  local md="${BENCH_ROOT}/summary.md"
  {
    echo "# RoboDojo benchmark: ${POLICY_NAME} / ${CKPT} / seed ${SEED}"
    echo
    ROBODOJO_REPORT_DIR="${RESULTS}" ROBODOJO_OUTPUT_DIR="${RESULTS}" bash "${SCRIPT_DIR}/robodojo_ci.sh" report
    echo
    python3 - "${RESULTS}" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "_ci", "*", "status.json"))):
    d = json.load(open(f))
    if d.get("eval_time"):
        rows.append(d)
if rows:
    n = sum(int(r["eval_time"]) for r in rows)
    sr = sum(float(r["success_rate"]) * int(r["eval_time"]) for r in rows) / n * 100
    sc = sum(float(r["score"]) for r in rows) / len(rows)
    print(f"**Overall (episode-weighted) success rate: {sr:.1f}% over {n} episodes; mean task score {sc:.1f}**")
PY
  } > "${md}"
  if ROBODOJO_EVAL_ROOT="${RESULTS}/RoboDojo" python3 "${REPO_DIR}/scripts/internal/summarize_result.py" > "${STATE}/summarize.log" 2>&1; then
    { echo; echo "---"; echo; cat "${RESULTS}/RoboDojo/_summary.md"; } >> "${md}"
  fi
  cat "${md}"
}

cmd_status() {
  local tasks expected total done=0 running
  mapfile -t tasks < <(task_list)
  expected="$(expected_json "${tasks[@]}")"
  total="${#tasks[@]}"
  for t in "${tasks[@]}"; do
    task_done "${t}" "$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "${expected}" "${t}")" && done=$((done + 1))
  done
  running="$(docker ps --filter "name=robodojo-bench-${POLICY_NAME}-" --format '{{.Names}} ({{.RunningFor}})' | sed "s/robodojo-bench-${POLICY_NAME}-//" | paste -sd ';' -)"
  echo "benchmark : ${BENCH_TAG}"
  echo "results   : ${RESULTS}"
  echo "progress  : ${done}/${total} tasks complete"
  echo "running   : ${running:-none}"
  if orchestrator_running; then
    echo "state     : RUNNING (pid $(cat "${PIDFILE}")), log ${LOG}"
    tail -5 "${LOG}" 2>/dev/null | sed 's/^/  | /'
    return 10
  fi
  if (( done == total )); then echo "state     : FINISHED"; return 0; fi
  echo "state     : NOT RUNNING (incomplete)"; return 11
}

cmd_run() {
  unset GITHUB_OUTPUT GITHUB_ENV GITHUB_STEP_SUMMARY GITHUB_PATH GITHUB_STATE
  mkdir -p "${RESULTS}" "${STATE}"
  rm -f "${STATE}/stop"
  echo $$ > "${PIDFILE}"
  trap 'rm -f "${PIDFILE}"' EXIT
  local tasks gpus groups expected i
  mapfile -t tasks < <(task_list)
  IFS=',' read -r -a gpus <<< "${BENCH_GPUS}"
  expected="$(expected_json "${tasks[@]}")"
  mapfile -t groups < <(partition "${#gpus[@]}" "${tasks[@]}")
  info "benchmark ${BENCH_TAG}: ${#tasks[@]} tasks on GPUs ${BENCH_GPUS} -> ${RESULTS}"
  local pids=()
  for i in "${!gpus[@]}"; do
    echo "[orchestrator] gpu${gpus[i]} port $((BENCH_BASE_PORT + gpus[i])): ${groups[i]}"
    worker "${gpus[i]}" "$((BENCH_BASE_PORT + gpus[i]))" "${groups[i]}" "${expected}" "$((i * ${BENCH_STAGGER:-60}))" &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "${p}" || true; done
  info "all workers finished $(date -Is)"
  cmd_summary
}

cmd_launch() {
  if orchestrator_running; then
    info "Benchmark ${BENCH_TAG} already running"
    cmd_status || true
    return 0
  fi
  mkdir -p "${BENCH_ROOT}" "${STATE}"
  # Snapshot the scripts so the run survives CI workspace cleanups.
  local snap="${BENCH_ROOT}/repo"
  rm -rf "${snap}"; mkdir -p "${snap}/task/RoboDojo"
  cp -r "${REPO_DIR}/scripts" "${REPO_DIR}/utils" "${snap}/"
  cp -r "${REPO_DIR}/task/RoboDojo/config" "${REPO_DIR}/task/RoboDojo/tasks" "${REPO_DIR}/task/RoboDojo/task_registry.py" "${snap}/task/RoboDojo/" 2>/dev/null || true
  touch "${snap}/task/__init__.py" "${snap}/task/RoboDojo/__init__.py"
  # RUNNER_TRACKING_ID= keeps the GitHub runner from killing it when the job ends.
  # The orchestrator outlives the CI job: drop the job's GITHUB_* file handles.
  RUNNER_TRACKING_ID="" setsid nohup env -u GITHUB_OUTPUT -u GITHUB_ENV -u GITHUB_STEP_SUMMARY -u GITHUB_PATH -u GITHUB_STATE \
    POLICY_NAME="${POLICY_NAME}" CKPT="${CKPT}" ACTION_TYPE="${ACTION_TYPE}" ENV_CFG="${ENV_CFG}" SEED="${SEED}" \
    EVAL_NUM="${EVAL_NUM}" BENCH_GPUS="${BENCH_GPUS}" BENCH_BASE_PORT="${BENCH_BASE_PORT}" \
    BENCH_TASKS="${BENCH_TASKS}" BENCH_DIMENSION="${BENCH_DIMENSION}" BENCH_TAG="${BENCH_TAG}" \
    BENCH_ROOT="$(dirname "${BENCH_ROOT}")" TASK_ATTEMPTS="${TASK_ATTEMPTS}" \
    bash "${snap}/scripts/ci/run_benchmark.sh" run >> "${LOG}" 2>&1 < /dev/null &
  sleep 5
  orchestrator_running || { tail -30 "${LOG}"; error "orchestrator failed to start"; }
  info "Benchmark ${BENCH_TAG} launched in background (pid $(cat "${PIDFILE}"))"
  echo "  log     : ${LOG}"
  echo "  results : ${RESULTS}"
  echo "  summary : ${BENCH_ROOT}/summary.md (written when finished)"
}

cmd_stop() {
  mkdir -p "${STATE}"; touch "${STATE}/stop"
  if orchestrator_running; then kill -- -"$(ps -o pgid= "$(cat "${PIDFILE}")" | tr -d ' ')" 2>/dev/null || true; fi
  docker ps -q --filter "name=robodojo-bench-${POLICY_NAME}-" | xargs -r docker rm -f >/dev/null
  rm -f "${PIDFILE}"; info "stopped ${BENCH_TAG} (policy servers left running)"
}

case "${1:-}" in
  launch) cmd_launch ;;
  status) cmd_status ;;
  run) cmd_run ;;
  stop) cmd_stop ;;
  summary) cmd_summary ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
