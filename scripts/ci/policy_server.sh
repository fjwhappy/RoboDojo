#!/usr/bin/env bash
# Deploy an XPolicyLab policy server (default Pi_05) as a detached Docker container
# on a GPU runner, reusing cached image / checkpoint / model data. Used by
# .github/workflows/robodojo-policy-server.yml; runnable locally with the same env vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m[WARNING] $*\e[0m" >&2; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash scripts/ci/policy_server.sh <command> [--dry-run]

Commands:
  status             Print server container info + listening address (exit 0 if running+listening, 3 otherwise)
  ensure-image       Build the policy image unless it already exists (or REBUILD_IMAGE=true)
  ensure-checkpoint  Download the published checkpoint unless it is already cached
  start              Start the server container detached and wait until the port listens;
                     if it is already running, print status and return
  stop               Remove the server container

Environment (defaults in brackets):
  POLICY_NAME            XPolicyLab policy dir name [Pi_05]
  CKPT                   checkpoint run dir under checkpoints/ [RoboDojo-sim-arx_x5-joint-0]
  CKPT_REMOTE_POLICY     remote folder under ckpt/RoboDojo/ on the HF dataset [POLICY_NAME]
  TASK                   task passed to the server [stack_bowls]
  POLICY_PORT            listening port [9999]
  BIND_HOST              bind address inside the container [0.0.0.0]
  POLICY_GPU             host GPU index for the server [1]
  ACTION_TYPE            [joint]   ENV_CFG [arx_x5]   SEED [0]
  POLICY_IMAGE           image ref [robodojo-policy-<policy lowercase>:latest]
  POLICY_CONTAINER       container name [robodojo-policy-<policy>]
  REBUILD_IMAGE          true|false [false]
  RESTART                true|false: replace a running server [false]
  ROBODOJO_POLICY_CACHE_DIR  persistent cache root [~/robodojo-policy-cache]
                         checkpoints -> <root>/checkpoints/<policy>/<ckpt>
                         model data  -> <root>/model-data/<policy> (OPENPI_DATA_HOME etc.)
  ROBODOJO_HF_ENDPOINT   HF endpoint for checkpoint download [https://huggingface.co]
  ROBODOJO_POLICY_BUILD_ARGS  extra `docker build` args (mirrors, base image), whitespace-separated
  SERVER_START_TIMEOUT   seconds to wait for the port after start [900]
EOF
}

DRY_RUN="false"; COMMAND=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN="true" ;;
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; error "unknown option: ${arg}" ;;
    *) [[ -z "${COMMAND}" ]] || error "only one command allowed"; COMMAND="${arg}" ;;
  esac
done
[[ -n "${COMMAND}" ]] || { usage; exit 2; }

POLICY_NAME="${POLICY_NAME:-Pi_05}"
CKPT="${CKPT:-RoboDojo-sim-arx_x5-joint-0}"
CKPT_REMOTE_POLICY="${CKPT_REMOTE_POLICY:-${POLICY_NAME}}"
TASK="${TASK:-stack_bowls}"
POLICY_PORT="${POLICY_PORT:-9999}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
POLICY_GPU="${POLICY_GPU:-1}"
ACTION_TYPE="${ACTION_TYPE:-joint}"
ENV_CFG="${ENV_CFG:-arx_x5}"
SEED="${SEED:-0}"
POLICY_IMAGE="${POLICY_IMAGE:-robodojo-policy-${POLICY_NAME,,}:latest}"
POLICY_CONTAINER="${POLICY_CONTAINER:-robodojo-policy-${POLICY_NAME}}"
CACHE_ROOT="${ROBODOJO_POLICY_CACHE_DIR:-${HOME}/robodojo-policy-cache}"
CKPT_CACHE="${CACHE_ROOT}/checkpoints/${POLICY_NAME}"
DATA_CACHE="${CACHE_ROOT}/model-data/${POLICY_NAME}"

for v in POLICY_NAME CKPT CKPT_REMOTE_POLICY TASK ACTION_TYPE ENV_CFG; do
  [[ "${!v}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || error "invalid ${v}: '${!v}'"
done
for v in POLICY_PORT POLICY_GPU SEED; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || error "invalid ${v}: '${!v}'"
done

run() {
  if [[ "${DRY_RUN}" == "true" ]]; then printf '[dry-run]'; printf ' %q' "$@"; printf '\n'; return 0; fi
  "$@"
}

state() { docker inspect -f '{{.State.Status}}' "${POLICY_CONTAINER}" 2>/dev/null || true; }

listening() {
  local probe="127.0.0.1"
  [[ "${BIND_HOST}" == "0.0.0.0" ]] || probe="${BIND_HOST}"
  timeout 3 bash -c ">/dev/tcp/${probe}/$1" 2>/dev/null
}

host_ips() { hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9.]+$' | head -3 | paste -sd, -; }

# Port the running container was started with (label), falling back to POLICY_PORT.
container_port() {
  docker inspect -f '{{index .Config.Labels "robodojo.policy.port"}}' "${POLICY_CONTAINER}" 2>/dev/null || true
}

print_address() {
  local port="$1" bind="$2"
  echo "================ policy server ================"
  echo "  container : ${POLICY_CONTAINER}"
  docker inspect -f '  image     : {{.Config.Image}}{{"\n"}}  started   : {{.State.StartedAt}}{{"\n"}}  status    : {{.State.Status}}' "${POLICY_CONTAINER}" 2>/dev/null || true
  docker inspect -f '  policy    : {{index .Config.Labels "robodojo.policy"}}  ckpt: {{index .Config.Labels "robodojo.policy.ckpt"}}  task: {{index .Config.Labels "robodojo.policy.task"}}  gpu: {{index .Config.Labels "robodojo.policy.gpu"}}' "${POLICY_CONTAINER}" 2>/dev/null || true
  echo "  listening : ${bind}:${port}   (protocol: ws)"
  echo "  reach it  : 127.0.0.1:${port} on this host; $(host_ips | sed "s/,/:${port}, /g"):${port} from other machines"
  echo "  client    : policy_host=127.0.0.1 policy_port=${port}"
  echo "  logs      : docker logs -f ${POLICY_CONTAINER}"
  echo "==============================================="
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    { echo "address=${bind}"; echo "port=${port}"; echo "container=${POLICY_CONTAINER}"; } >> "${GITHUB_OUTPUT}"
  fi
}

cmd_status() {
  local st port bind
  st="$(state)"
  if [[ -z "${st}" ]]; then info "No policy server container ${POLICY_CONTAINER}"; return 3; fi
  port="$(container_port)"; port="${port:-${POLICY_PORT}}"
  bind="$(docker inspect -f '{{index .Config.Labels "robodojo.policy.bind"}}' "${POLICY_CONTAINER}" 2>/dev/null || true)"
  bind="${bind:-${BIND_HOST}}"
  if [[ "${st}" == "running" ]] && listening "${port}"; then
    info "Policy server ${POLICY_CONTAINER} is running and listening"
    print_address "${port}" "${bind}"
    return 0
  fi
  warn "Policy server ${POLICY_CONTAINER} is '${st}' but ${port} is not accepting connections"
  docker logs --tail 30 "${POLICY_CONTAINER}" 2>&1 || true
  return 3
}

cmd_ensure_image() {
  if [[ "${REBUILD_IMAGE:-false}" != "true" ]] && docker image inspect "${POLICY_IMAGE}" >/dev/null 2>&1; then
    info "Image ${POLICY_IMAGE} already present ($(docker image inspect -f '{{.Created}}' "${POLICY_IMAGE}")); skipping build"
    return 0
  fi
  [[ -d "${ROOT_DIR}/XPolicyLab/policy/${POLICY_NAME}" ]] || \
    error "XPolicyLab/policy/${POLICY_NAME} not found (submodules missing?)"
  local build_args=()
  [[ -z "${ROBODOJO_POLICY_BUILD_ARGS:-}" ]] || read -r -a build_args <<< "${ROBODOJO_POLICY_BUILD_ARGS}"
  info "Building ${POLICY_IMAGE} for ${POLICY_NAME}"
  run docker build "${build_args[@]}" --build-arg "POLICY=${POLICY_NAME}" \
    -f "${ROOT_DIR}/docker/policy/Dockerfile" -t "${POLICY_IMAGE}" "${ROOT_DIR}"
}

cmd_ensure_checkpoint() {
  local dl=("${SCRIPT_DIR}/download_checkpoint.py" --policy "${CKPT_REMOTE_POLICY}" --run "${CKPT}" --dest "${CKPT_CACHE}")
  if [[ "${DRY_RUN}" == "true" ]]; then run python3 "${dl[@]}"; return 0; fi
  if python3 "${dl[@]}" --check-only; then return 0; fi
  if [[ -d "${CKPT_CACHE}/${CKPT}" ]] && find "${CKPT_CACHE}/${CKPT}" -maxdepth 2 -type d -name params | grep -q . \
      && [[ "${CKPT_SKIP_VERIFY:-false}" == "true" ]]; then
    info "Using existing checkpoint at ${CKPT_CACHE}/${CKPT} (CKPT_SKIP_VERIFY=true)"
    return 0
  fi
  mkdir -p "${CKPT_CACHE}"
  python3 "${dl[@]}" --endpoint "${ROBODOJO_HF_ENDPOINT:-https://huggingface.co}"
}

cmd_stop() { run docker rm -f "${POLICY_CONTAINER}" >/dev/null 2>&1 || true; info "Removed ${POLICY_CONTAINER} (if it existed)"; }

cmd_start() {
  local st
  st="$(state)"
  if [[ "${RESTART:-false}" == "true" && -n "${st}" ]]; then
    info "RESTART=true: replacing ${POLICY_CONTAINER} (${st})"; cmd_stop; st=""
  fi
  if [[ "${st}" == "running" ]]; then
    if cmd_status; then return 0; fi
    info "Container is running but not listening yet; waiting"
  else
    [[ -z "${st}" ]] || { info "Removing previous ${st} container"; run docker rm -f "${POLICY_CONTAINER}" >/dev/null; }
    if [[ "${DRY_RUN}" != "true" ]]; then
      [[ -f "${CKPT_CACHE}/${CKPT}/.robodojo_ckpt_complete" || -d "${CKPT_CACHE}/${CKPT}" ]] || \
        error "checkpoint ${CKPT_CACHE}/${CKPT} missing (run ensure-checkpoint)"
      docker image inspect "${POLICY_IMAGE}" >/dev/null 2>&1 || error "image ${POLICY_IMAGE} missing (run ensure-image)"
      if listening "${POLICY_PORT}"; then error "port ${POLICY_PORT} is already in use by another process"; fi
      mkdir -p "${DATA_CACHE}"
    fi
    run docker run -d --name "${POLICY_CONTAINER}" --restart unless-stopped \
      --gpus "device=${POLICY_GPU}" --network host --ipc host \
      --label robodojo.policy="${POLICY_NAME}" --label robodojo.policy.ckpt="${CKPT}" \
      --label robodojo.policy.task="${TASK}" --label robodojo.policy.port="${POLICY_PORT}" \
      --label robodojo.policy.bind="${BIND_HOST}" --label robodojo.policy.gpu="${POLICY_GPU}" \
      -e XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
      -v "${CKPT_CACHE}:/workspace/RoboDojo/XPolicyLab/policy/${POLICY_NAME}/checkpoints:ro" \
      -v "${DATA_CACHE}:/root/.cache/openpi" \
      "${POLICY_IMAGE}" serve --task "${TASK}" --ckpt "${CKPT}" --port "${POLICY_PORT}" \
      --bind-host "${BIND_HOST}" --action-type "${ACTION_TYPE}" --env-cfg "${ENV_CFG}" --seed "${SEED}"
    [[ "${DRY_RUN}" == "true" ]] && return 0
  fi

  local deadline=$(( SECONDS + ${SERVER_START_TIMEOUT:-900} ))
  info "Waiting for ${POLICY_CONTAINER} to listen on ${POLICY_PORT} (model load can take minutes)"
  until listening "${POLICY_PORT}"; do
    st="$(state)"
    if [[ "${st}" != "running" ]]; then
      docker logs --tail 80 "${POLICY_CONTAINER}" 2>&1 || true
      error "policy server container is '${st:-gone}' before listening"
    fi
    if (( SECONDS >= deadline )); then
      docker logs --tail 80 "${POLICY_CONTAINER}" 2>&1 || true
      error "port ${POLICY_PORT} not listening after ${SERVER_START_TIMEOUT:-900}s (container left running)"
    fi
    sleep 10
  done
  info "Policy server is up"
  print_address "${POLICY_PORT}" "${BIND_HOST}"
}

case "${COMMAND}" in
  status) cmd_status ;;
  ensure-image) cmd_ensure_image ;;
  ensure-checkpoint) cmd_ensure_checkpoint ;;
  start) cmd_start ;;
  stop) cmd_stop ;;
  *) usage >&2; error "unknown command: ${COMMAND}" ;;
esac
