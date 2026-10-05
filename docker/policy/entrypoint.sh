#!/usr/bin/env bash
# Policy-server container entrypoint.
#   docker run ... robodojo-policy-<p> serve --task T --ckpt C [--port 9999] [--action-type joint] [--env-cfg arx_x5]
# Any other command is executed as-is from /workspace/RoboDojo.
set -euo pipefail
cd /workspace/RoboDojo

if [[ "${1:-}" != "serve" ]]; then
  exec "$@"
fi
shift

task="stack_bowls" ckpt="" port="9999" action_type="joint" env_cfg="arx_x5" seed="0" host="0.0.0.0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) task="$2"; shift 2 ;;
    --ckpt) ckpt="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --action-type) action_type="$2"; shift 2 ;;
    --env-cfg) env_cfg="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --bind-host) host="$2"; shift 2 ;;
    *) echo "[policy-entrypoint] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${ckpt}" ]] || { echo "[policy-entrypoint] --ckpt is required" >&2; exit 2; }

policy_dir="XPolicyLab/policy/${POLICY}"
# Policy env argument: uv-managed policies take "uv", conda ones the env name.
policy_env="uv"
[[ -d "${policy_dir}/openpi/.venv" || -d "${policy_dir}/.venv" ]] || policy_env="base"

echo "[policy-entrypoint] policy=${POLICY} task=${task} ckpt=${ckpt} bind=${host}:${port} action_type=${action_type}"
# GPU selection happens at `docker run --gpus device=N`, so inside the container it is GPU 0.
exec bash scripts/robodojo.sh server \
  --policy-dir "${policy_dir}" --task "${task}" --ckpt "${ckpt}" --policy-env "${policy_env}" \
  --policy-port "${port}" --bind-host "${host}" --action-type "${action_type}" \
  --env-cfg "${env_cfg}" --seed "${seed}" --policy-gpu 0
