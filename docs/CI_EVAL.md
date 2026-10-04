# CI eval client (GitHub Actions)

`.github/workflows/robodojo-eval-client.yml` deploys the RoboDojo simulator container
(`robodojo:<tag>`, built from the root `Dockerfile`) on a self-hosted GPU runner and runs it as
the **evaluation client** against a policy server that is already running somewhere else
(e.g. Pi_05 on the host, in another container, or on another machine).

```
┌─ GitHub runner (self-hosted, GPU) ─────────┐    WebSocket     ┌─ Policy server ─────────────┐
│ robodojo-client container                  │ ◄──────────────► │ XPolicyLab/policy/<POLICY>  │
│ Isaac Sim + tasks + scoring                │   host:port      │ robodojo.sh server          │
└────────────────────────────────────────────┘                  └─────────────────────────────┘
```

The workflow does **not** start the policy server. Start it first, bound to `0.0.0.0`, e.g.:

```bash
bash scripts/robodojo.sh server --policy-dir XPolicyLab/policy/Pi_05 \
  --task stack_bowls --ckpt <CKPT> --policy-env <ENV> --policy-port 9999
```

## 1. Runner setup (one time)

GitHub-hosted runners have no NVIDIA GPU and too little disk, so use a **self-hosted** Linux
runner:

| Need | Details |
| --- | --- |
| Labels | `self-hosted`, `linux`, `a100` |
| GPU / driver | NVIDIA driver ≥ 570 (CUDA 12.8) |
| Docker | Docker Engine + NVIDIA Container Toolkit: `sudo bash docker/install_docker_nvidia.sh`; add the runner user to the `docker` group and restart the runner service |
| Disk | 300 GB+ free (image ≈ 200 GB plus build cache) |
| Tools | `git`, `python3` (≥ 3.8, standard library only), `bash`, `timeout` |
| Network | TCP access from the runner to the policy server host/port |

Prepare the persistent assets and cache directories once:

```bash
git clone --recurse-submodules <repo> /data/robodojo/src && cd /data/robodojo/src
bash scripts/init_assets.sh          # downloads Assets/ (also writes curobo.yml with absolute paths)
mkdir -p /data/robodojo/cache
```

Then set these **repository (or organization) variables**
(*Settings → Secrets and variables → Actions → Variables*):

| Variable | Example | Purpose |
| --- | --- | --- |
| `ROBODOJO_ASSETS_DIR` | `/data/robodojo/src/Assets` | Persistent `Assets/` mounted read-only into the client |
| `ROBODOJO_CACHE_DIR` | `/data/robodojo/cache` | Warp/Omniverse caches reused across runs (avoids multi-minute cold starts) |
| `ROBODOJO_ASSETS_BAKED_DIR` | *(optional)* | Path baked into `Assets/Robots/**/curobo.yml`. Auto-detected; only set if detection fails |
| `ROBODOJO_ENV_CFG` | *(optional, `arx_x5`)* | env_cfg stem |
| `ROBODOJO_ENV_GPU` | *(optional, `0`)* | Isaac Sim GPU id on the runner |
| `ROBODOJO_IMAGE_TAG` | *(optional, `cuda12.8`)* | Client image `robodojo:<tag>` |
| `ROBODOJO_OUTPUT_DIR` | *(optional, `~/robodojo-eval_result`)* | Persistent results dir for background clients |
| `ROBODOJO_HF_ENDPOINT` | *(optional, `https://huggingface.co`)* | Asset download endpoint, e.g. `https://hf-mirror.com` |
| `ROBODOJO_GIT_MIRROR_DIR` | *(optional)* | Dir of bare submodule mirrors (`XPolicyLab.git`, `IsaacLab.git`, `curobo.git`) |
| `ROBODOJO_CUDA_CHECK_IMAGE` | *(optional)* | CUDA image for the GPU check, e.g. `docker.m.daocloud.io/nvidia/cuda:12.8.1-base-ubuntu22.04` when Docker Hub is blocked |
| `ROBODOJO_BUILD_ARGS` | *(optional)* | Extra `docker build` args, e.g. `--build-arg CUDA_IMAGE=docker.m.daocloud.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |

The custom runner labels are declared in `.github/actionlint.yaml` so `actionlint` accepts them.

**Why Assets is mounted twice:** the curobo robot configs contain absolute paths from the
location where `init_assets.sh` ran. The helper mounts the assets at
`/workspace/RoboDojo/Assets` and also at that original path, so those paths resolve inside the
container. Don't move `Assets/` after initializing it, or re-run
`python3 utils/update_embodiment_config_path.py` from the new repo root.

## 2. Run it

*Actions → RoboDojo eval client → Run workflow*. Inputs:

| Input | Default | Notes |
| --- | --- | --- |
| `policy_name` | `Pi_05` | Directory under `XPolicyLab/policy/` (needs `deploy.py`) |
| `policy_host` | `127.0.0.1` | Where the policy server runs, **as seen from the runner** (see below) |
| `policy_port` | `9999` | The `--policy-port` the server was started with |
| `ckpt` | `external` | Label recorded in result paths |
| `tasks` | `stack_bowls` | Comma list; leave empty to use `dimension` |
| `dimension` | — | `generalization`, `memory`, `precision`, `long-horizon`, `open`, `all` |
| `eval_num` | `1` | Integer, or `native` for per-task counts from `_task.yml` |
| `action_type` | `ee` | Must match the checkpoint (`ee` or `joint`) |
| `seed` | `0` | Eval / layout seed |
| `rebuild_image` | `false` | Force a `docker build` (otherwise only built when missing, ~1 h) |

GitHub limits manual (`workflow_dispatch`) runs to 10 inputs, so `env_cfg`, `env_gpu`, and
`image_tag` come from the repository variables above. When the workflow is called with
`workflow_call`, they can also be passed as inputs, which take precedence over the variables.

**Choosing `policy_host`.** The client container uses `--network host`, so the address is
resolved from the runner machine itself:

| Policy server runs on… | `policy_host` |
| --- | --- |
| the same machine as the runner (host or a `--network host` container) | `127.0.0.1` (default) |
| another machine on the LAN | that machine's IP, e.g. `192.168.1.50` (`hostname -I` on it) |
| a remote/cloud machine | its public IP or DNS name (port must be open in its firewall) |

The server must be started with `--bind-host 0.0.0.0` (the default) to be reachable from
another machine. Check from the runner with `nc -vz <host> <port>`.

Or from the CLI:

```bash
gh workflow run robodojo-eval-client.yml \
  -f policy_host=10.0.0.5 -f policy_port=9999 -f ckpt=pi05_robodojo -f tasks=stack_bowls -f eval_num=1
```

The workflow is also `workflow_call`-able, so another workflow can start the policy server and
then call this one.

## 3. What the workflow does

A single job, **start-client**, starts the evaluation client **in the background** and finishes.
It does not wait for the evaluation to complete.

1. Checks out the repo (falling back to an API tarball if github.com git is unreachable) and
   fills in the submodules from the local mirrors in `ROBODOJO_GIT_MIRROR_DIR`, or from GitHub.
2. Validates the inputs and resolves the task list.
3. **Already running?** If a container `robodojo-client-<policy>-<task>` is running for every
   requested task, prints its status and recent logs, then ends successfully. Nothing else runs.
4. Otherwise it checks the GPU, then reuses whatever is cached:
   - **Client image:** if `robodojo:<tag>` already exists on the runner, it is used as is.
     Submodule fetching and `docker build` are skipped entirely, and the policy-adapter and
     GPU checks run against that image. It is built (after fetching submodules) only if it's
     missing or `rebuild_image` is set.
   - **Assets:** `ensure-assets` returns immediately when `ROBODOJO_ASSETS_DIR` is complete.
     The marker is `<parent>/.robodojo_assets_complete`, and a manually prepared Assets dir
     also counts. Otherwise it downloads the assets once (resumable, from
     `ROBODOJO_HF_ENDPOINT`, e.g. `https://hf-mirror.com`) and writes the marker.
5. Waits for the policy server port.
6. Starts one **detached** container per task (`docker run -d`, no `--rm`). Results go to
   `ROBODOJO_OUTPUT_DIR` on the runner, and the client log to
   `<ROBODOJO_OUTPUT_DIR>/_ci/<task>/client.log`. A summary table lists each task as
   `started` or `already running`.

Watch or stop a client on the runner:

```bash
docker ps --filter label=robodojo.client=1
docker logs -f robodojo-client-<policy>-<task>
TASK=<task> ROBODOJO_CONTAINER_NAME=robodojo-client-<policy>-<task> bash scripts/ci/robodojo_ci.sh client-status
docker rm -f robodojo-client-<policy>-<task>
```

Runs share the concurrency group `robodojo-gpu-eval`, so two start requests never race.
Several tasks started in one run share GPU `ROBODOJO_ENV_GPU`, so keep task lists small, or
start one task per run.

## 4. Running the same steps locally

Every step is a plain script that works without GitHub:

```bash
export ROBODOJO_ASSETS_DIR=$PWD/Assets ROBODOJO_CACHE_DIR=$HOME/.cache/robodojo-ci
export TASK=stack_bowls POLICY_NAME=Pi_05 POLICY_HOST=127.0.0.1 POLICY_PORT=9999 CKPT=<CKPT>
bash scripts/ci/robodojo_ci.sh check-gpu
bash scripts/ci/robodojo_ci.sh ensure-image
bash scripts/ci/robodojo_ci.sh run-task --dry-run   # print the docker command
bash scripts/ci/robodojo_ci.sh run-task
bash scripts/ci/robodojo_ci.sh report
```

## 5. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `policy server ... not reachable` | Server not started, bound to `127.0.0.1`, wrong port, or a firewall is blocking it. Check with `nc -vz <host> <port>` on the runner. |
| `submodule ... is empty` during the build | The checkout must include submodules. The workflow does this; for local builds run `git submodule update --init --recursive`. |
| `ValueError: .../X5A.urdf is not a file` | The baked curobo path wasn't mounted. Set `ROBODOJO_ASSETS_BAKED_DIR` to the `<dir>/Assets` shown in `Assets/Robots/**/curobo.yml`. |
| First run very slow | Cold Warp/Omniverse caches. They persist in `ROBODOJO_CACHE_DIR` after the first run. |
| OOM | The policy server and Isaac Sim share a GPU. Run the server on another GPU or machine, or set `env_gpu`. |
| Leftover container after a cancel | `docker ps -a --filter name=robodojo-client-` then `docker rm -f <name>`. The cleanup step does this automatically. |
