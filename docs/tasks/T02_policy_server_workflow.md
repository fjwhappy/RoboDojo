# T02: GitHub Actions workflow to deploy the policy server container

## Goal

Add `.github/workflows/robodojo-policy-server.yml`. It deploys an XPolicyLab policy server
(default **Pi 0.5**, `XPolicyLab/policy/Pi_05`) as a **detached Docker container** on the
self-hosted GPU runner, prints the listening address and port, and finishes. It pairs with the
eval-client workflow (T01, `.github/workflows/robodojo-eval-client.yml`).

```
GitHub runner (a100)
├── robodojo-policy-<policy>  (this workflow)  listens 0.0.0.0:<port>
└── robodojo-client-<policy>-<task>  (T01)     connects 127.0.0.1:<port>
```

## Behaviour

1. **Already running?** If a container `robodojo-policy-<policy>` is running and its port
   accepts TCP connections, print the container info and listening address, then exit 0.
   Don't touch anything else.
2. Otherwise, make sure everything is in place, **reusing caches** on the runner:
   - **Docker image** `robodojo-policy-<policy>:<tag>`: build it only if it's missing (or
     `rebuild_image=true`).
   - **Model checkpoint**: published RoboDojo checkpoint
     `ckpt/RoboDojo/<policy>/<run>/<step>` from the HF dataset (via `ROBODOJO_HF_ENDPOINT`, e.g.
     hf-mirror). Inference only needs `params/`, `assets/` and `_CHECKPOINT_METADATA`. Skip
     `train_state/` (~32 GB). Download only what's missing, verify the sha256 of each file, and
     write a completion marker. If the marker exists, skip the download.
   - **Model data**: auxiliary downloads at runtime (e.g. the PaliGemma tokenizer from GCS) go to
     a persistent `OPENPI_DATA_HOME` cache that's mounted into the container, so they're fetched
     once.
3. Start the server container detached (`docker run -d`, `--restart unless-stopped`, host
   network, one GPU) running `scripts/robodojo.sh server` for the policy.
4. Wait until the port accepts connections, failing if the container exits. Then print the
   **listening address and port**, and the client connection settings (`policy_host`,
   `policy_port`), and finish. Don't wait for evaluations.

## Inputs

`policy_name` (default `Pi_05`), `ckpt` (checkpoint run directory, default
`RoboDojo-sim-arx_x5-joint-0`), `task` (passed to the server, default `stack_bowls`), `port`
(default `9999`), `gpu` (default `1`), `action_type` (default `joint`), `rebuild_image`,
`restart` (replace a running server).

Runner variables: `ROBODOJO_POLICY_CACHE_DIR` (checkpoints + model data, default
`~/robodojo-policy-cache`), `ROBODOJO_HF_ENDPOINT`, `ROBODOJO_BUILD_ARGS`, `ROBODOJO_GIT_MIRROR_DIR`.

## Deliverables

- `docker/policy/Dockerfile`: generic XPolicyLab policy server image. Pi_05 uses the uv-managed
  OpenPI env (`install.sh`). No checkpoints are baked in.
- `scripts/ci/policy_server.sh`: `ensure-image`, `ensure-checkpoint`, `start`, `status`,
  `stop`, with `--dry-run`.
- `scripts/ci/download_checkpoint.py`: idempotent, resumable, sha256-verified checkpoint
  download over HTTPS (works with HF mirrors, no git-lfs).
- `.github/workflows/robodojo-policy-server.yml`
- Docs in `docs/CI_EVAL.md`.

## Acceptance

- actionlint, shellcheck and ruff are clean.
- Run 1 on the runner: the image is built, the checkpoint downloaded, and the server started; the
  workflow prints `listening on 0.0.0.0:<port>`.
- Run 2: reports "already running" with the address and port, and doesn't download or build.
- After the container is removed, a run reuses the cached image and checkpoint and only starts
  the container.
