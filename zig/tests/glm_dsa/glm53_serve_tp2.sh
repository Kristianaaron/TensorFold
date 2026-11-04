#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kristian Aaron (Krishando)
set -euo pipefail

# Launch the native CUDA server on a two-Spark TP2 pair. HEAD_HOST must be the
# IPv4 address rank 1 can reach for NCCL bootstrap and the HTTP smoke request.
# The native RoCE transport reads active_mtu from the selected port (1024 on
# these Sparks); NCCL uses the HCA, socket interface, and GID settings below.
action=${1:-start}
: "${HEAD_HOST:?set HEAD_HOST to rank 0's reachable IPv4 address}"
: "${WORKER_HOST:?set WORKER_HOST to rank 1's SSH host}"
: "${MODEL:?set MODEL to the materialized checkpoint path on both hosts}"
: "${SRC:?set SRC to this source checkout path on both hosts}"
: "${IMG:?set IMG to the CUDA runtime image available on both hosts}"
PORT=${PORT:-8000}
COMM_PORT=${COMM_PORT:-29570}
HEAD_NAME=${HEAD_NAME:-glm53-zserve-head}
WORKER_NAME=${WORKER_NAME:-glm53-zserve-worker}

stop_pair() {
    ssh "$WORKER_HOST" docker stop -t 60 "$WORKER_NAME" >/dev/null 2>&1 || true
    ssh "$HEAD_HOST" docker stop -t 60 "$HEAD_NAME" >/dev/null 2>&1 || true
}

case "$action" in
    start)
        # Rank 1 waits for NCCL bootstrap from rank 0, so bring it up first.
        ssh "$WORKER_HOST" bash -s -- "$WORKER_NAME" "$IMG" "$MODEL" "$SRC" "$HEAD_HOST" "$COMM_PORT" <<'REMOTE'
set -euo pipefail
name=$1 image=$2 model=$3 src=$4 head=$5 comm_port=$6
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all --network host --ipc=host \
  --device /dev/infiniband \
  -v "$model:$model:ro" -v "$src:/workspace/tensorfold-source:ro" \
  -e TF_GLM_RANK=1 -e TF_GLM_WORLD=2 \
  -e NCCL_COMM_ID="$head:$comm_port" \
  -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_SOCKET_IFNAME=enp1s0f1 -e NCCL_IB_GID_INDEX=3 \
  "$image" tensorfold serve "$model" --backend cuda --context 262144 --no-drafts
REMOTE
        sleep 2
        ssh "$HEAD_HOST" bash -s -- "$HEAD_NAME" "$IMG" "$MODEL" "$SRC" "$HEAD_HOST" "$COMM_PORT" "$PORT" <<'REMOTE'
set -euo pipefail
name=$1 image=$2 model=$3 src=$4 head=$5 comm_port=$6 port=$7
docker rm -f "$name" >/dev/null 2>&1 || true
docker run -d --name "$name" --gpus all --network host --ipc=host \
  --device /dev/infiniband \
  -v "$model:$model:ro" -v "$src:/workspace/tensorfold-source:ro" \
  -e TF_GLM_RANK=0 -e TF_GLM_WORLD=2 \
  -e NCCL_COMM_ID="$head:$comm_port" \
  -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_SOCKET_IFNAME=enp1s0f1 -e NCCL_IB_GID_INDEX=3 \
  "$image" tensorfold serve "$model" --backend cuda --context 262144 --no-drafts --host 0.0.0.0 --port "$port"
REMOTE
        for _ in $(seq 1 60); do
            if curl --fail --silent "http://$HEAD_HOST:$PORT/health" >/dev/null; then break; fi
            sleep 2
        done
        curl --fail --silent --show-error "http://$HEAD_HOST:$PORT/v1/chat/completions" \
          -H 'content-type: application/json' \
          --data '{"model":"glm_moe_dsa","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":4,"temperature":0}'
        ;;
    stop)
        stop_pair
        ;;
    smoke)
        curl --fail --silent --show-error "http://$HEAD_HOST:$PORT/v1/chat/completions" \
          -H 'content-type: application/json' \
          --data '{"model":"glm_moe_dsa","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":4,"temperature":0}'
        ;;
    *)
        echo "usage: $0 [start|stop|smoke]" >&2
        exit 2
        ;;
esac
