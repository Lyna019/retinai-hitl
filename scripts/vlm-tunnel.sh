#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VLM SSH Tunnel — bridges Kubeflow HPC VLM service to localhost
#
# Usage:
#   ./scripts/vlm-tunnel.sh                                              # interactive
#   ./scripts/vlm-tunnel.sh <username>@hpc.ensia.edu.dz 8000 vlm-svc.kubeflow:8000
#
# Once running, set in .env:
#   VLM_SERVICE_URL=http://host.docker.internal:8888
#
# Then restart the backend:
#   docker compose restart backend
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LOCAL_PORT="${VLM_LOCAL_PORT:-8888}"
HPC_HOST="${1:-}"
HPC_REMOTE_PORT="${2:-8000}"
VLM_INTERNAL="${3:-vllm-gemma4-nodeport.vllm-ns:8000}"

if [ -z "$HPC_HOST" ]; then
  echo "╭─ RetinAI VLM SSH Tunnel ─────────────────────────────╮"
  echo "│                                                       │"
  echo "│  This script creates an SSH tunnel from localhost      │"
  echo "│  to the Kubeflow VLM service on the HPC.              │"
  echo "╰───────────────────────────────────────────────────────╯"
  echo ""
  read -rp "  HPC gateway (user@hpc.ensia.edu.dz): " HPC_HOST
  read -rp "  VLM service inside cluster (host:port) [vllm-gemma4-nodeport.vllm-ns:8000]: " VLM_INTERNAL
  VLM_INTERNAL="${VLM_INTERNAL:-vllm-gemma4-nodeport.vllm-ns:8000}"
fi

VLM_INTERNAL="${VLM_INTERNAL:-vlm-svc.kubeflow:${HPC_REMOTE_PORT}}"

echo ""
echo "  Tunnel: localhost:${LOCAL_PORT} → ${HPC_HOST} → ${VLM_INTERNAL}"
echo "  Set in .env: VLM_SERVICE_URL=http://host.docker.internal:${LOCAL_PORT}"
echo ""
echo "  Press Ctrl+C to close the tunnel."
echo ""

# Bind on 0.0.0.0 so Docker containers can reach the tunnel via host.docker.internal
ssh -N -L "0.0.0.0:${LOCAL_PORT}:${VLM_INTERNAL}" "${HPC_HOST}"
