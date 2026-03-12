#!/usr/bin/env bash
# =============================================================================
# Launch Docker container for H200 DeepSeek R1 benchmarking
#
# Usage:
#   bash launch_h200_docker.sh [--name H200_trtllm] [--image nvcr.io/...]
#
# After launch:
#   docker attach <name>
#   bash /home/kqian/ClaudeSkillsE2EPerf/scripts/sa_bench_h200.sh --help
# =============================================================================

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4}"
NAME="${NAME:-H200_trtllm}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)   NAME="$2"; shift 2 ;;
        --image)  IMAGE="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Launching Docker container:"
echo "  Image: ${IMAGE}"
echo "  Name:  ${NAME}"

docker run -itd --rm \
    --gpus all \
    --ulimit core=0:0 \
    --ulimit memlock=-1:-1 \
    --shm-size 8G \
    --cap-add=CAP_SYS_PTRACE \
    --cap-add=SYS_NICE \
    --net host \
    -v /mnt/raid0:/mnt/raid0 \
    -v /home:/home \
    -v /data:/data \
    -v /docker:/docker \
    --name "${NAME}" \
    --entrypoint /bin/bash \
    "${IMAGE}"

echo ""
echo "Container started. Attach with:"
echo "  docker attach ${NAME}"
echo ""
echo "Inside the container, run:"
echo "  bash /home/kqian/ClaudeSkillsE2EPerf/scripts/sa_bench_h200.sh \\"
echo "    --model /path/to/DeepSeek-R1-FP8 \\"
echo "    --configs all"
