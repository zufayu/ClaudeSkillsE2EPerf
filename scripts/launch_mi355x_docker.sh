#!/usr/bin/env bash
# =============================================================================
# Launch Docker container for MI355X DeepSeek R1 benchmarking (ROCm + ATOM)
#
# Usage:
#   bash launch_mi355x_docker.sh [--name MI355X_atom] [--image <ATOM_IMAGE>]
#
# After launch:
#   docker attach <name>
#   bash /home/zufayu/ClaudeSkillsE2EPerf/scripts/sa_bench_mi355x.sh --help
# =============================================================================

set -euo pipefail

IMAGE="${IMAGE:-}"
NAME="${NAME:-MI355X_atom}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)   NAME="$2"; shift 2 ;;
        --image)  IMAGE="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$IMAGE" ]]; then
    echo "ERROR: --image is required (ATOM docker image)"
    echo "  Check https://github.com/ROCm/ATOM for the recommended image"
    echo ""
    echo "Usage: bash launch_mi355x_docker.sh --image <ATOM_DOCKER_IMAGE>"
    exit 1
fi

echo "Launching Docker container:"
echo "  Image: ${IMAGE}"
echo "  Name:  ${NAME}"

docker run -itd --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ulimit core=0:0 \
    --ulimit memlock=-1:-1 \
    --shm-size 64G \
    --cap-add=CAP_SYS_PTRACE \
    --net host \
    -v /home:/home \
    -v /data:/data \
    --name "${NAME}" \
    --entrypoint /bin/bash \
    "${IMAGE}"

echo ""
echo "Container started. Attach with:"
echo "  docker attach ${NAME}"
echo ""
echo "Inside the container, run:"
echo "  bash /home/zufayu/ClaudeSkillsE2EPerf/scripts/sa_bench_mi355x.sh \\"
echo "    --model /data/DeepSeek-R1-0528 \\"
echo "    --configs bf16-throughput"
echo ""
echo "For MTP-3 latency config:"
echo "  bash /home/zufayu/ClaudeSkillsE2EPerf/scripts/sa_bench_mi355x.sh \\"
echo "    --model /data/DeepSeek-R1-0528 \\"
echo "    --model-mtp /data/DeepSeek-R1-0528-mtp3 \\"
echo "    --configs all"
