#!/usr/bin/env bash
# B200 一键启动脚本 — 每天换 node 后执行一次
# Usage: bash /home/zufayu/ClaudeSkillsE2EPerf/scripts/start_b200.sh
set -euo pipefail

REPO=/home/zufayu/ClaudeSkillsE2EPerf
RUNNER_DIR=/home/zufayu/actions-runner
SGLANG_IMAGE=lmsysorg/sglang:v0.5.9-cu130
MODEL=/SFS-aGqda6ct/models/DeepSeek-R1-0528-NVFP4-v2

echo "=== B200 Node Setup ==="
echo "  Host: $(hostname)"
echo "  GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"

# 1. Repo sync
echo ""
echo "1. Sync repo..."
cd $REPO && git fetch origin && git reset --hard origin/main && echo "  OK"

# 2. Docker container
echo ""
echo "2. Docker container..."
if docker ps --format '{{.Names}}' | grep -q zufa_sglang; then
    echo "  zufa_sglang already running"
else
    docker rm -f zufa_sglang 2>/dev/null || true
    docker run -d --name zufa_sglang \
        --privileged --gpus all --ipc=host --net=host \
        -v /home/zufayu:/home/zufayu \
        -v /SFS-aGqda6ct:/SFS-aGqda6ct \
        -w $REPO \
        $SGLANG_IMAGE sleep infinity
    echo "  Created zufa_sglang"
fi
docker exec zufa_sglang python3 -c "import sglang; print(f'  SGLang {sglang.__version__}')"
docker exec zufa_sglang ls $MODEL/config.json > /dev/null && echo "  Model OK"

# 3. Runner
echo ""
echo "3. GitHub Actions runner..."
if pgrep -f "actions-runner.*run.sh" > /dev/null 2>&1; then
    echo "  Runner already running"
else
    cd $RUNNER_DIR
    # Re-configure with current hostname (--replace overwrites old registration)
    TOKEN=$(curl -sf -X POST \
        -H "Authorization: token $(git -C $REPO remote get-url origin | sed 's|https://\(.*\)@github.com.*|\1|')" \
        "https://api.github.com/repos/zufayu/ClaudeSkillsE2EPerf/actions/runners/registration-token" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
    ./config.sh --url https://github.com/zufayu/ClaudeSkillsE2EPerf \
        --token "$TOKEN" \
        --name "b200-$(hostname | cut -d- -f4-)" \
        --labels self-hosted,Linux,X64,b200 \
        --unattended --replace 2>/dev/null || true
    nohup ./run.sh > runner.log 2>&1 &
    echo "  Runner started (PID $!)"
fi

echo ""
echo "=== Done! ==="
