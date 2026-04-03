#!/usr/bin/env bash
# =============================================================================
# One-shot setup: GitHub Actions self-hosted runner for MI355X
#
# Run this ON the MI355X machine (my-gpu-41):
#   bash /shared/amdgpu/home/zufa_yu_qle/ClaudeSkillsE2EPerf/scripts/setup_mi355x_runner.sh
#
# Prerequisites:
#   - Internet access to github.com
#   - curl, tar installed
#   - sudo access (for svc.sh install)
# =============================================================================
set -euo pipefail

REPO_OWNER="zufayu"
REPO_NAME="ClaudeSkillsE2EPerf"
RUNNER_DIR="$HOME/actions-runner"
RUNNER_NAME="mi355x-runner"
RUNNER_LABELS="self-hosted,mi355x"
RUNNER_VERSION="2.322.0"

echo "=== MI355X GitHub Actions Runner Setup ==="
echo "Runner dir: $RUNNER_DIR"
echo ""

# Step 1: Get registration token via GitHub API
# Need a PAT with admin:repo scope — read from git remote or env
REPO_DIR="/shared/amdgpu/home/zufa_yu_qle/ClaudeSkillsE2EPerf"
PAT=""
if [[ -d "$REPO_DIR/.git" ]]; then
    PAT=$(git -C "$REPO_DIR" config --get remote.origin.url 2>/dev/null | grep -oP 'ghp_[^@]+' || true)
fi
if [[ -z "$PAT" ]]; then
    echo "ERROR: Could not extract PAT from git remote URL."
    echo "Set remote URL: git remote set-url origin https://ghp_XXX@github.com/$REPO_OWNER/$REPO_NAME.git"
    exit 1
fi

echo "Getting registration token..."
REG_TOKEN=$(curl -s -X POST \
    -H "Authorization: token $PAT" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/runners/registration-token" \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('token',''))")

if [[ -z "$REG_TOKEN" ]]; then
    echo "ERROR: Failed to get registration token. Check PAT permissions (needs admin:repo or Actions R/W)."
    exit 1
fi
echo "Got registration token: ${REG_TOKEN:0:8}..."

# Step 2: Download and extract runner
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [[ ! -f "./config.sh" ]]; then
    echo "Downloading actions-runner v${RUNNER_VERSION}..."
    curl -sL -o actions-runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
    tar xzf actions-runner.tar.gz
    rm -f actions-runner.tar.gz
    echo "Extracted runner."
else
    echo "Runner already downloaded, skipping."
fi

# Step 3: Configure runner
echo "Configuring runner..."
./config.sh \
    --url "https://github.com/$REPO_OWNER/$REPO_NAME" \
    --token "$REG_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --unattended \
    --replace

# Step 4: Install and start as service
echo "Installing as service..."
sudo ./svc.sh install || true
sudo ./svc.sh start || true

echo ""
echo "=== Done ==="
echo "Runner '$RUNNER_NAME' registered with labels: $RUNNER_LABELS"
echo "Verify at: https://github.com/$REPO_OWNER/$REPO_NAME/settings/actions/runners"
echo ""
echo "To check status: sudo ./svc.sh status"
echo "To view logs:    journalctl -u actions.runner.* -f"
