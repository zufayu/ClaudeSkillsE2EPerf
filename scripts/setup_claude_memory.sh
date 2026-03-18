#!/usr/bin/env bash
# setup_claude_memory.sh — 一次性设置 Claude Code 记忆持久化
# 适用于：在 Docker 容器内运行 Claude Code，容器挂载了宿主机 home 目录
#
# 原理：将 Claude 的记忆目录从容器内部路径 symlink 到挂载的宿主机目录，
#       这样记忆直接写入宿主机磁盘，容器重建后只需重新跑一次本脚本即可恢复。
#
# 用法：
#   bash scripts/setup_claude_memory.sh                    # 自动检测挂载点
#   bash scripts/setup_claude_memory.sh /home/zufayu       # 手动指定挂载路径

set -euo pipefail

CLAUDE_MEMORY="/root/.claude/projects/-/memory"
MOUNT_BASE="${1:-}"

# 自动检测挂载的 home 目录
if [ -z "$MOUNT_BASE" ]; then
    MOUNT_BASE=$(mount | grep -oP '/home/\S+' | head -1 | awk '{print $1}')
    if [ -z "$MOUNT_BASE" ]; then
        echo "ERROR: 未检测到挂载的 home 目录，请手动指定："
        echo "  bash $0 /home/your_username"
        exit 1
    fi
fi

PERSISTENT_DIR="$MOUNT_BASE/.claude_memory"

echo "=== Claude Code 记忆持久化设置 ==="
echo "  挂载路径: $MOUNT_BASE"
echo "  持久目录: $PERSISTENT_DIR"
echo "  符号链接: $CLAUDE_MEMORY -> $PERSISTENT_DIR"

# 创建持久目录
mkdir -p "$PERSISTENT_DIR"

# 如果已经是正确的 symlink，跳过
if [ -L "$CLAUDE_MEMORY" ] && [ "$(readlink "$CLAUDE_MEMORY")" = "$PERSISTENT_DIR" ]; then
    echo "已经设置好了，无需重复操作。"
    echo "当前记忆文件:"
    ls "$CLAUDE_MEMORY"/ 2>/dev/null || echo "  (空)"
    exit 0
fi

# 确保父目录存在
mkdir -p "$(dirname "$CLAUDE_MEMORY")"

# 如果存在旧的记忆目录（非 symlink），迁移内容
if [ -d "$CLAUDE_MEMORY" ] && [ ! -L "$CLAUDE_MEMORY" ]; then
    echo "迁移现有记忆文件..."
    cp -a "$CLAUDE_MEMORY"/* "$PERSISTENT_DIR"/ 2>/dev/null || true
    rm -rf "$CLAUDE_MEMORY"
fi

# 创建 symlink
ln -sf "$PERSISTENT_DIR" "$CLAUDE_MEMORY"

echo "设置完成！"
echo "当前记忆文件:"
ls "$CLAUDE_MEMORY"/ 2>/dev/null || echo "  (空)"
echo ""
echo "容器重建后，重新运行本脚本即可恢复记忆。"
