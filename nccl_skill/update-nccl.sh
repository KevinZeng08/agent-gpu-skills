#!/usr/bin/env bash
# Fetch / update the NCCL source from GitHub so the agent can co-read code
# alongside the documentation in references/.
#
# Usage: bash update-nccl.sh [--full] [--ref <tag-or-branch>]
#
#   (default)        sparse checkout of the key source dirs (~small)
#   --full           full checkout (depth=1) of the whole repo
#   --ref <ref>      clone a specific tag or branch (default: v2.30.7-1,
#                    matching the bundled NCCL 2.30.7 docs)
#
# The source is cloned into repos/nccl/ next to this script (matching the other
# skills' repos/ layout). If the pinned ref is unavailable upstream, the script
# falls back to the default branch.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repos/nccl"
REPO_URL="https://github.com/NVIDIA/nccl.git"
REF="v2.30.7-1"          # matches the NCCL version documented in references/
FALLBACK_BRANCH="master"

FULL_MODE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL_MODE=true; shift ;;
        --ref)  REF="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Key directories worth reading (cone-mode sparse checkout).
SPARSE_DIRS=(
    "src"                 # core implementation: include/, device/, nccl_device/,
                          # gin/, rma/, graph/, transport/, ras/, register/, ...
    "bindings/nccl4py"    # Python bindings (the source behind the nccl4py docs)
    "makefiles"           # version.mk and build glue
    "docs/examples"       # runnable C / Python example programs
    "plugins"             # net / tuner / profiler / gin plugin interfaces & samples
)

clone_ref() {
    local ref="$1"
    if [ "$FULL_MODE" = true ]; then
        echo "模式: 完整 clone（depth=1, ref=$ref）"
        git clone --depth 1 --branch "$ref" "$REPO_URL" "$REPO_DIR"
    else
        echo "模式: sparse checkout（只拉关键目录, ref=$ref）"
        git clone --filter=blob:none --no-checkout --depth 1 --branch "$ref" "$REPO_URL" "$REPO_DIR"
        cd "$REPO_DIR"
        git sparse-checkout init --cone
        git sparse-checkout set "${SPARSE_DIRS[@]}"
        git checkout "$ref"
    fi
}

if [ -d "$REPO_DIR/.git" ]; then
    echo "更新 NCCL 源码 (ref=$REF)..."
    cd "$REPO_DIR"
    git fetch --depth 1 origin "$REF" 2>/dev/null && git checkout -f FETCH_HEAD 2>/dev/null \
        || { echo "无法 fetch ref '$REF'，尝试默认分支 '$FALLBACK_BRANCH'"; \
             git fetch --depth 1 origin "$FALLBACK_BRANCH" && git checkout -f FETCH_HEAD; }
    echo "更新完成."
else
    echo "首次 clone NCCL 源码..."
    clone_ref "$REF" || {
        echo "ref '$REF' 不可用，回退到默认分支 '$FALLBACK_BRANCH'"
        rm -rf "$REPO_DIR"
        REF="$FALLBACK_BRANCH"
        clone_ref "$REF"
    }
    echo "Clone 完成."
fi

# --- 验证 -------------------------------------------------------------------
echo ""
echo "--- 验证 ---"
PASS=0
FAIL=0
check() {
    if [ -e "$REPO_DIR/$1" ]; then
        echo "  OK: $2"
        PASS=$((PASS + 1))
    else
        echo "  缺失: $2"
        FAIL=$((FAIL + 1))
    fi
}

check "src/nccl.h.in"                "public header (nccl.h.in)"
check "src/include/nccl_device"      "device API headers"
check "src/init.cc"                  "communicator init"
check "src/collectives.cc"           "collective entry points"
check "src/graph"                    "topology / graph search"
check "src/transport"                "transports (p2p/net/shm/...)"
check "bindings/nccl4py/nccl"        "nccl4py Python package"
check "makefiles/version.mk"         "version.mk"

echo ""
echo "验证: $PASS 通过, $FAIL 失败"
du -sh "$REPO_DIR" 2>/dev/null | awk '{print "仓库大小: "$1}'
echo "源码位置: $REPO_DIR"
