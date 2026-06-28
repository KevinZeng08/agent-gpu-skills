#!/usr/bin/env bash
# Fetch / update the NVSHMEM source from GitHub so the agent can co-read code
# alongside the documentation in references/.
#
# Usage: bash update-nvshmem.sh [--full] [--ref <tag-or-branch>]
#
#   (default)        sparse checkout of the key source dirs (~small)
#   --full           full checkout (depth=1) of the whole repo
#   --ref <ref>      clone a specific tag or branch (default: v3.7.0-0,
#                    matching the bundled NVSHMEM 3.7.0 docs)
#
# The source is cloned into repos/nvshmem/ next to this script (matching the
# other skills' repos/ layout). If the pinned ref is unavailable upstream, the
# script falls back to the default branch.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repos/nvshmem"
REPO_URL="https://github.com/NVIDIA/nvshmem.git"
REF="v3.7.0-0"          # matches the NVSHMEM version documented in references/
FALLBACK_BRANCH="main"

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
    "src"                 # core: host/, device/, include/, modules/, bin/
    "nvshmem4py"          # Python bindings (the source behind the NVSHMEM4Py docs)
    "examples"            # runnable C/CUDA + Python example programs
    "perftest"            # micro-benchmarks (RMA / AMO / collectives)
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
    echo "更新 NVSHMEM 源码 (ref=$REF)..."
    cd "$REPO_DIR"
    git fetch --depth 1 origin "$REF" 2>/dev/null && git checkout -f FETCH_HEAD 2>/dev/null \
        || { echo "无法 fetch ref '$REF'，尝试默认分支 '$FALLBACK_BRANCH'"; \
             git fetch --depth 1 origin "$FALLBACK_BRANCH" && git checkout -f FETCH_HEAD; }
    echo "更新完成."
else
    echo "首次 clone NVSHMEM 源码..."
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

check "src/include/nvshmem.h"        "public host header (nvshmem.h)"
check "src/include/nvshmemx.h"       "NVSHMEM extensions header (nvshmemx.h)"
check "src/include/device"           "device-side API headers"
check "src/host/comm"                "host RMA/AMO/signal implementation"
check "src/host/coll"                "host collectives implementation"
check "src/host/team"                "team management"
check "src/device/init"             "device init"
check "nvshmem4py/nvshmem/core"      "NVSHMEM4Py core package"
check "examples"                     "example programs"

echo ""
echo "验证: $PASS 通过, $FAIL 失败"
du -sh "$REPO_DIR" 2>/dev/null | awk '{print "仓库大小: "$1}'
echo "源码位置: $REPO_DIR"
