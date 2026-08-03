#!/bin/bash
# Fetch or update the SGLang source from GitHub using sparse checkout.
# Usage: bash update-sglang.sh [--full]
#
# Sparse checkout is used by default.
# --full fetches the full repository with depth=1.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repos/sglang"
REPO_URL="https://github.com/sgl-project/sglang.git"
BRANCH="main"

FULL_MODE=false
if [ "$1" = "--full" ]; then
    FULL_MODE=true
fi

# Sparse-checkout paths.
SPARSE_DIRS=(
    # Python core
    "python/sglang/srt"
    "python/sglang/jit_kernel"
    "python/sglang/lang"
    # CUDA/C++ kernels
    "sgl-kernel/csrc"
    "sgl-kernel/include"
    "sgl-kernel/python"
    "sgl-kernel/tests"
    "sgl-kernel/benchmark"
    # Examples and documentation
    "examples"
    "benchmark"
    "docs"
    "test"
)

mkdir -p "$(dirname "$REPO_DIR")"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating the SGLang source..."
    cd "$REPO_DIR"
    git pull --ff-only origin "$BRANCH" 2>/dev/null || git pull origin "$BRANCH"
    echo "Update complete."
else
    echo "Cloning the SGLang source for the first time..."

    if [ "$FULL_MODE" = true ]; then
        echo "Mode: full clone (depth=1)."
        git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        echo "Mode: sparse checkout (required paths only)."
        git clone --filter=blob:none --no-checkout --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
        cd "$REPO_DIR"
        git sparse-checkout init --cone
        git sparse-checkout set "${SPARSE_DIRS[@]}"
        git checkout "$BRANCH"
    fi

    echo "Clone complete."
fi

# Validate the checkout.
echo ""
echo "--- Validation ---"
PASS=0
FAIL=0

check() {
    if [ -e "$1" ]; then
        echo "  OK: $2"
        PASS=$((PASS + 1))
    else
        echo "  Missing: $2"
        FAIL=$((FAIL + 1))
    fi
}

check "$REPO_DIR/python/sglang/srt/layers/attention" "SRT attention layers"
check "$REPO_DIR/python/sglang/srt/models" "SRT models"
check "$REPO_DIR/python/sglang/srt/managers" "SRT managers"
check "$REPO_DIR/python/sglang/srt/mem_cache" "SRT mem_cache"
check "$REPO_DIR/python/sglang/jit_kernel" "JIT kernels"
check "$REPO_DIR/sgl-kernel/csrc" "sgl-kernel CUDA source"
check "$REPO_DIR/examples" "Examples"
check "$REPO_DIR/docs" "Documentation"

echo ""
echo "Validation: $PASS passed, $FAIL failed."

du -sh "$REPO_DIR" 2>/dev/null | awk '{print "Repository size: "$1}'
