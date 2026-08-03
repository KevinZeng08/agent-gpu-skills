#!/bin/bash
# Fetch or update the CUTLASS source from GitHub using sparse checkout.
# Usage: bash update-cutlass.sh [--full]
#
# Sparse checkout fetches only the required paths by default.
# --full fetches the full repository with depth=1.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repos/cutlass"
REPO_URL="https://github.com/NVIDIA/cutlass.git"
BRANCH="main"

FULL_MODE=false
if [ "$1" = "--full" ]; then
    FULL_MODE=true
fi

# Sparse-checkout paths.
SPARSE_DIRS=(
    # CuTeDSL Python DSL
    "python/CuTeDSL"
    "python/pycute"
    "python/cutlass_library"
    # CuTeDSL and C++ examples
    "examples"
    # CuTe and CUTLASS headers
    "include"
    # Tools
    "tools/library"
    "tools/util"
)

mkdir -p "$(dirname "$REPO_DIR")"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating the CUTLASS source..."
    cd "$REPO_DIR"
    git pull --ff-only origin "$BRANCH" 2>/dev/null || git pull origin "$BRANCH"
    echo "Update complete."
else
    echo "Cloning the CUTLASS source for the first time..."

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

check "$REPO_DIR/python/CuTeDSL/cutlass" "CuTeDSL source"
check "$REPO_DIR/python/pycute/layout.py" "pycute"
check "$REPO_DIR/examples/python/CuTeDSL" "CuTeDSL examples"
check "$REPO_DIR/examples/cute/tutorial" "CuTe tutorials"
check "$REPO_DIR/examples/49_hopper_gemm_with_collective_builder" "Hopper GEMM example"
check "$REPO_DIR/examples/70_blackwell_gemm" "Blackwell GEMM example"
check "$REPO_DIR/include/cute/layout.hpp" "CuTe headers"
check "$REPO_DIR/include/cutlass/gemm" "CUTLASS GEMM headers"

echo ""
echo "Validation: $PASS passed, $FAIL failed."

du -sh "$REPO_DIR" 2>/dev/null | awk '{print "Repository size: "$1}'
