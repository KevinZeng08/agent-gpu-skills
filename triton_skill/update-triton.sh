#!/bin/bash
# Fetch or update the Triton source from GitHub using sparse checkout.
# Usage: bash update-triton.sh [--full]
#
# Sparse checkout fetches only the required paths by default.
# --full fetches the full repository with depth=1.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$SCRIPT_DIR/repos/triton"
REPO_URL="https://github.com/triton-lang/triton.git"
BRANCH="main"

FULL_MODE=false
if [ "$1" = "--full" ]; then
    FULL_MODE=true
fi

# Sparse-checkout paths.
SPARSE_DIRS=(
    # Python: tutorials, kernels, language API
    "python/tutorials"
    "python/triton_kernels"
    "python/triton/language"
    "python/triton/experimental/gluon"
    "python/triton/runtime"
    "python/triton/compiler"
    "python/triton/tools"
    "python/examples"
    # C++: compiler IR definitions and passes
    "include"
    "lib"
)

mkdir -p "$(dirname "$REPO_DIR")"

if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating the Triton source..."
    cd "$REPO_DIR"
    git pull --ff-only origin "$BRANCH" 2>/dev/null || git pull origin "$BRANCH"
    echo "Update complete."
else
    echo "Cloning the Triton source for the first time..."

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

check "$REPO_DIR/python/tutorials/01-vector-add.py" "Triton tutorials"
check "$REPO_DIR/python/tutorials/gluon/01-intro.py" "Gluon tutorials"
check "$REPO_DIR/python/triton_kernels/triton_kernels/matmul.py" "Triton kernels"
check "$REPO_DIR/python/triton/language/__init__.py" "Triton language"
check "$REPO_DIR/python/triton/experimental/gluon" "Gluon experimental"
check "$REPO_DIR/python/examples" "Examples"
check "$REPO_DIR/include/triton/Dialect" "C++ Dialect headers"
check "$REPO_DIR/lib/Dialect" "C++ Dialect implementations"

echo ""
echo "Validation: $PASS passed, $FAIL failed."

du -sh "$REPO_DIR" 2>/dev/null | awk '{print "Repository size: "$1}'
