#!/bin/bash
# Fetch or update all source repositories.
# Usage: bash update-repos.sh [repo_name]
#
# Without an argument: update all repositories.
# With an argument: update one repository (triton, cutlass, or sglang).
#
# Repositories are stored under repos/ in their corresponding skill directories:
#   triton_skill/repos/triton/
#   cutlass_skill/repos/cutlass/
#   sglang_skill/repos/sglang/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

clone_or_update() {
    local name="$1"
    local skill_dir="$2"
    local url="$3"
    local branch="$4"
    shift 4
    local sparse_dirs=("$@")

    local repos_dir="$SCRIPT_DIR/$skill_dir/repos"
    local repo_dir="$repos_dir/$name"

    mkdir -p "$repos_dir"

    echo ""
    echo "=== $name ==="

    if [ -d "$repo_dir/.git" ]; then
        echo "  Updating..."
        cd "$repo_dir"
        git pull --ff-only origin "$branch" 2>/dev/null || git pull origin "$branch"
        echo "  Update complete."
    else
        echo "  Initial clone with sparse checkout..."
        git clone --filter=blob:none --no-checkout --depth 1 --branch "$branch" "$url" "$repo_dir"
        cd "$repo_dir"
        git sparse-checkout init --cone
        git sparse-checkout set "${sparse_dirs[@]}"
        git checkout "$branch"
        echo "  Clone complete."
    fi

    du -sh "$repo_dir" 2>/dev/null | awk '{print "  Size: "$1}'
}

# Triton sparse-checkout paths.
triton_dirs=(
    "python/tutorials"
    "python/triton_kernels"
    "python/triton/language"
    "python/triton/experimental/gluon"
    "python/triton/runtime"
    "python/triton/compiler"
    "python/triton/tools"
    "python/examples"
    "include"
    "lib"
)

# CUTLASS sparse-checkout paths.
cutlass_dirs=(
    "python/CuTeDSL"
    "python/pycute"
    "python/cutlass_library"
    "examples"
    "include"
    "tools/library"
    "tools/util"
)

# SGLang sparse-checkout paths.
sglang_dirs=(
    "python/sglang/srt"
    "python/sglang/jit_kernel"
    "python/sglang/lang"
    "sgl-kernel/csrc"
    "sgl-kernel/include"
    "sgl-kernel/python"
    "sgl-kernel/tests"
    "sgl-kernel/benchmark"
    "examples"
    "benchmark"
    "docs"
    "test"
)

TARGET="${1:-all}"

case "$TARGET" in
    triton)
        clone_or_update "triton" "triton_skill" "https://github.com/triton-lang/triton.git" "main" "${triton_dirs[@]}"
        ;;
    cutlass)
        clone_or_update "cutlass" "cutlass_skill" "https://github.com/NVIDIA/cutlass.git" "main" "${cutlass_dirs[@]}"
        ;;
    sglang)
        clone_or_update "sglang" "sglang_skill" "https://github.com/sgl-project/sglang.git" "main" "${sglang_dirs[@]}"
        ;;
    all)
        clone_or_update "triton" "triton_skill" "https://github.com/triton-lang/triton.git" "main" "${triton_dirs[@]}"
        clone_or_update "cutlass" "cutlass_skill" "https://github.com/NVIDIA/cutlass.git" "main" "${cutlass_dirs[@]}"
        clone_or_update "sglang" "sglang_skill" "https://github.com/sgl-project/sglang.git" "main" "${sglang_dirs[@]}"
        ;;
    *)
        echo "Unknown repository: $TARGET"
        echo "Usage: bash update-repos.sh [triton|cutlass|sglang|all]"
        exit 1
        ;;
esac

echo ""
echo "=== Summary ==="
for sk in triton_skill cutlass_skill sglang_skill; do
    if [ -d "$SCRIPT_DIR/$sk/repos" ]; then
        du -sh "$SCRIPT_DIR/$sk/repos/"*/ 2>/dev/null
    fi
done
