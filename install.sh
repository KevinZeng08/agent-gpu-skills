#!/bin/bash
# GPU skill installer.
# Usage: bash install.sh [--agent NAME] [--copy] [--skill NAME] [--target-dir DIR]
#
# Installs for Cursor by default. Use --agent to select another tool.
#
# Installation modes (hybrid mode by default):
#   - Skill directory: regular directory, because most tools do not follow symlinked directories.
#   - SKILL.md: regular copied file.
#   - Subdirectories and files such as repos and references: symlinks to this repository.
#
# --copy enables full-copy mode for environments that cannot use symlinks.
#
# Safety policy: create or overwrite matching files only. Never remove existing directories or links.
# Exit on path conflicts so the user can decide how to resolve them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AGENT="cursor"
COPY_MODE=false
SELECTED_SKILL=""
TARGET_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent) AGENT="$2"; shift 2 ;;
        --copy)  COPY_MODE=true; shift ;;
        --skill) SELECTED_SKILL="$2"; shift 2 ;;
        --target-dir) TARGET_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash install.sh [--agent NAME] [--copy] [--skill NAME] [--target-dir DIR]"
            echo ""
            echo "Initial setup:"
            echo "  bash update-repos.sh    # Fetch source repositories."
            echo "  bash install.sh         # Install for Cursor (default and validated)."
            echo ""
            echo "Install for other tools (not validated):"
            echo "  bash install.sh --agent claude   # Claude Code (~/.claude/skills/)"
            echo "  bash install.sh --agent codex    # Codex (~/.codex/skills/)"
            echo "  bash install.sh --agent gemini   # Gemini CLI (~/.gemini/skills/)"
            echo ""
            echo "Options:"
            echo "  --copy       Copy all files instead of creating symlinks."
            echo "  --skill      Install one skill only, for example cuda-skill."
            echo "  --target-dir Override the agent's default skill root."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

get_skill_dir() {
    if [ -n "$TARGET_DIR" ]; then
        echo "$TARGET_DIR"
        return
    fi

    case $1 in
        cursor) echo "${HOME}/.cursor/skills" ;;
        claude) echo "${HOME}/.claude/skills" ;;
        codex)  echo "${CODEX_HOME:-${HOME}/.codex}/skills" ;;
        gemini) echo "${HOME}/.gemini/skills" ;;
        qoder)  echo "${HOME}/.qoder/skills" ;;
        *)      echo "Unknown agent: $1" >&2; return 1 ;;
    esac
}

if [ ! -d "$SCRIPT_DIR/cuda_skill" ]; then
    echo "Error: cuda_skill/ was not found."
    echo "Run this script from the repository root."
    exit 1
fi

declare -A SKILLS
SKILLS[cuda-skill]="cuda_skill"
SKILLS[triton-skill]="triton_skill"
SKILLS[cutlass-skill]="cutlass_skill"
SKILLS[sglang-skill]="sglang_skill"
SKILLS[cutedsl-skill]="cutedsl_skill"
SKILLS[ncu-analysis]="ncu-analysis"
SKILLS[nccl-skill]="nccl_skill"
SKILLS[nvshmem-skill]="nvshmem_skill"

if [ -n "$SELECTED_SKILL" ] && [ -z "${SKILLS[$SELECTED_SKILL]+set}" ]; then
    echo "Error: unknown skill: $SELECTED_SKILL"
    exit 1
fi

install_to_agent() {
    local agent=$1
    local SKILL_DIR
    SKILL_DIR=$(get_skill_dir "$agent")

    echo "================================"
    echo "Installing for $agent ($SKILL_DIR)"
    echo "================================"
    echo ""

    mkdir -p "$SKILL_DIR"

    local install_failed=0

    for skill_name in "${!SKILLS[@]}"; do
        if [ -n "$SELECTED_SKILL" ] && [ "$skill_name" != "$SELECTED_SKILL" ]; then
            continue
        fi

        local src_dir="${SKILLS[$skill_name]}"
        local src_path="$SCRIPT_DIR/$src_dir"
        local target="$SKILL_DIR/$skill_name"

        echo "--- $skill_name ---"

        # Report the legacy name without removing it.
        if [ "$skill_name" = "triton-skill" ]; then
            local old_target="$SKILL_DIR/triton-gluon-skill"
            if [ -L "$old_target" ] || [ -d "$old_target" ]; then
                echo "  Notice: legacy triton-gluon-skill detected and left unchanged."
            fi
        fi

        if [ ! -d "$src_path" ]; then
            echo "  Skip: $src_dir/ does not exist."
            continue
        fi

        if [ -L "$target" ] || { [ -e "$target" ] && [ ! -d "$target" ]; }; then
            echo "  Conflict: $target is not a regular directory and was left unchanged."
            install_failed=1
            continue
        fi

        if [ "$COPY_MODE" = true ]; then
            mkdir -p "$target"
            cp -a "$src_path/." "$target/"
            echo "  Merged copy: $src_path -> $target"
        else
            mkdir -p "$target"
            cp "$src_path/SKILL.md" "$target/SKILL.md"
            echo "  Copied: SKILL.md"

            for item in "$src_path"/*; do
                local item_name
                item_name="$(basename "$item")"
                [ "$item_name" = "SKILL.md" ] && continue
                [[ "$item_name" == update-*.sh ]] && continue

                local target_item="$target/$item_name"
                if [ -L "$target_item" ]; then
                    if [ "$(readlink -f "$target_item")" = "$(readlink -f "$item")" ]; then
                        echo "  Already linked: $item_name"
                    else
                        echo "  Conflict: $target_item points elsewhere and was left unchanged."
                        install_failed=1
                    fi
                elif [ -e "$target_item" ]; then
                    echo "  Conflict: $target_item exists and is not a symlink. It was left unchanged."
                    install_failed=1
                else
                    ln -s "$item" "$target_item"
                    echo "  Linked: $item_name"
                fi
            done
        fi
    done
    echo ""

    if [ "$install_failed" -ne 0 ]; then
        echo "Installation incomplete: path conflicts were found. No existing content was removed."
        return 1
    fi
}

install_to_agent "$AGENT"

# Validate the installation.
echo "================================"
echo "Validation"
echo "================================"
echo ""

verify_agent() {
    local agent=$1
    local SKILL_DIR
    SKILL_DIR=$(get_skill_dir "$agent")
    local PASS=0 FAIL=0

    echo "--- $agent ($SKILL_DIR) ---"

    check() {
        if [ -e "$1" ]; then
            echo "  OK: $2"
            PASS=$((PASS + 1))
        else
            echo "  Missing: $2"
            FAIL=$((FAIL + 1))
        fi
    }

    for skill_name in "${!SKILLS[@]}"; do
        if [ -n "$SELECTED_SKILL" ] && [ "$skill_name" != "$SELECTED_SKILL" ]; then
            continue
        fi
        check "$SKILL_DIR/$skill_name/SKILL.md" "$skill_name/SKILL.md"
    done

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "cuda-skill" ]; then
        local REFS="$SKILL_DIR/cuda-skill/references"
        check "$REFS/MANIFEST.md" "CUDA documentation manifest"
        check "$REFS/ptx-docs/INDEX.md" "CUDA docs: ptx-docs"
        check "$REFS/cuda-guide/INDEX.md" "CUDA docs: cuda-guide"
        check "$REFS/cuda-runtime-docs/INDEX.md" "CUDA docs: cuda-runtime-docs"
        check "$REFS/cuda-driver-docs/INDEX.md" "CUDA docs: cuda-driver-docs"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "triton-skill" ]; then
        local TRITON_REPO="$SKILL_DIR/triton-skill/repos/triton"
        check "$TRITON_REPO/python/tutorials" "Triton tutorials"
        check "$TRITON_REPO/python/tutorials/gluon" "Gluon tutorials"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "cutlass-skill" ]; then
        local CUTLASS_REPO="$SKILL_DIR/cutlass-skill/repos/cutlass"
        check "$CUTLASS_REPO/python/CuTeDSL" "CuTeDSL source"
        check "$CUTLASS_REPO/include/cute" "CuTe headers"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "sglang-skill" ]; then
        local SGLANG_REPO="$SKILL_DIR/sglang-skill/repos/sglang"
        check "$SGLANG_REPO/python/sglang/srt" "SGLang SRT core"
        check "$SGLANG_REPO/sgl-kernel/csrc" "sgl-kernel CUDA source"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "cutedsl-skill" ]; then
    check "$SKILL_DIR/cutedsl-skill/references/add-inline-ptx.md" "CuTeDSL: add-inline-ptx"
        check "$SKILL_DIR/cutedsl-skill/references/tma-guide.md" "CuTeDSL: tma-guide"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "ncu-analysis" ]; then
        check "$SKILL_DIR/ncu-analysis/SKILL.md" "NCU Analysis"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "nccl-skill" ]; then
        check "$SKILL_DIR/nccl-skill/references/env.md" "NCCL: NCCL_* 环境变量"
        check "$SKILL_DIR/nccl-skill/references/api/comms.md" "NCCL: C API (comms)"
    fi

    if [ -z "$SELECTED_SKILL" ] || [ "$SELECTED_SKILL" = "nvshmem-skill" ]; then
        check "$SKILL_DIR/nvshmem-skill/references/env.md" "NVSHMEM: NVSHMEM_* 环境变量"
        check "$SKILL_DIR/nvshmem-skill/references/api/rma.md" "NVSHMEM: C API (rma)"
    fi

    echo "  Validation: $PASS passed, $FAIL failed."
    echo ""

    if [ $FAIL -gt 0 ]; then
        echo "  Hint: missing paths may reduce skill search coverage."
        echo "    - CUDA docs: scrape into a temporary directory, then review and merge into references/."
        echo "    - Source repositories: run 'bash update-repos.sh'."
        echo ""
    fi
}

verify_agent "$AGENT"

echo "Installation complete."
