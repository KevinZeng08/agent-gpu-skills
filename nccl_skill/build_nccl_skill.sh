#!/usr/bin/env bash
#
# Regenerate the NCCL documentation skill references from the in-repo doc source.
#
# This drives Sphinx + sphinx-markdown-builder over docs/userguide/source and
# emits a clean, grep-able Markdown mirror into nccl_skill/references/.
#
# Why Sphinx instead of a hand-rolled RST converter: the Python bindings pages
# (nccl4py/*) use Sphinx autodoc, so their text lives in Python docstrings, not
# the RST. conf.py.in already mocks the compiled/heavy deps
# (autodoc_mock_imports), so autodoc can introspect the pure-Python `nccl`
# package without building the C extension.
#
# Usage:
#   nccl_skill/build_nccl_skill.sh
#
# Hand-written files (SKILL.md, README.md, references/INDEX.md) are NOT touched;
# only generated documentation pages and image assets are (re)written.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOC_SRC="$REPO_ROOT/docs/userguide/source"
BINDINGS="$REPO_ROOT/bindings/nccl4py"
VERSION_MK="$REPO_ROOT/makefiles/version.mk"
REQS="$REPO_ROOT/docs/userguide/requirements.txt"
REF="$SCRIPT_DIR/references"

VENV="${VENV:-$SCRIPT_DIR/.venv}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# --- Resolve version from makefiles/version.mk -----------------------------
ver() { grep -E "^$1[[:space:]]*:?=" "$VERSION_MK" | grep -oE '[0-9]+' | head -1; }
MAJOR="$(ver NCCL_MAJOR)"; MINOR="$(ver NCCL_MINOR)"; PATCH="$(ver NCCL_PATCH)"
echo ">> NCCL version: ${MAJOR}.${MINOR}.${PATCH}"

# --- Python venv + dependencies --------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo ">> Creating venv at $VENV"
  python3 -m venv "$VENV"
fi
echo ">> Installing doc dependencies"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$REQS" sphinx-markdown-builder

# --- Stage source + generate conf.py ---------------------------------------
echo ">> Staging doc source"
cp -r "$DOC_SRC" "$WORK/src"
MAJOR="$MAJOR" MINOR="$MINOR" PATCH="$PATCH" BINDINGS="$BINDINGS" \
SRC_IN="$DOC_SRC/conf.py.in" OUT_CONF="$WORK/src/conf.py" \
"$VENV/bin/python" - <<'PY'
import os, pathlib
s = pathlib.Path(os.environ['SRC_IN']).read_text()
s = (s.replace('${nccl:Major}', os.environ['MAJOR'])
      .replace('${nccl:Minor}', os.environ['MINOR'])
      .replace('${nccl:Patch}', os.environ['PATCH']))
# The staged copy lives in a tempdir, so the original relative bindings path
# (../../bindings/nccl4py) no longer resolves. Point it at the real repo.
s = s.replace(
    "sys.path.insert(0, str(Path('../../bindings/nccl4py').resolve()))",
    "sys.path.insert(0, %r)" % os.environ['BINDINGS'])
# Register the Markdown builder.
s = s.replace("extensions = [", "extensions = [\n    'sphinx_markdown_builder',")
pathlib.Path(os.environ['OUT_CONF']).write_text(s)
print('   conf.py generated')
PY

# --- Build Markdown ---------------------------------------------------------
echo ">> Building Markdown with Sphinx"
"$VENV/bin/python" -m sphinx -b markdown -q "$WORK/src" "$WORK/out"

# --- Copy + post-process into references/ -----------------------------------
echo ">> Post-processing into $REF"
OUT="$WORK/out" SRC="$WORK/src" REF="$REF" "$VENV/bin/python" - <<'PY'
import os, re, shutil, pathlib
out = pathlib.Path(os.environ['OUT'])
src = pathlib.Path(os.environ['SRC'])
ref = pathlib.Path(os.environ['REF'])
ref.mkdir(parents=True, exist_ok=True)

# 1. Copy every generated Markdown page (preserving the directory layout).
copied = 0
for md in sorted(out.rglob('*.md')):
    dest = ref / md.relative_to(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(md, dest)
    copied += 1
print(f'   copied {copied} markdown pages')

# 2. Copy diagram images used by usage/collectives.md.
img_src = src / 'usage' / 'images'
if img_src.is_dir():
    shutil.copytree(img_src, ref / 'usage' / 'images', dirs_exist_ok=True)
    print(f'   copied images: {len(list(img_src.glob("*")))}')

# 3. Fix image links. The builder emits paths relative to the doc root
#    (usage/images/foo.png); rewrite them relative to each usage/*.md file.
for md in (ref / 'usage').glob('*.md'):
    t = md.read_text()
    t2 = t.replace('](usage/images/', '](images/')
    if t2 != t:
        md.write_text(t2)
        print(f'   fixed image links in {md.relative_to(ref)}')

# 4. Strip the Sphinx "Indices and tables" footer from index.md: it links to
#    genindex / modindex / search pages that do not exist in a Markdown build.
index_md = ref / 'index.md'
if index_md.is_file():
    t = index_md.read_text()
    cut = t.find('\n# Indices and tables')
    if cut != -1:
        index_md.write_text(t[:cut].rstrip() + '\n')
        print('   stripped dead "Indices and tables" footer from index.md')

# 5. Re-insert the one admonition the Markdown builder drops: the
#    "Application responsibilities" block under NCCL_GRAPH_STREAM_ORDERING.
env_md = ref / 'env.md'
if env_md.is_file():
    t = env_md.read_text()
    needle = '`graphStreamOrdering` for details.'
    admonition = (
        '\n\n> **Application responsibilities**\n>\n'
        '> With `NCCL_GRAPH_STREAM_ORDERING=0` (or `graphStreamOrdering` `0`), NCCL stops '
        'enforcing device-side serialization of communication kernels on the graph-capture '
        'path. The application must then guarantee:\n>\n'
        '> 1. **Serialization on the GPU.** NCCL operations must not overlap: at most one '
        'NCCL operation may execute on a given GPU at a time.\n'
        '> 2. **Scope.** The rule applies across communicators, across different captured '
        'graphs, and between captured and uncaptured NCCL work. The ordering must hold at '
        'replay / execution, not only as expressed at capture time.\n'
        '> 3. **How to satisfy it.** The simplest approach is to enqueue **all** NCCL '
        'operations on the **same CUDA stream**. Equivalent serialization can be achieved '
        'with device-wide synchronization and/or CUDA event dependencies between streams.\n>\n'
        '> Network (proxy) transports are unaffected by this setting; NCCL continues to '
        'provide its normal host-side ordering guarantees for those transports regardless '
        'of the value of `NCCL_GRAPH_STREAM_ORDERING`.')
    if 'Application responsibilities' not in t and t.count(needle) == 1:
        i = t.find(needle) + len(needle)
        env_md.write_text(t[:i] + admonition + t[i:])
        print('   patched dropped admonition in env.md')

print('   done')
PY

echo ">> References written to $REF"
