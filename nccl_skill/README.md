# NCCL 2.30 Documentation Skill

A self-contained, grep-able mirror of the NVIDIA Collective Communication
Library (NCCL) **2.30.7** user guide, packaged as a Cursor / Claude Agent
Skill. Organized after
[technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown):
a small always-on `SKILL.md` entry point plus a `references/` tree of clean
Markdown that the agent searches on demand.

Source documentation:
<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>

## Layout

```
nccl_skill/
  SKILL.md              # entry point: trigger terms, quick-reference, search recipes
  README.md             # this file
  build_nccl_skill.sh   # regenerates references/ from the in-repo doc source
  update-nccl.sh        # fetches matching NCCL source into repos/nccl/ (gitignored)
  references/
    INDEX.md            # curated topic map + search recipes
    index.md            # full auto-generated deep table of contents
    overview.md  setup.md  nccl1.md  examples.md  mpi.md  env.md
    usage/              # NCCL concepts (communicators, collectives, p2p, device API, ...)
    api/                # C API reference (comms, colls, types, device_*, param, ...)
    nccl4py/            # Python bindings reference (autodoc-resolved)
    troubleshooting/    # gpu, networking, runtime/mpi, perf, logging, ras
    usage/images/       # collective-operation diagrams
```

57 Markdown pages (~11k lines) covering the entire user guide, including the
C API, the `nccl4py` Python bindings, every `NCCL_*` environment variable, the
device-initiated communication API, and troubleshooting / RAS.

## Using it

The skill activates from its `description` (NCCL / nccl4py work, `NCCL_*`
tuning, multi-GPU/multi-node hangs, looking up `ncclXxx` symbols). Typical
searches:

```bash
rg -n "ncclCommInitRankConfig" references/api/comms.md
rg -n -A6 "^### NCCL_IB_HCA" references/env.md
rg -rl "fault tolerance" references/
```

Start from [`SKILL.md`](SKILL.md) (quick reference) or
[`references/INDEX.md`](references/INDEX.md) (topic map). The full deep table of
contents is [`references/index.md`](references/index.md).

## Reading the source alongside the docs

To let the agent co-read NCCL source with the documentation, fetch the matching
source (tag `v2.30.7-1`) into `repos/nccl/`:

```bash
nccl_skill/update-nccl.sh            # sparse checkout of key dirs (~12 MB)
nccl_skill/update-nccl.sh --full     # whole repo
nccl_skill/update-nccl.sh --ref master   # a different tag/branch
```

It clones `src/`, `bindings/nccl4py/`, `makefiles/`, `docs/examples/`, and
`plugins/` from <https://github.com/NVIDIA/nccl>, verifies key paths, and falls
back to the default branch if the pinned tag is unavailable. `repos/nccl/` is
gitignored. `SKILL.md` has a source map (which file implements what) and
cross-source search examples.

## Activating as a skill

This folder lives in the repo. To make it a discoverable Agent Skill, copy or
symlink it into a skills directory:

```bash
# personal (all projects)
cp -r nccl_skill ~/.cursor/skills/nccl
# or project-scoped
cp -r nccl_skill .cursor/skills/nccl
```

## Regenerating

The pages are generated from the in-repo doc source
([`docs/userguide/source`](../docs/userguide/source)) so they always match the
NCCL version in [`makefiles/version.mk`](../makefiles/version.mk). Re-run after
the docs change:

```bash
nccl_skill/build_nccl_skill.sh
```

The script:

1. creates a local `.venv` and installs the doc requirements plus
   `sphinx-markdown-builder`;
2. generates `conf.py` from `conf.py.in` (substitutes the version, points the
   bindings path at this repo, and registers the Markdown builder);
3. runs `sphinx-build -b markdown`;
4. copies the result into `references/`, copies diagram images, fixes image
   links, and re-inserts the one admonition the Markdown builder drops.

`SKILL.md`, `README.md`, and `references/INDEX.md` are hand-written and are not
overwritten by the build.

### Why Sphinx (not a regex converter)

The `nccl4py` pages use Sphinx **autodoc**, so their text lives in Python
docstrings, not the `.rst`. `conf.py.in` mocks the compiled/heavy dependencies
(`autodoc_mock_imports`), so autodoc introspects the pure-Python `nccl` package
without building the C extension. Driving Sphinx therefore yields a faithful
render of both the hand-written C-API pages and the docstring-based bindings.

Documentation (c) NVIDIA Corporation. This is an unofficial offline conversion
for convenience; refer to NVIDIA's official documentation for the authoritative
reference.
