# NVSHMEM 3.7.0 Documentation Skill

A self-contained, grep-able mirror of the NVIDIA OpenSHMEM Library (NVSHMEM)
**3.7.0** documentation, packaged as a Cursor / Claude Agent Skill. Organized
after [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown)
and the sibling `nccl_skill`: a small always-on `SKILL.md` entry point plus a
`references/` tree of clean Markdown that the agent searches on demand.

Source documentation:
<https://docs.nvidia.com/nvshmem/api/index.html>

## Layout

```
nvshmem_skill/
  SKILL.md               # entry point: trigger terms, quick-reference, search recipes
  README.md              # this file
  build_nvshmem_skill.py # regenerates references/ by scraping the rendered docs site
  update-nvshmem.sh      # fetches matching NVSHMEM source into repos/nvshmem/ (gitignored)
  references/
    INDEX.md             # curated topic map + search recipes (hand-written)
    introduction.md  using.md  cuda-model.md  tma.md
    memory-model.md  execution-model.md  constants.md  handles.md  env.md
    api/                 # C/C++ API reference (index, setup, rma, amo, signal,
                         #   collectives, sync, ordering, teams, qp, memory, launch, overview)
    nvshmem4py/          # Python bindings (host + Numba-CUDA / CuTe device DSLs)
    examples/            # Python (NVSHMEM4Py) worked examples
    images/              # memory-model, bootstrap, and collective-op diagrams
    examples.md  faq.md
```

44 Markdown pages (~12k lines) covering the entire documentation set: the
concepts/programming-model chapters, every `NVSHMEM_*` environment variable, the
full C/C++ API (RMA, atomics, signaling, collectives, teams, synchronization,
ordering, QP device APIs), the NVSHMEM4Py Python bindings, worked examples, and
the troubleshooting FAQ.

## Using it

The skill activates from its `description` (NVSHMEM / OpenSHMEM GPU communication
work, `NVSHMEM_*` tuning, multi-GPU/multi-node put/get/collective hangs, looking
up `nvshmem*`/`nvshmemx*` symbols). Typical searches:

```bash
rg -n "nvshmem_put_signal" references/api/signal.md
rg -n -A4 "^`NVSHMEM_IB_ENABLE_IBGDA`" references/env.md
rg -rl "symmetric heap|active set" references/
rg -n "barrier_all|atomic_add" references/nvshmem4py/
```

Start from [`SKILL.md`](SKILL.md) (quick reference), [`references/INDEX.md`](references/INDEX.md)
(topic map), or [`references/api/index.md`](references/api/index.md) (full API
symbol inventory).

## Reading the source alongside the docs

To let the agent co-read NVSHMEM source with the documentation, fetch the matching
source (tag `v3.7.0-0`) into `repos/nvshmem/`:

```bash
nvshmem_skill/update-nvshmem.sh            # sparse checkout of key dirs (small)
nvshmem_skill/update-nvshmem.sh --full     # whole repo
nvshmem_skill/update-nvshmem.sh --ref main # a different tag/branch
```

It clones `src/`, `nvshmem4py/`, `examples/`, and `perftest/` from
<https://github.com/NVIDIA/nvshmem>, verifies key paths, and falls back to the
default branch if the pinned tag is unavailable. `repos/nvshmem/` is gitignored.
`SKILL.md` has a source map (which file implements what) and cross-source search
examples.

## Activating as a skill

This folder lives in the repo. To make it a discoverable Agent Skill, copy or
symlink it into a skills directory (or use the repo's `install.sh`):

```bash
# personal (all projects)
cp -r nvshmem_skill ~/.cursor/skills/nvshmem
# or project-scoped
cp -r nvshmem_skill .cursor/skills/nvshmem
```

## Regenerating

The pages are scraped from the rendered NVIDIA docs site and converted to clean
Markdown by [`build_nvshmem_skill.py`](build_nvshmem_skill.py) (a self-contained
`uv` script). Unlike NCCL — whose RST doc source ships in its GitHub repo and can
be driven through Sphinx — NVSHMEM does not publish a cleanly buildable doc
source, so this mirrors the rendered HTML instead. Re-run after the docs change:

```bash
nvshmem_skill/build_nvshmem_skill.py            # fetch + convert all pages
nvshmem_skill/build_nvshmem_skill.py --offline  # reuse cached HTML in .cache-html/
```

The script:

1. fetches each documentation page (page list curated in `PAGES`), caching raw
   HTML under `.cache-html/` (gitignored);
2. extracts the Sphinx article body, drops permalink pilcrows and empty
   definition rows, and renders C/C++ signature terms as inline code so they
   stay on one grep-able line;
3. converts to Markdown with `html2text` (no line wrapping), strips dangling
   in-page anchors, and writes the curated `references/` tree.

`SKILL.md`, `README.md`, and `references/INDEX.md` are hand-written and are not
overwritten by the build.

Documentation (c) NVIDIA Corporation. This is an unofficial offline conversion
for convenience; refer to NVIDIA's official documentation for the authoritative
reference.
