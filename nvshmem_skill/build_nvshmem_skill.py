#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "beautifulsoup4",
#   "html2text",
#   "requests",
# ]
# ///
"""
Regenerate the NVSHMEM documentation skill `references/` from the rendered
NVIDIA docs site.

The NVSHMEM doc source is not published in a clean, buildable form (unlike NCCL,
whose RST source ships in its GitHub repo), so this mirrors the *rendered*
Sphinx HTML at https://docs.nvidia.com/nvshmem/api/ and converts each page to
clean, grep-able Markdown. Same idea as the repo's `scrape_docs.py` /
technillogue's ptx-isa-markdown: a small always-on SKILL.md plus a searchable
references/ tree.

Usage:
    nvshmem_skill/build_nvshmem_skill.py            # fetch + convert all pages
    nvshmem_skill/build_nvshmem_skill.py --offline  # reuse cached HTML only

Hand-written files (SKILL.md, README.md, references/INDEX.md) are never touched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import html2text
import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://docs.nvidia.com/nvshmem/api/"
DOC_VERSION = "3.7.0"

SCRIPT_DIR = Path(__file__).resolve().parent
REF_DIR = SCRIPT_DIR / "references"
CACHE_DIR = SCRIPT_DIR / ".cache-html"

# Source HTML page (relative to BASE_URL) -> output Markdown file (under references/).
# Curated, friendly layout mirroring the nccl_skill tree. Navigation-only pages
# (genindex, py-modindex, search) and the license text are intentionally omitted.
PAGES: dict[str, str] = {
    # Guide / concepts
    "introduction.html": "introduction.md",
    "using.html": "using.md",
    "cuda-interactions.html": "cuda-model.md",
    "tma.html": "tma.md",
    "gen/mem-model.html": "memory-model.md",
    "gen/exec-model.html": "execution-model.md",
    "gen/const.html": "constants.md",
    "gen/handles.html": "handles.md",
    "gen/env.html": "env.md",
    # C/C++ API reference
    "api.html": "api/index.md",
    "api/overview.html": "api/overview.md",
    "gen/api/setup.html": "api/setup.md",
    "api/launch.html": "api/launch.md",
    "gen/api/memory.html": "api/memory.md",
    "gen/api/qp.html": "api/qp.md",
    "gen/api/teams.html": "api/teams.md",
    "gen/api/rma.html": "api/rma.md",
    "gen/api/amo.html": "api/amo.md",
    "gen/api/signal.html": "api/signal.md",
    "gen/api/collectives.html": "api/collectives.md",
    "gen/api/sync.html": "api/sync.md",
    "gen/api/ordering.html": "api/ordering.md",
    # Python bindings (NVSHMEM4Py)
    "api/language_bindings/index.html": "nvshmem4py/index.md",
    "api/language_bindings/python/index.html": "nvshmem4py/overview-index.md",
    "api/language_bindings/python/overview.html": "nvshmem4py/overview.md",
    "api/language_bindings/python/initialization.html": "nvshmem4py/initialization.md",
    "api/language_bindings/python/memory_management.html": "nvshmem4py/memory_management.md",
    "api/language_bindings/python/interoperability.html": "nvshmem4py/interoperability.md",
    "api/language_bindings/python/collectives.html": "nvshmem4py/collectives.md",
    "api/language_bindings/python/rma.html": "nvshmem4py/rma.md",
    "api/language_bindings/python/utils.html": "nvshmem4py/utils.md",
    "api/language_bindings/python/device/index.html": "nvshmem4py/device/index.md",
    "api/language_bindings/python/device/numba/index.html": "nvshmem4py/device/numba/index.md",
    "api/language_bindings/python/device/numba/collectives.html": "nvshmem4py/device/numba/collectives.md",
    "api/language_bindings/python/device/numba/rma.html": "nvshmem4py/device/numba/rma.md",
    "api/language_bindings/python/device/numba/amo.html": "nvshmem4py/device/numba/amo.md",
    "api/language_bindings/python/device/cute/index.html": "nvshmem4py/device/cute/index.md",
    "api/language_bindings/python/device/cute/collectives.html": "nvshmem4py/device/cute/collectives.md",
    "api/language_bindings/python/device/cute/rma.html": "nvshmem4py/device/cute/rma.md",
    "api/language_bindings/python/device/cute/amo.html": "nvshmem4py/device/cute/amo.md",
    # Examples & FAQ
    "examples.html": "examples.md",
    "examples/language_bindings/index.html": "examples/language_bindings.md",
    "examples/language_bindings/python/index.html": "examples/python.md",
    "faq.html": "faq.md",
}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; nvshmem-skill-builder)"})
    return s


def fetch_html(s: requests.Session, rel: str, offline: bool) -> str | None:
    cache = CACHE_DIR / rel
    if offline:
        return cache.read_text(encoding="utf-8") if cache.is_file() else None
    url = urljoin(BASE_URL, rel)
    try:
        resp = s.get(url, timeout=30)
        resp.raise_for_status()
        # NVIDIA docs are UTF-8 but omit the charset header, so requests would
        # otherwise fall back to ISO-8859-1 and mangle ' / § / etc.
        resp.encoding = "utf-8"
    except Exception as e:  # noqa: BLE001
        print(f"  ! fetch failed {url}: {e}", file=sys.stderr)
        return cache.read_text(encoding="utf-8") if cache.is_file() else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text, encoding="utf-8")
    return resp.text


def make_converter() -> html2text.HTML2Text:
    h = html2text.HTML2Text()
    h.body_width = 0
    h.ignore_links = True  # cross-page hrefs don't map to renamed .md files
    h.ignore_images = False  # keep diagrams; download + rewrite paths below
    h.ignore_emphasis = False
    h.unicode_snob = True
    h.decode_errors = "ignore"
    h.mark_code = False
    return h


_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
IMG_DIR = REF_DIR / "images"


def localize_images(md: str, page_url: str, out_rel: str,
                    s: requests.Session, offline: bool) -> str:
    """Download referenced images into references/images/ and rewrite links."""
    def repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2).strip()
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        # Some source images carry their own path as alt text; drop that noise.
        if "/" in alt or alt.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
            alt = ""
        name = Path(src.split("?")[0]).name
        dest = IMG_DIR / name
        if not dest.is_file() and not offline:
            try:
                r = s.get(urljoin(page_url, src), timeout=30)
                r.raise_for_status()
                IMG_DIR.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"  ! image fetch failed {src}: {e}", file=sys.stderr)
                return alt  # drop the broken image, keep the caption text
        if not dest.is_file():
            return alt
        depth = Path(out_rel).parent.parts
        prefix = "../" * len(depth)
        return f"![{alt}]({prefix}images/{name})"

    return _IMG.sub(repl, md)


def clean_article(soup: BeautifulSoup) -> Tag:
    """Pull the Sphinx article body and normalize it for Markdown conversion."""
    body = soup.find("div", attrs={"itemprop": "articleBody"})
    if body is None:
        body = soup.find("div", attrs={"role": "main"}) or soup.find("body")
    if body is None:
        raise ValueError("no article body found")

    # Drop permalink pilcrows (¶) and any leftover nav widgets.
    for a in body.select("a.headerlink"):
        a.decompose()
    for sel in (".sphinx-tabs-tab",):
        for el in body.select(sel):
            el.unwrap()

    # Render C/C++ signature definition terms as inline code so they stay on one
    # readable, grep-able line instead of a soup of italics from <em> params.
    for dl in body.find_all("dl"):
        classes = set(dl.get("class") or [])
        if classes & {"function", "type", "var", "member", "macro", "typedef",
                      "enum", "struct", "class", "cpp", "c"}:
            for dt in dl.find_all("dt", recursive=False):
                text = re.sub(r"\s+", " ", dt.get_text(" ", strip=True)).strip()
                if not text:
                    continue
                dt.clear()
                code = soup.new_tag("code")
                code.string = text
                dt.append(code)
            # Empty <dd></dd> would render as a stray blank definition; drop them.
            for dd in dl.find_all("dd", recursive=False):
                if not dd.get_text(strip=True):
                    dd.decompose()
    return body


_BLANKS = re.compile(r"\n{3,}")
_ANCHOR_LINK = re.compile(r"\[([^\]]+)\]\(#[^)]*\)")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def postprocess(md: str, title: str | None) -> str:
    md = _ANCHOR_LINK.sub(r"\1", md)          # [text](#anchor) -> text
    md = _TRAILING_WS.sub("\n", md)
    md = _BLANKS.sub("\n\n", md).strip()
    if title and not md.lstrip().startswith("# "):
        md = f"# {title}\n\n{md}"
    return md + "\n"


def page_title(soup: BeautifulSoup) -> str | None:
    h1 = soup.select_one("div[itemprop=articleBody] h1") or soup.find("h1")
    if h1:
        return re.sub(r"\s+", " ", h1.get_text(" ", strip=True)).strip().rstrip("¶").strip()
    t = soup.find("title")
    return t.get_text(strip=True).split("—")[0].strip() if t else None


def convert(rel: str, out_rel: str, html: str, conv: html2text.HTML2Text,
            s: requests.Session, offline: bool) -> int:
    soup = BeautifulSoup(html, "html.parser")
    title = page_title(soup)
    body = clean_article(soup)
    md = postprocess(conv.handle(str(body)), title)
    md = localize_images(md, urljoin(BASE_URL, rel), out_rel, s, offline)
    dest = REF_DIR / out_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    return md.count("\n") + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="use cached HTML only")
    args = ap.parse_args()

    print(f">> NVSHMEM docs {DOC_VERSION}  ->  {REF_DIR}")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    s = session()
    conv = make_converter()

    ok = 0
    total_lines = 0
    for rel, out_rel in PAGES.items():
        html = fetch_html(s, rel, args.offline)
        if html is None:
            print(f"  ! skip {rel} (no html)")
            continue
        try:
            lines = convert(rel, out_rel, html, conv, s, args.offline)
        except Exception as e:  # noqa: BLE001
            print(f"  ! convert failed {rel}: {e}", file=sys.stderr)
            continue
        total_lines += lines
        ok += 1
        print(f"  ok {out_rel} ({lines} lines)")

    print(f">> wrote {ok}/{len(PAGES)} pages, {total_lines} lines into {REF_DIR}")


if __name__ == "__main__":
    main()
