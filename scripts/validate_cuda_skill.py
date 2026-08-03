#!/usr/bin/env python3
"""Validate the CUDA skill structure and representative documentation content."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "cuda_skill"
REFS = SKILL / "references"

EXPECTED_MARKDOWN_COUNTS = {
    "ptx-docs": 489,
    "cuda-runtime-docs": 106,
    "cuda-driver-docs": 135,
    "cuda-guide": 44,
    "best-practices-guide": 73,
    "ncu-docs": 9,
    "nsys-docs": 5,
}

GUIDES = [
    REFS / "MANIFEST.md",
    REFS / "ptx-isa.md",
    REFS / "cuda-runtime.md",
    REFS / "cuda-driver.md",
    REFS / "ncu-guide.md",
    REFS / "nsys-guide.md",
    REFS / "debugging-tools.md",
    REFS / "nvtx-patterns.md",
    REFS / "performance-traps.md",
]


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def validate_structure(self) -> None:
        required = [SKILL / "SKILL.md", *GUIDES]
        for path in required:
            self.check(path.is_file(), f"missing file: {path.relative_to(ROOT)}")

        for directory, expected in EXPECTED_MARKDOWN_COUNTS.items():
            path = REFS / directory
            self.check(path.is_dir(), f"missing directory: {path.relative_to(ROOT)}")
            if path.is_dir():
                actual = sum(1 for _ in path.rglob("*.md"))
                self.check(
                    actual == expected,
                    f"{directory}: expected {expected} Markdown files, found {actual}",
                )
                self.check((path / "INDEX.md").is_file(), f"{directory}: missing INDEX.md")

    def validate_skill_file(self) -> None:
        path = SKILL / "SKILL.md"
        if not path.is_file():
            return

        text = path.read_text(encoding="utf-8")
        self.check(text.startswith("---\n"), "SKILL.md: missing YAML frontmatter")
        self.check("name: cuda-skill" in text, "SKILL.md: wrong skill name")
        self.check("description:" in text, "SKILL.md: missing description")
        self.check("MANIFEST.md" in text, "SKILL.md: does not route version checks to MANIFEST.md")
        self.check("~/.cursor/skills/cuda-skill" not in text, "SKILL.md: hardcodes Cursor path")
        self.check("ptx-simple" not in text, "SKILL.md: references removed ptx-simple content")

    def validate_guides(self) -> None:
        combined = ""
        for path in GUIDES:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            combined += text
            self.check(text.count("```") % 2 == 0, f"{path.name}: unbalanced code fence")
            self.check("\uf0c1" not in text, f"{path.name}: contains Sphinx anchor glyph")

        forbidden = {
            "ptx-simple": "references removed ptx-simple content",
            "LaunchStatistics": "uses obsolete NCU section identifier LaunchStatistics",
            "SchedulerStatistics": "uses obsolete NCU section identifier SchedulerStatistics",
            "WarpStateStatistics": "uses obsolete NCU section identifier WarpStateStatistics",
            "2025.3.1": "binds guidance to the old local NCU version",
        }
        for token, reason in forbidden.items():
            self.check(token not in combined, f"guides: {reason}")

        self.check("Nsight Compute | 2026.2.1" in combined, "manifest: wrong NCU version")
        self.check("Nsight Systems | 2026.3" in combined, "manifest: wrong Nsys version")

    def validate_generated_docs(self) -> None:
        bad_nav = "Search In: Entire Site Just This Document clear search search"
        for directory in EXPECTED_MARKDOWN_COUNTS:
            for path in (REFS / directory).rglob("*.md"):
                text = path.read_text(encoding="utf-8")
                relative = path.relative_to(ROOT)
                self.check("\uf0c1" not in text, f"{relative}: contains anchor glyph")
                self.check(bad_nav not in text, f"{relative}: contains navigation noise")
                self.check(
                    not any(line.endswith((" ", "\t")) for line in text.splitlines()),
                    f"{relative}: contains trailing whitespace",
                )

        ptx_page = (
            REFS
            / "ptx-docs/9-instruction-set/"
            "9.7.16.5-asynchronous-warpgroup-level-matrix-multiply-accumulate-"
            "operation-usingwgmmamma_asyncinstruction.md"
        )
        if ptx_page.is_file():
            text = ptx_page.read_text(encoding="utf-8")
            self.check("PTX ISA Notes" in text, "focused WGMMA page lost PTX ISA Notes")
            self.check("Target ISA Notes" in text, "focused WGMMA page lost Target ISA Notes")

        version_checks = [
            (REFS / "ncu-docs/ReleaseNotes.md", "2026.2.1"),
            (REFS / "nsys-docs/ReleaseNotes.md", "2026.3"),
        ]
        for path, token in version_checks:
            if path.is_file():
                self.check(token in path.read_text(encoding="utf-8"), f"{path.name}: missing {token}")

    def validate_indexes_and_symbols(self) -> None:
        indexes = [REFS / directory / "INDEX.md" for directory in EXPECTED_MARKDOWN_COUNTS]
        for path in [*GUIDES, *indexes]:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = match.group(1)
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                target_path = target.split("#", 1)[0]
                self.check(
                    (path.parent / target_path).exists(),
                    f"{path.relative_to(ROOT)}: broken link to {target}",
                )

        representative_queries = {
            "ptx-docs": "wgmma.mma_async",
            "cuda-runtime-docs": "cudaStreamSynchronize",
            "cuda-driver-docs": "cuMemMap",
            "ncu-docs": "--query-metrics",
            "nsys-docs": "jsonlines",
        }
        for directory, token in representative_queries.items():
            found = any(
                token in path.read_text(encoding="utf-8")
                for path in (REFS / directory).rglob("*.md")
            )
            self.check(found, f"{directory}: representative query not found: {token}")

    def validate_scripts(self) -> None:
        scraper = ROOT / "scrape_docs.py"
        installer = ROOT / "install.sh"

        try:
            ast.parse(scraper.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            self.errors.append(f"scrape_docs.py: {exc}")

        for path in [scraper, installer]:
            text = path.read_text(encoding="utf-8")
            self.check("shutil.rmtree" not in text, f"{path.name}: contains recursive deletion")
            self.check("rm -rf" not in text, f"{path.name}: contains recursive deletion")

        shell_check = subprocess.run(
            ["bash", "-n", str(installer)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.check(shell_check.returncode == 0, f"install.sh: {shell_check.stderr.strip()}")

    def run(self) -> int:
        self.validate_structure()
        self.validate_skill_file()
        self.validate_guides()
        self.validate_generated_docs()
        self.validate_indexes_and_symbols()
        self.validate_scripts()

        if self.errors:
            print(f"CUDA skill validation failed with {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"- {error}")
            return 1

        print("CUDA skill validation passed")
        for directory, expected in EXPECTED_MARKDOWN_COUNTS.items():
            print(f"- {directory}: {expected} Markdown files")
        return 0


if __name__ == "__main__":
    sys.exit(Validator().run())
