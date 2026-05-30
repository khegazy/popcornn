"""Generate one Markdown stub per `popcornn` module for mkdocstrings.

Driven by the `mkdocs-gen-files` plugin (configured in mkdocs.yml).
On each build, walks `popcornn/`, drops `reference/<dotted>.md` stubs
that contain a single `::: popcornn.<dotted>` directive, and writes
`reference/SUMMARY.md` so `mkdocs-literate-nav` can build the
sidebar tree automatically.

Editing the source modules' docstrings is the only thing needed to
keep these pages up to date — this script doesn't need to change.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files


SRC_ROOT = Path("popcornn")
REF_ROOT = Path("reference")

nav = mkdocs_gen_files.Nav()

for path in sorted(SRC_ROOT.rglob("*.py")):
    module_path = path.relative_to(SRC_ROOT.parent).with_suffix("")
    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    if not parts:
        continue

    doc_path = Path(*parts).with_suffix(".md")
    full_doc_path = REF_ROOT / doc_path

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# `{'.'.join(parts)}`\n\n")
        fd.write(f"::: {'.'.join(parts)}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(REF_ROOT / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
