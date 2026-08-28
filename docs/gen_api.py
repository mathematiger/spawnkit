"""Generate one API page per module, and the nav that lists them.

Run by mkdocs-gen-files at build time. Writing a page per module rather than one long page keeps
each module's docstring — which in this package carries the reasoning, not just the signature — at
the top of its own page where it reads as prose.
"""

from pathlib import Path

import mkdocs_gen_files

SOURCE = Path("src")
nav = mkdocs_gen_files.Nav()

for path in sorted(SOURCE.rglob("*.py")):
    module_path = path.relative_to(SOURCE).with_suffix("")
    doc_path = path.relative_to(SOURCE).with_suffix(".md")
    parts = tuple(module_path.parts)

    if parts[-1].startswith("_") and parts[-1] != "__init__":
        continue  # private modules are implementation detail, not API
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    if not parts:
        continue

    nav[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(Path("api") / doc_path, "w") as page:
        page.write(f"# `{'.'.join(parts)}`\n\n::: {'.'.join(parts)}\n")
    mkdocs_gen_files.set_edit_path(Path("api") / doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as summary:
    summary.writelines(nav.build_literate_nav())
