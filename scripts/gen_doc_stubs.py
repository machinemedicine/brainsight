import os
from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
# src is the root for the source code, in our case it is root of the repository
src = root

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    # Skip this very file and exclude from the docs
    if str(module_path).startswith("scripts/"):
        continue
    doc_path = path.relative_to(src).with_suffix(".md")
    # Create a folder for all documentation files to be generated in
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    # Update the nav object with full path
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print("::: " + ".".join(parts), file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)


# Custom function to build the literate nav with full paths
def build_custom_nav(nav):
    result = ""
    for item in nav.items():
        # We dont use the title, instead we parse the path
        line, _ = os.path.splitext(item.filename)
        line = ".".join(line.removesuffix("/__init__").split("/"))
        # line = item.title
        if item.filename:
            line = f"[{line}]({item.filename})"
        indent = "    " * item.level
        result += f"{indent}* {line}\n"
    return result


# Write the navigation structure to the SUMMARY.md
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write(build_custom_nav(nav))
