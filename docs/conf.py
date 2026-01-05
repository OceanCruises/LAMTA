project = "LAMTA"
author = "OceanCruises"

extensions = [
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"

html_theme = "pydata_sphinx_theme"

exclude_patterns = ["_build"]
