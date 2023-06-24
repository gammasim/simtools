# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import yaml

import simtools.version

sys.path.insert(0, os.path.abspath("../../simtools"))
sys.path.insert(0, os.path.abspath("../../applications"))
sys.path.insert(0, os.path.abspath("../.."))


def get_authors_from_citation_file(file_name):
    """
    Read list of authors from CITATION.cff file

    """
    try:
        with open("../../CITATION.cff") as file:
            citation = yaml.safe_load(file)
    except FileNotFoundError:
        raise

    author = ""
    try:
        for person in citation["authors"]:
            author = author + person["given-names"] + " " + person["family-names"]
            author += " (" + person["affiliation"] + "), "
    except KeyError:
        pass
    return author[:-2]


# -- Project information -----------------------------------------------------

project = "simtools"
copyright = "2022, gammasim-tools, simtools developers"
author = get_authors_from_citation_file("../CITATION.cff")
rst_epilog = f"""
.. |author| replace:: {author}
"""

# The short X.Y version
version = str(simtools.version.__version__)
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # 'sphinx.ext.napoleon',  # make sphinx understand google docstring format
    "sphinx.ext.todo",  # enabling todo's
    # "sphinx.ext.autosectionlabel",  # allows refs to section by its name
    "numpydoc",
]

autosectionlabel_prefix_document = True

# autodoc_default_options = {"members": True, "undoc-members": True, "private-members": True}

# Display todos by setting to True
todo_include_todos = True

autodoc_mock_imports = [
    "matplotlib",
    "yaml",
    "numpy",
    "astropy",
    "bson",
    "pymongo",
    "gridfs",
    "scipy",
    "cycler",
    "eventio",
]

# Change the look of autodoc classes
# napoleon_use_ivar = True

numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
html_theme = "bizstyle"

html_title = f"{project} v{version} Manual"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_sidebars = {
    "**": ["globaltoc.html", "sourcelink.html", "searchbox.html"],
    "using/windows": ["windowssidebar.html", "searchbox.html"],
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "simtoolsdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "simtools.tex",
        "simtools Documentation",
        author,
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "simtools", "simtools Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "simtools",
        "simtools Documentation",
        author,
        "simtools",
        "Prototype of a software package for the Simulation System of CTA Observatory"
        "Miscellaneous",
    ),
]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}


# -- Raul: making sphinx add classes' __init__ always


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
