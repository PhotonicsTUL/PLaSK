# -*- coding: utf-8 -*-
#
# PLaSK documentation build configuration file, created by
# sphinx-quickstart on Tue Oct  8 15:58:59 2013.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys, os, re   #re for processing signatures

sys.path.insert(0, os.path.abspath('./_lib'))

import plask

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.mathjax',
              'sphinx.ext.autosummary', 'sphinx_domain_xml', 'sphinx_autodoc_cpp',
#             'rst2pdf.pdfbuilder'
             ]


# Use Napoleon if available for pretty docstrings formatting

try:
    import sphinxcontrib.napoleon
except ImportError:
    import warnings
    warnings.warn("No napolen installed. API doc will not be properly formatted!")
else:
    extensions.append('sphinxcontrib.napoleon')
    napoleon_use_param = True
    napoleon_use_admonition_for_examples = True
    napoleon_numpy_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'PLaSK'
copyright = u'2013, Lodz University of Technology'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = plask.version[:10]
# The full version, including alpha/beta/rc tags.
release = plask.version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_api*', '_templates']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Autodoc options -------------------------------------------------------

autoclass_content = 'class'

autodoc_docstring_signature = False

# -- Autosummary options -------------------------------------------------------

autosummary_generate = True

# Use our own generate script

import sphinx.ext.autosummary

def process_generate_options(app):
    genfiles = app.config.autosummary_generate
    if genfiles and not hasattr(genfiles, '__len__'):
        env = app.builder.env
        genfiles = [env.doc2path(x,None) for x in env.found_docs
                    if os.path.isfile(env.doc2path(x))]
    if not genfiles:
        return
    from autosummary_generate import generate_autosummary_docs
    generate_autosummary_docs(genfiles, builder=app.builder,
                              warn=app.warn, info=app.info,
                              base_path=app.srcdir)

sphinx.ext.autosummary.process_generate_options = process_generate_options



# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
if 'epub' in sys.argv:
    html_theme = 'epub'
elif 'qthelp' in sys.argv:
    html_theme = 'qthelp'
else:
    html_theme = 'agogo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    #'linkcolor': '#7f1111',
    #'headerlinkcolor': '#fc573d'
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ['_themes']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'plask'


# -- Options for QtHelp output -------------------------------------------------

# The theme to use for QtHelp.  See the documentation for
# a list of builtin themes.
qthelp_theme = 'qthelp'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
qthelp_theme_options = {
}

# Output file base name for QtHelp help builder.
qthelp_basename = 'plask'


# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
#'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
'preamble': r"""
\usepackage{enumitem}
\setlistdepth{99}
\setlist[itemize]{labelsep=0.5em}
\DeclareUnicodeCharacter{2264}{\ensuremath{\le}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\ge}}
""",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'PLaSK.tex', u'PLaSK Documentation', u'M. Dems, P. Beling', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'plask', u'PLaSK Documentation',
     [u'M. Dems, P. Beling'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output ------------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'PLaSK', u'PLaSK Documentation',
   u'M. Dems, P. Beling', 'PLaSK', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# -- Options for Texinfo output ------------------------------------------------

epub_author = 'M. Dems, P. Beling'

epub_publisher = 'Lodz University of Technology'


# -- Options for inheritance diagram -------------------------------------------

#inheritance_graph_attrs = dict(rankdir="TB", size='""')


# -- Exec directive that allows to execute artbitray Python code--------------------
# http://stackoverflow.com/questions/7250659/python-code-to-generate-part-of-sphinx-documentation-is-it-possible

import sys
from os.path import basename

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    from sphinx.util.compat import Directive
except ImportError:
    from docutils.parsers.rst import Directive
from docutils import nodes, statemachine

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content), globals(), {'app': ExecDirective.app})
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (basename(source), self.lineno)), nodes.paragraph(text = str(sys.exc_info()[1])))]
        finally:
            sys.stdout = oldStdout

# -- Hook for better LaTeX autosummary output --------------------------------------

#import sphinx.writers.latex

#def latex_visit_table(self, node):
    #if self.table:
        #raise sphinx.writers.latex.UnsupportedError(
            #'%s:%s: nested tables are not yet implemented.' %
            #(self.curfilestack[-1], node.line or ''))
    #self.table = sphinx.writers.latex.Table()
    ##self.table.longtable = 'longtable' in node['classes']
    #self.table.longtable = False
    #self.tablebody = []
    #self.tableheaders = []
    ## Redirect body output until table is finished.
    #self._body = self.body
    #self.body = self.tablebody

#def latex_visit_thead(self, node):
    #self.table.had_head = True
    #if self.next_table_colspec:
        #if not self.table.longtable and self.next_table_colspec == 'll':
            #self.table.colspec = '{LL}\n'
        #else:
            #self.table.colspec = '{%s}\n' % self.next_table_colspec
    #self.next_table_colspec = None
    ## Redirect head output until header is finished. see visit_tbody.
    #self.body = self.tableheaders


# -- Register custom elements ------------------------------------------------------

def setup(app):
    app.add_directive('exec', ExecDirective)
    ExecDirective.app = app
    #sphinx.writers.latex.LaTeXTranslator.visit_table = latex_visit_table
    #sphinx.writers.latex.LaTeXTranslator.visit_thead = latex_visit_thead


# -- RST2PDF configuration ---------------------------------------------------------

# Grouping the document tree into PDF files. List of tuples
# (source start file, target name, title, author, options).
#
# If there is more than one author, separate them with \\.
# For example: r'Guido van Rossum\\Fred L. Drake, Jr., editor'

pdf_documents = [
    ('index', u'PLaSK', u'PLaSK Documentation', u'M. Dems\\P. Beling'),
]

# A comma-separated list of custom stylesheets. Example:
pdf_stylesheets = ['sphinx', 'kerning', 'a4']

# A list of folders to search for stylesheets. Example:
pdf_style_path = ['_styles']

# Create a compressed PDF
# Use True/False or 1/0
# Example: compressed=True
pdf_compressed = True

# A colon-separated list of folders to search for fonts. Example:
# pdf_font_path = ['/usr/share/fonts', '/usr/share/texmf-dist/fonts/']

# Language to be used for hyphenation support
#pdf_language = "en_US"

# Mode for literal blocks wider than the frame. Can be
# overflow, shrink or truncate
#pdf_fit_mode = "shrink"

# Section level that forces a break page.
# For example: 1 means top-level sections start in a new page
# 0 means disabled
#pdf_break_level = 0

# When a section starts in a new page, force it to be 'even', 'odd',
# or just use 'any'
#pdf_breakside = 'any'

# Insert footnotes where they are defined instead of
# at the end.
#pdf_inline_footnotes = True

# verbosity level. 0 1 or 2
#pdf_verbosity = 0

# If false, no index is generated.
#pdf_use_index = True

# If false, no modindex is generated.
#pdf_use_modindex = True

# If false, no coverpage is generated.
#pdf_use_coverpage = True

# Name of the cover page template to use
#pdf_cover_template = 'sphinxcover.tmpl'

# Documents to append as an appendix to all manuals.
#pdf_appendices = []

# Enable experimental feature to split table cells. Use it
# if you get "DelayedTable too big" errors
#pdf_splittables = False

# Set the default DPI for images
#pdf_default_dpi = 72

# Enable rst2pdf extension modules (default is only vectorpdf)
# you need vectorpdf if you want to use sphinx's graphviz support
#pdf_extensions = ['vectorpdf']

# Page template name for "regular" pages
#pdf_page_template = 'cutePage'

# Show Table Of Contents at the beginning?
#pdf_use_toc = True

# How many levels deep should the table of contents be?
pdf_toc_depth = 9999

# Add section number to section references
pdf_use_numbered_links = False

# Background images fitting mode
pdf_fit_background_mode = 'scale'


# -- Some tricks with plask for better documentation -------------------------------

vec = plask.vec(0., 0.)
doc = plask.vec.__doc__
plask.vec = type(vec)
plask.vec.__doc__ = doc
del vec

plask.Data = plask._plask._Data
plask.Data.__name__ = 'Data'

del doc

plask.config = type(plask.config)
