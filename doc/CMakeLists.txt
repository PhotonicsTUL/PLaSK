project(plask-doc)

find_package(Sphinx)

set(SPHINX_PAPER "a4" CACHE STRING "Paper format for Sphinx documentation")
set(SPHINX_OPTIONS "" CACHE STRING "Other Sphinx options")

# Internal variables.
set(PAPEROPT_a4 -D latex_paper_size=a4)
set(PAPEROPT_letter -D latex_paper_size=letter)
set(ALLSPHINXOPTS -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees ${PAPEROPT_${SPHINX_PAPER}} ${SPHINX_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR})
# the i18n builder cannot share the environment and doctrees with the others
set(I18NSPHINXOPTS ${PAPEROPT_${SPHINX_PAPER}} ${SPHINX_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR})

find_program(MAKE make)

# Make PDFs out from SVG
find_program(INKSCAPE inkscape)
if (INKSCAPE)
    foreach(svg ${svg_files})
        get_filename_component(base ${svg} NAME_WE)
        set(pdf ${CMAKE_CURRENT_SOURCE_DIR}/${base}.pdf)
        add_custom_command(OUTPUT ${pdf}
            COMMAND ${INKSCAPE} -f ${svg} -A ${pdf}
            DEPENDS ${svg}
        )
        list(APPEND pdf_images ${pdf})
    endforeach()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plaskconf.py.in ${CMAKE_CURRENT_BINARY_DIR}/plaskconf.py)

#TODO get rid of this is stubs are created for us automatically
set(sphinx export CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR} COMMAND plask ${SPHINX_EXECUTABLE})

if(SPHINX_FOUND)

    file(GLOB_RECURSE rst_files *.rst)
    file(GLOB_RECURSE svg_files *.svg)

    add_custom_target(sphinx-html
        COMMAND ${sphinx} -b html ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/html
        COMMAND echo "Build finished. The HTML pages are in ${CMAKE_CURRENT_BINARY_DIR}/html." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-dirhtml
        COMMAND ${sphinx} -b dirhtml ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/dirhtml
        COMMAND echo "Build finished. The HTML pages are in ${CMAKE_CURRENT_BINARY_DIR}/dirhtml." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-singlehtml
        COMMAND ${sphinx} -b singlehtml ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/singlehtml
        COMMAND echo "Build finished. The HTML page is in ${CMAKE_CURRENT_BINARY_DIR}/singlehtml." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-pickle
        COMMAND ${sphinx} -b pickle ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/pickle
        COMMAND echo "Build finished; now you can process the pickle files." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-json
        COMMAND ${sphinx} -b json ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/json
        COMMAND echo "Build finished; now you can process the JSON files." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-htmlhelp
        COMMAND ${sphinx} -b htmlhelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/htmlhelp
        COMMAND echo "Build finished; now you can run HTML Help Workshop with the"
                     ".hhp project file in ${CMAKE_CURRENT_BINARY_DIR}/htmlhelp." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-qthelp
        COMMAND ${sphinx} -b qthelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/qthelp
        COMMAND echo "Build finished; now you can run \\\"qcollectiongenerator\\\" with the"
                     ".qhcp project file in ${CMAKE_CURRENT_BINARY_DIR}/qthelp, like this:" VERBATIM
        COMMAND echo "# qcollectiongenerator ${CMAKE_CURRENT_BINARY_DIR}/qthelp/PLaSK.qhcp" VERBATIM
        COMMAND echo "To view the help file:" VERBATIM
        COMMAND echo "# assistant -collectionFile ${CMAKE_CURRENT_BINARY_DIR}/qthelp/PLaSK.qhc" VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-devhelp
        COMMAND ${sphinx} -b devhelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/devhelp
        COMMAND echo "Build finished." VERBATIM
        COMMAND echo "To view the help file:" VERBATIM
        COMMAND echo "# mkdir -p $$HOME/.local/share/devhelp/PLaSK" VERBATIM
        COMMAND echo "# ln -s ${CMAKE_CURRENT_BINARY_DIR}/devhelp $$HOME/.local/share/devhelp/PLaSK" VERBATIM
        COMMAND echo "# devhelp" VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-epub
        COMMAND ${sphinx} -b epub ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/epub
        COMMAND echo "Build finished. The epub file is in ${CMAKE_CURRENT_BINARY_DIR}/epub." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-latex
        COMMAND ${sphinx} -b latex ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/latex
        COMMAND echo "Build finished; the LaTeX files are in ${CMAKE_CURRENT_BINARY_DIR}/latex." VERBATIM
        COMMAND echo "Run 'make' in that directory to run these through (pdf)latex"
                     "(use 'make latexpdf' here to do that automatically)." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-latexpdf
        COMMAND ${sphinx} -b latex ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/latex
        COMMAND echo "Running LaTeX files through pdflatex..." VERBATIM
        COMMAND ${MAKE} -C ${CMAKE_CURRENT_BINARY_DIR}/latex all-pdf
        COMMAND echo "pdflatex finished; the PDF files are in ${CMAKE_CURRENT_BINARY_DIR}/latex." VERBATIM
        DEPENDS ${rst_files} ${pdf_images} conf.py
    )

    add_custom_target(sphinx-text
        COMMAND ${sphinx} -b text ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/text
        COMMAND echo "Build finished. The text files are in ${CMAKE_CURRENT_BINARY_DIR}/text." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-man
        COMMAND ${sphinx} -b man ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/man
        COMMAND echo "Build finished. The manual pages are in ${CMAKE_CURRENT_BINARY_DIR}/man." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-texinfo
        COMMAND ${sphinx} -b texinfo ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/texinfo
        COMMAND echo "Build finished. The Texinfo files are in ${CMAKE_CURRENT_BINARY_DIR}/texinfo." VERBATIM
        COMMAND echo "Run 'make' in that directory to run these through makeinfo"
                     "(use 'make info' here to do that automatically)." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-info
        COMMAND ${sphinx} -b texinfo ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/texinfo
        COMMAND echo "Running Texinfo files through makeinfo..." VERBATIM
        COMMAND ${MAKE} -C ${CMAKE_CURRENT_BINARY_DIR}/texinfo info
        COMMAND echo "makeinfo finished; the Info files are in ${CMAKE_CURRENT_BINARY_DIR}/texinfo." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-gettext
        COMMAND ${sphinx} -b gettext ${I18NSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/locale
        COMMAND echo "Build finished. The message catalogs are in ${CMAKE_CURRENT_BINARY_DIR}/locale." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-changes
        COMMAND ${sphinx} -b changes ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/changes
        COMMAND echo "The overview file is in ${CMAKE_CURRENT_BINARY_DIR}/changes." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-linkcheck
        COMMAND ${sphinx} -b linkcheck ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/linkcheck
        COMMAND echo "Link check complete; look for any errors in the above output "
                     "or in ${CMAKE_CURRENT_BINARY_DIR}/linkcheck/output.txt." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

    add_custom_target(sphinx-doctest
        COMMAND ${sphinx} -b doctest ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/doctest
        COMMAND echo "Testing of doctests in the sources finished, look at the "
                     "results in ${CMAKE_CURRENT_BINARY_DIR}/doctest/output.txt." VERBATIM
        DEPENDS ${rst_files} ${svg_files} conf.py
    )

endif()

