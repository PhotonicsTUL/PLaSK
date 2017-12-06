cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0017 OLD)   # ensure our CMake modules are loaded first

project(plask-doc NONE)

get_directory_property(subproject PARENT_DIRECTORY)
if(NOT subproject)
    set(auto_doc ALL)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/_cmake)

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

file(GLOB_RECURSE svg_files *.svg)

# Make PDFs out from SVG
find_program(INKSCAPE inkscape)
if (INKSCAPE)
    foreach(svg ${svg_files})
        get_filename_component(base ${svg} NAME_WE)
        get_filename_component(directory ${svg} DIRECTORY)
        set(pdf ${directory}/${base}.pdf)
        add_custom_command(OUTPUT ${pdf}
            COMMAND ${INKSCAPE} -f ${svg} -A ${pdf}
            DEPENDS ${svg}
        )
        list(APPEND pdf_images ${pdf})
    endforeach()
endif()

set(sphinx plask ${SPHINX_EXECUTABLE})

if(SPHINX_FOUND)

    file(GLOB rst_files *.rst api/*.rst xpl/*.rst xpl/geometry/*.rst xpl/geometry/2d/*.rst xpl/geometry/3d/*.rst xpl/solvers/filters.rst xpl/solvers/boundary_conditions.rst)

    if(EXISTS ${CMAKE_SOURCE_DIR}/toolset/make_solvers_xpl_doc.py)
        file(GLOB_RECURSE solvers_xml ${CMAKE_SOURCE_DIR}/solvers/*/solvers.xml)
        set(solvers_timestamp ${CMAKE_CURRENT_BINARY_DIR}/solvers.timestamp)
        add_custom_command(OUTPUT ${solvers_timestamp}
            COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/toolset/make_solvers_xpl_doc.py
            COMMAND ${CMAKE_COMMAND} -E touch ${solvers_timestamp}
            DEPENDS ${solvers_xml}
        )
    endif()

    add_custom_target(doc-html ${auto_doc}
        COMMAND ${sphinx} -b html ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/html
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-dirhtml
        COMMAND ${sphinx} -b dirhtml ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/dirhtml
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-singlehtml
        COMMAND ${sphinx} -b singlehtml ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/singlehtml
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-pickle
        COMMAND ${sphinx} -b pickle ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/pickle
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-json
        COMMAND ${sphinx} -b json ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/json
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-htmlhelp
        COMMAND ${sphinx} -b htmlhelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/htmlhelp
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-qthelp
        COMMAND ${sphinx} -b qthelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/qthelp
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-devhelp
        COMMAND ${sphinx} -b devhelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/devhelp
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-epub
        COMMAND ${sphinx} -b epub ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/epub
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-latex
        COMMAND ${sphinx} -b latex ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/latex
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-latexpdf
        COMMAND ${sphinx} -b latex ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/latex
        COMMAND ${MAKE} -C ${CMAKE_CURRENT_BINARY_DIR}/latex all-pdf
        DEPENDS ${rst_files} ${pdf_images} ${solvers_timestamp} conf.py
    )

#    add_custom_target(doc-pdf
#        COMMAND ${sphinx} -b pdf ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/pdf
#        DEPENDS ${rst_files} ${pdf_images} ${solvers_timestamp} conf.py
#    )

    add_custom_target(doc-text
        COMMAND ${sphinx} -b text ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/text
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-man
        COMMAND ${sphinx} -b man ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/man
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-texinfo
        COMMAND ${sphinx} -b texinfo ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/texinfo
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-info
        COMMAND ${sphinx} -b texinfo ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/texinfo
        COMMAND ${MAKE} -C ${CMAKE_CURRENT_BINARY_DIR}/texinfo info
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-gettext
        COMMAND ${sphinx} -b gettext ${I18NSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/locale
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-changes
        COMMAND ${sphinx} -b changes ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/changes
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-linkcheck
        COMMAND ${sphinx} -b linkcheck ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/linkcheck
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    add_custom_target(doc-doctest
        COMMAND ${sphinx} -b doctest ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/doctest
        DEPENDS ${rst_files} ${svg_files} ${solvers_timestamp} conf.py
    )

    # Remove generated files
    add_custom_target(doc-clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/api/plask
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/api/solvers
    )

endif()

