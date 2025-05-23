# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

cmake_minimum_required(VERSION 3.14)
# cmake_policy(SET CMP0017 OLD)   # ensure our CMake modules are loaded first

project(plask-doc NONE)

get_directory_property(subproject PARENT_DIRECTORY)
if(NOT subproject)
    set(auto_doc ALL)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/_cmake)

find_package(Sphinx)
if(SPHINX_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sphinx" RESULT_VARIABLE sphinx_status ERROR_QUIET)
    if(NOT sphinx_status EQUAL 0)
        set(SPHINX_FOUND NOTFOUND)
    endif()
endif()

set(SPHINX_PAPER "a4" CACHE STRING "Paper format for Sphinx documentation")
set(SPHINX_OPTIONS "" CACHE STRING "Other Sphinx options")

# Internal variables.
set(PAPEROPT_a4 -D latex_paper_size=a4)
set(PAPEROPT_letter -D latex_paper_size=letter)
set(ALLSPHINXOPTS -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees ${PAPEROPT_${SPHINX_PAPER}paper} ${SPHINX_OPTIONS} ${CMAKE_CURRENT_SOURCE_DIR})
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

option(BUILD_QTHELP "Build and install QtHelp for GUI" OFF)
find_program(QHELPGENERATOR qhelpgenerator HINTS ${QT_PATH}/bin)

if(QHELPGENERATOR)
    execute_process(COMMAND ${QHELPGENERATOR} -v OUTPUT_VARIABLE qhelpgenerator_version_string)
    string(REGEX REPLACE "^Qt Help Generator version ([0-9]+(\\.[0-9]+)*) \\(Qt ([0-9]+(\\.[0-9]+)*)\\)" "\\3"
           qhelpgenerator_version "${qhelpgenerator_version_string}")
    if(qhelpgenerator_version VERSION_LESS "5.12.0")
        find_program(QCOLLECTIONGENERATOR qcollectiongenerator HINTS ${QT_PATH}/bin)
    else()
        set(QCOLLECTIONGENERATOR ${QHELPGENERATOR})
    endif()
endif()


set(sphinx ${CMAKE_BINARY_DIR}/bin/plask ${SPHINX_EXECUTABLE})

set(plask_depends plask ${PLASK_MATERIALS})

if(SPHINX_FOUND)
    file(GLOB rst_files *.rst api/*.rst xpl/*.rst xpl/geometry/*.rst xpl/geometry/2d/*.rst xpl/geometry/3d/*.rst xpl/solvers/filters.rst xpl/solvers/boundary_conditions.rst)

    set(solvers_api_timestamp "${CMAKE_CURRENT_BINARY_DIR}/solvers_api.timestamp")
    add_custom_command(OUTPUT ${solvers_api_timestamp}
        COMMAND ${CMAKE_COMMAND} -E touch ${solvers_api_timestamp}
        DEPENDS ${SOLVERS_PYTHON_TARGETS}
    )

    if(EXISTS ${CMAKE_SOURCE_DIR}/toolset/make_solvers_xpl_doc.py)
        file(GLOB_RECURSE solvers_yml ${CMAKE_SOURCE_DIR}/solvers/*/solvers.yml)
        set(solvers_xpl_timestamp ${CMAKE_CURRENT_BINARY_DIR}/solvers_xpl.timestamp)
        add_custom_command(OUTPUT ${solvers_xpl_timestamp}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/toolset/make_solvers_xpl_doc.py
            COMMAND ${CMAKE_COMMAND} -E touch ${solvers_xpl_timestamp}
            DEPENDS ${solvers_xml}
        )
    endif()

    add_custom_target(doc-html ${auto_doc}
        COMMAND ${sphinx} -b html ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/html
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py _themes/qthelp/theme.conf _themes/qthelp/static/qthelp.css
    )

    add_custom_target(doc-dirhtml
        COMMAND ${sphinx} -b dirhtml ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/dirhtml
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py _themes/qthelp/theme.conf _themes/qthelp/static/qthelp.css
    )

    add_custom_target(doc-singlehtml
        COMMAND ${sphinx} -b singlehtml ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/singlehtml
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py _themes/qthelp/theme.conf _themes/qthelp/static/qthelp.css
    )

    add_custom_target(doc-pickle
        COMMAND ${sphinx} -b pickle ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/pickle
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-json
        COMMAND ${sphinx} -b json ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/json
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-htmlhelp
        COMMAND ${sphinx} -b htmlhelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/htmlhelp
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py _themes/qthelp/theme.conf _themes/qthelp/static/qthelp.css
    )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhp ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhcp
        COMMAND ${sphinx} -b qthelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/qthelp
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py _themes/qthelp/theme.conf _themes/qthelp/static/qthelp.css
    )
    add_custom_target(doc-qthelp DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhp)

    if(QHELPGENERATOR AND QCOLLECTIONGENERATOR)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qch
            COMMAND ${QHELPGENERATOR} ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhp -o ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qch
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhp
        )
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhc
            COMMAND ${QCOLLECTIONGENERATOR} ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhcp -o ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhc
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhcp ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qch
        )
        if(BUILD_QTHELP)
            install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhc ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qch
                    DESTINATION share/doc/plask COMPONENT core)
            add_custom_target(qthelp ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qhc ${CMAKE_CURRENT_BINARY_DIR}/qthelp/plask.qch)
        endif()
    endif()

    add_custom_target(doc-devhelp
        COMMAND ${sphinx} -b devhelp ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/devhelp
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-epub
        COMMAND ${sphinx} -b epub ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/epub
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-latex
        COMMAND ${sphinx} -b latex ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/latex
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-latexpdf
        COMMAND ${sphinx} -b latex ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/latex
        COMMAND ${MAKE} -C ${CMAKE_CURRENT_BINARY_DIR}/latex all-pdf
        DEPENDS ${rst_files} ${pdf_images} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

#    add_custom_target(doc-pdf
#        COMMAND ${sphinx} -b pdf ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/pdf
#        DEPENDS ${rst_files} ${pdf_images} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
#    )

    add_custom_target(doc-text
        COMMAND ${sphinx} -b text ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/text
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-man
        COMMAND ${sphinx} -b man ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/man
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-texinfo
        COMMAND ${sphinx} -b texinfo ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/texinfo
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-info
        COMMAND ${sphinx} -b texinfo ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/texinfo
        COMMAND ${MAKE} -C ${CMAKE_CURRENT_BINARY_DIR}/texinfo info
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-gettext
        COMMAND ${sphinx} -b gettext ${I18NSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/locale
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-changes
        COMMAND ${sphinx} -b changes ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/changes
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-linkcheck
        COMMAND ${sphinx} -b linkcheck ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/linkcheck
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    add_custom_target(doc-doctest
        COMMAND ${sphinx} -b doctest ${ALLSPHINXOPTS} ${CMAKE_CURRENT_BINARY_DIR}/doctest
        DEPENDS ${rst_files} ${svg_files} ${plask_depends} ${solvers_xpl_timestamp} ${solvers_api_timestamp} conf.py
    )

    # Remove generated files
    add_custom_target(doc-clean
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/api/plask
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/api/solvers
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/xpl/solvers
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/solvers_xpl.timestamp
        COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/solvers_api.timestamp
    )
endif()

if(BUILD_QTHELP AND NOT (SPHINX_FOUND AND QHELPGENERATOR AND QCOLLECTIONGENERATOR))
    message(WARNING "BUILD_QTHELP option set but no Sphinx or Qt help tools found. Skipping...")
endif()
