#!/bin/sh
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

iconsdir=$(realpath $(dirname ${0})/../gui/icons)
utilsdir=$(realpath $(dirname ${0})/../utils)

if [ "${1}" = "" ]; then

    for s in 16 24 32 48; do
        inkscape -e ${iconsdir}/hicolor/${s}x${s}/apps/plask.png ${iconsdir}/hicolor/scalable/apps/plask.svg -C -w ${s} -h ${s}
        inkscape -e ${iconsdir}/hicolor/${s}x${s}/apps/plaskgui.png ${iconsdir}/hicolor/scalable/apps/plaskgui.svg -C -w ${s} -h ${s}
        inkscape -e ${iconsdir}/hicolor/${s}x${s}/mimetypes/application-x-plask.png \
            ${iconsdir}/hicolor/scalable/mimetypes/application-x-plask.svg -C -w ${s} -h ${s}
    done

    for s in 16 32 64 128 256; do
        inkscape -e /tmp/plask-${s}.png ${iconsdir}/hicolor/scalable/apps/plask.svg -C -w ${s} -h ${s}
        inkscape -e /tmp/xpl-${s}.png ${iconsdir}/hicolor/scalable/mimetypes/application-x-plask.svg -C -w ${s} -h ${s}
    done
    convert /tmp/plask-16.png /tmp/plask-32.png /tmp/plask-64.png /tmp/plask-128.png /tmp/plask-256.png ${utilsdir}/plask.ico
    convert /tmp/xpl-16.png /tmp/xpl-32.png /tmp/xpl-64.png /tmp/xpl-128.png /tmp/xpl-256.png ${utilsdir}/xpl.ico
    rm /tmp/plask-16.png /tmp/plask-32.png /tmp/plask-64.png /tmp/plask-128.png /tmp/plask-256.png
    rm /tmp/xpl-16.png /tmp/xpl-32.png /tmp/xpl-64.png /tmp/xpl-128.png /tmp/xpl-256.png

else

    for d in hicolor breeze; do
        for f in $@; do
            f="${f%.*}"
            for s in 16 24 32; do
                rsvg-convert -w ${s} -h ${s} -o ${iconsdir}/$d/${s}x${s}/${f}.png ${iconsdir}/$d/scalable/${f}.svg
                git add ${iconsdir}/$d/${s}x${s}/${f}.png
            done
        done
    done

fi
