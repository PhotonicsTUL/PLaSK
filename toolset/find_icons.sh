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

guidir=`dirname $0`/../gui

if [ "$1" = "" ]; then
    icons=$(grep -r fromTheme ${guidir}/* | perl -ne "print \$2.\"\n\" if /fromTheme\(('|\")(.*)\\1\)/" | sort | uniq)
else
    icons=$@
fi

for icon in ${icons}; do
    echo "Adding ${icon}"
    for size in 16 24 32; do
        iconfile=${guidir}/icons/hicolor/${size}x${size}/actions/${icon}.png
        if [ ! -f ${iconfile} ]; then
            cp /usr/share/icons/Tango/${size}x${size}/actions/${icon}.png ${iconfile} && \
            git add ${iconfile}
        fi
    done
    iconfile=${guidir}/icons/hicolor/scalable/actions/${icon}.svg
    if [ ! -f ${iconfile} ]; then
        cp /usr/share/icons/Tango/scalable/actions/${icon}.svg ${iconfile} && \
        git add ${iconfile}
    fi
done
