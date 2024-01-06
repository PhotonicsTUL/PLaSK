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

guidir=$(realpath $(dirname ${0})/../gui)
utilsdir=$(realpath $(dirname ${0})/../utils)

CURRENT_YEAR=$(date '+%Y')

for s in 620 868 1116; do
    sed "s/{CURRENT_YEAR}/${CURRENT_YEAR}/" ${utilsdir}/splash.svg | rsvg-convert -o ${guidir}/splash${s}.png -w ${s}
done
