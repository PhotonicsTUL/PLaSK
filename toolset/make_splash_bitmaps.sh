#!/bin/sh
guidir=$(realpath $(dirname ${0})/../gui)
utilsdir=$(realpath $(dirname ${0})/../utils)

for s in 620 868 1116; do
    inkscape -e ${guidir}/splash${s}.png ${utilsdir}/splash.svg -C -w ${s}
done
