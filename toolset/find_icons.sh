#!/bin/sh
guidir=`dirname $0`/../gui
icons=$(grep -r fromTheme ${guidir}/* | perl -ne "print \$2.\"\n\" if /fromTheme\(('|\")(.*)\\1\)/" | sort | uniq)

for icon in ${icons}; do
    for size in 16 24 32 64; do
        iconfile=${guidir}/icons/hicolor/${size}x${size}/${icon}.png
        if [ ! -f ${iconfile} ]; then
            cp /usr/share/icons/Tango/${size}x${size}/*/${icon}.png ${iconfile} && \
            svn add ${iconfile}
        fi
    done
done