#!/bin/sh
iconsdir=`dirname $0`/../gui/icons/hicolor
utilsdir=`dirname $0`/../utils

for s in 16 24 32 48; do
    inkscape -e ${iconsdir}/${s}x${s}/apps/plask.png ${iconsdir}/scalable/apps/plask.svg -C -w $s -h $s
    inkscape -e ${iconsdir}/${s}x${s}/mimetypes/application-x-plask.png \
        ${iconsdir}/scalable/mimetypes/application-x-plask.svg -C -w ${s} -h ${s}
done

for s in 16 32 64 128 256; do
    inkscape -e /tmp/plask-${s}.png ${iconsdir}/scalable/apps/plask.svg -C -w ${s} -h ${s}
    inkscape -e /tmp/xpl-${s}.png ${iconsdir}/scalable/mimetypes/application-x-plask.svg -C -w ${s} -h ${s}
done
convert /tmp/plask-16.png /tmp/plask-32.png /tmp/plask-64.png /tmp/plask-128.png /tmp/plask-256.png ${utilsdir}/plask.ico
convert /tmp/xpl-16.png /tmp/xpl-32.png /tmp/xpl-64.png /tmp/xpl-128.png /tmp/xpl-256.png ${utilsdir}/xpl.ico
rm /tmp/plask-16.png /tmp/plask-32.png /tmp/plask-64.png /tmp/plask-128.png /tmp/plask-256.png
rm /tmp/xpl-16.png /tmp/xpl-32.png /tmp/xpl-64.png /tmp/xpl-128.png /tmp/xpl-256.png
