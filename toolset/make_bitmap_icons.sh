cd gui/icons/hicolor/
for s in 24 32 48; do inkscape {s}x${s}/apps/plask.png scalable/apps/plask.svg -C -w$s -h$s; done
for s in 24 32 48; do inkscape -e {s}x${s}/apps/plask.png scalable/apps/plask.svg -C -w $s -h $s; done
for s in 24 32 48; do inkscape -e ${s}x${s}/apps/plask.png scalable/apps/plask.svg -C -w $s -h $s; done
for s in 16 24 32 48; do inkscape -e ${s}x${s}/mimetypes/application-x-plask.svg scalable/mimetypes/application-x-plask.svg -C -w $s -h $s; done
for s in 16 32 64 128 256; do inkscape -e ../../../utils/xpl-${s}.png scalable/mimetypes/application-x-plask.svg -C -w $s -h $s; done
for s in 16 32 64 128 256; do inkscape -e ../../../utils/plask-${s}.png scalable/apps/plask.svg -C -w $s -h $s; done
