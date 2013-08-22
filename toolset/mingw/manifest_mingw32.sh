#!/bin/bash
python create_manifest.py
i686-pc-mingw32-windres --input msvcr_exe.rc --output msvcrc_exe.o
i686-pc-mingw32-windres --input msvcr_dll.rc --output msvcrc_dll.o

i686-pc-mingw32-gcc -dumpspecs | sed -e '/\*cpp:/ {n; s/^/-D__MSVCRT_VERSION__=0x0900 /}' -e 's/-lmoldname/-lmoldname90/' -e 's/-lmsvcrt/-lmsvcr90/' > specs
