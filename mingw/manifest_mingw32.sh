#!/bin/bash
python create_manifest.py
i686-pc-mingw32-windres --input msvcr.rc --output msvcrc.o
