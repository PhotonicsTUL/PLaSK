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

import sys
import PIL.Image
import numpy as np
import os.path


CHUNK = 32


def make_c_image(imname, destdir=None):
    image = np.array(PIL.Image.open(imname))
    image[:,:,3] = 0
    image = image[:,:,(2,1,0,3)]
    height, width, _ = image.shape
    data = image.ravel()
    basename = os.path.splitext(os.path.basename(imname))[0]
    cname = (os.path.join(destdir, basename) if destdir is not None else basename) + ".h"
    with open(cname, 'w') as out:
        print(
            "static const struct {",
            "  unsigned int width;",
            "  unsigned int height;",
            "  const char* data;",
            "}} {} = {{".format(basename),
            "  {}, {},".format(width, height),
            sep="\n",
            file=out
        )
        for line in (data[i:i + CHUNK] for i in range(0, len(data), CHUNK)):
            print('  "{}"'.format("".join("\\x{:02x}".format(c) for c in line)), file=out)
        print("};", file=out)


make_c_image(*sys.argv[1:])
