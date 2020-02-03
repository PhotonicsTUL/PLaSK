# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import os

if os.name == 'nt':

    import ctypes

    INVALID_FILE_ATTRIBUTES = -1
    FILE_ATTRIBUTE_HIDDEN = 2

    def hide_file(filename):
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        attrs = kernel32.GetFileAttributesW(filename)
        if attrs == -1:
            raise ctypes.WinError(ctypes.get_last_error())
        attrs |= FILE_ATTRIBUTE_HIDDEN
        if not kernel32.SetFileAttributesW(filename, attrs):
            raise ctypes.WinError(ctypes.get_last_error())

else:

    def hide_file(filename):
        pass