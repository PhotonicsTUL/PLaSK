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

# coding: utf8

import os

if os.name == 'nt':

    import ctypes

    FILE_ATTRIBUTE_READONLY = 0x1
    FILE_ATTRIBUTE_HIDDEN = 0x2
    FILE_ATTRIBUTE_SYSTEM = 0x4
    FILE_ATTRIBUTE_ARCHIVE = 0x20
    FILE_ATTRIBUTE_NORMAL = 0x80
    FILE_ATTRIBUTE_TEMPORARY = 0x100
    FILE_ATTRIBUTE_OFFLINE = 0x1000
    FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 0x2000

    def set_file_attributes(filename, attrs):
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        if not kernel32.SetFileAttributesW(filename, attrs):
            raise ctypes.WinError(ctypes.get_last_error())
