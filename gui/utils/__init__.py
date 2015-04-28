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

from bisect import bisect_left

def sorted_index(sorted_list, x):
    'Locate the leftmost value exactly equal to x, raise ValueError if x is not in sorted_list.'
    i = bisect_left(sorted_list, x)
    if i != len(sorted_list) and sorted_list[i] == x:
        return i
    raise ValueError