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

import re


def empty_to_none(str):
    """ :param str str: sring
        :return: None if str is empty or consists only with white characters, str in other cases"""
    return None if len(str) == 0 or str.isspace() else str


_re_i = re.compile("<i>(.*?)</i>")
_re_sub = re.compile("<sub>(.*?)</sub>")
_re_sup = re.compile("<sup>(.*?)</sup>")

def html_to_tex(s):
    """Poor man's HTML to MathText conversion"""
    s = s.replace(" ", "\/")
    s = _re_i.sub(r"\mathit{\g<1>}", s)
    s = _re_sub.sub(r"_{\g<1>}", s)
    s = _re_sup.sub(r"^{\g<1>}", s)
    return r"$\sf " + s + "$"

