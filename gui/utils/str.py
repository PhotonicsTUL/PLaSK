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


import re


def empty_to_none(str):
    """ :param str str: string
        :return: None if str is empty or consists only with white characters, str in other cases"""
    return None if str is None or len(str) == 0 or str.isspace() else str


def none_to_empty(str):
    return '' if str is None else str

_re_br = re.compile("<br */>\s*")
_re_i = re.compile("<i>(.*?)</i>")
_re_sub = re.compile("<sub>\$?(.*?)\$?</sub>")
_re_sup = re.compile("<sup>\$?(.*?)\$?</sup>")


def html_to_tex(s):
    """Poor man's HTML to MathText conversion"""
    # s = s.replace(" ", "\/")ipython

    s = _re_br.sub("\n", s)
    s = _re_i.sub(r"$\g<1>$", s)
    s = _re_sub.sub(r"$_{\g<1>}$", s)
    s = _re_sup.sub(r"$^{\g<1>}$", s)
    s = s.replace("$$", "")
    return s
