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
from copy import copy

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)
else:
    # 'unicode' exists, must be Python 2
    bytes = str


try:
    import plask
except ImportError:
    plask = None


def sorted_index(sorted_list, x):
    """
        Locate the leftmost value exactly equal to x, raise ValueError if x is not in sorted_list.
        :param list sorted_list: sorted list
        :param x: object to find in sorted_list
        :return int: index of x in sorted_list
        :except ValueError: if x is not included in sorted_list
    """
    i = bisect_left(sorted_list, x)
    if i != len(sorted_list) and sorted_list[i] == x:
        return i
    raise ValueError


class _GETATTR_BY_PATH_NO_DEFAULT(object):
    pass


def getattr_by_path(obj, name, default=_GETATTR_BY_PATH_NO_DEFAULT):
    """
    Extended version of standard getattr, which support strings, ints and iterable over strings and ints as name.
    iterable leads to sequence of getting attributes, ints are interpreted as indexes for [] operator
    :param obj: object whose attribute should be get or first object of path
    :param name: string or int or iterable over strings and ints
    :param default: optional default value, if given exceptions are not raised but this value is returned instead
    :return: value of object's attribute
    :except AttributeError: when object hasn't got attribute with given name
    :except IndexError: when int is given and [] operator raise IndexError
    :except TypeError: when int is given and object hasn't got [] operator
    """
    if isinstance(name, basestring):
        if default == _GETATTR_BY_PATH_NO_DEFAULT:
            return getattr(obj, name)
        else:
            return getattr(obj, name, default)
    try:
        if isinstance(name, int): return obj[name]
        for a in name:
            if isinstance(a, int):
                obj = obj[a]
            else:
                obj = getattr_by_path(obj, a)
    except (AttributeError, IndexError, TypeError):
        if default != _GETATTR_BY_PATH_NO_DEFAULT:
            return default
        else:
            raise
    return obj


def setattr_by_path(obj, name, value):
    """
    Extended version of standard setattr, which support strings, ints and iterable over strings and ints as name.
    iterable leads to sequence of getting attributes, ints are interpreted as indexes for [] operator
    :param obj: object whose attribute should be set or first object of path
    :param name: string or int or iterable over strings and ints
    :param value: new value for attribute
    :except AttributeError: when path can't be resolved because object hasn't got attribute with given name
    :except IndexError: when int is given and [] operator raise IndexError
    :except TypeError: when int is given and object hasn't got [] operator
    """
    if isinstance(name, basestring):
        setattr(obj, name, value)
    elif isinstance(name, int):
        obj[name] = value
    else:
        setattr_by_path(getattr_by_path(obj, name[:-1]), name[-1], value)


def require_str_first_attr_path_component(path):
    while not isinstance(path, basestring): path = path[0]
    return path


def get_manager():
    if plask is None: return
    materials = copy(plask.material.db)
    manager = plask.Manager(materials, draft=True)
    if 'wl' not in manager.defs:
        def wl(mat, lam, T=300.):
            try: nr = materials.get(mat).Nr(lam, T).real
            except: nr = 1.
            return 1e-3 * lam / nr
        plask._plask.__xml__globals['wl'] = wl
        plask._plask.__xml__globals['phys'].__dict__['wl'] = wl
    return manager