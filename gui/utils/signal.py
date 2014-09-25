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

""" A signal/slot implementation + __iadd__ and __isub__ (by Piotr Beling)

Author:  Thiago Marcos P. Santos
Author:  Christopher S. Case
Author:  David H. Bronke
Author:  Piotr Beling
Created: August 28, 2008
Updated: 2014
License: MIT

"""

import inspect
from weakref import WeakKeyDictionary

class Signal(object):
    def __init__(self):
        self._functions = set()
        self._methods = WeakKeyDictionary()

    def __call__(self, *args, **kwargs):
        # Call handler functions
        for func in self._functions:
            func(*args, **kwargs)

        # Call handler methods
        for obj, funcs in self._methods.items():
            for func in funcs:
                func(obj, *args, **kwargs)

    def connect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ not in self._methods:
                self._methods[slot.__self__] = set()

            self._methods[slot.__self__].add(slot.__func__)

        else:
            self._functions.add(slot)
            
    def __iadd__(self, slot):
        self.connect(slot)
        return self

    def disconnect(self, slot):
        if inspect.ismethod(slot):
            if slot.__self__ in self._methods:
                self._methods[slot.__self__].remove(slot.__func__)
        else:
            if slot in self._functions:
                self._functions.remove(slot)
                
    def __isub__(self, slot):
        self.disconnect(slot)
        return self

    def clear(self):
        self._functions.clear()
        self._methods.clear()
