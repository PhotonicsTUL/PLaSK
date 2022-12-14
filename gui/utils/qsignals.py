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



class BlockQtSignals:
    """
        Helper class to temporary block signals emitted by qt objects, usage::

            with BlockQtSignals(obj1, obj2, ...):
                # after with signals are reverted
    """

    def __init__(self, *objects):
        super().__init__()
        self.objects = objects
        self.signals_blocked_state = tuple(o.blockSignals(True) for o in self.objects)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            Revert objects blockSignals values.
        """
        for obj, val in zip(self.objects, self.signals_blocked_state):
            obj.blockSignals(val)
