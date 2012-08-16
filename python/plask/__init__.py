# -*- coding: utf-8 -*-
'''
PLaSK (Photonic Laser Simultion Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with defferent models: thermal, electrial, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consitent analysis of complete devices.
'''
from .plaskcore import *

from numpy import *

## ## plask.material ## ##

materialdb = material.database = material.MaterialsDB.getDefault()

def updateFactories():
    '''For each material in default database make factory in plask.material'''
    def factory(name):
        return lambda **kwargs: materialdb.get(name, **kwargs)
    for mat in material.database:
        if mat == 'air': continue
        name = mat.split(":")[0]
        if name not in material.__dict__:
            material.__dict__[name] = factory(name)
material.updateFactories = updateFactories
del updateFactories

#TODO do it in C++
def importLibrary(name):
    import os.path
    dirname = os.path.dirname(os.path.realpath(__file__))
    from ctypes import cdll
    for lib in [name+".so", name]:
        try:
            cdll.LoadLibrary(os.path.join(dirname, "..", "..", "materials", lib))
        except OSError:
            pass
        else:
            material.updateFactories()
            return
    raise OSError("Cannot import library '%s'" % name)
material.importLibrary = importLibrary
del importLibrary
material.importLibrary("nitrides")
#end of code to replace and mode to C++

material.air = materialdb.get("air")
material.Air = lambda: material.air

def register_material(cls=None, name=None, complex=False, DB=None):
    '''Register a custom Python material'''

    # A trick allowing passing arguments to decorator
    if cls is None:
        return lambda M: register_material(M, name=name, DB=DB)
    elif not issubclass(cls, material.Material):
        raise TypeError("Wrong decorated class (must be a subclass of plask.material.Material")

    if 'name' in cls.__dict__:
        if name is not None:
            raise ValueError("Name specified both in decorator parameter and class body")
    elif name is not None:
        cls.name = name
    else:
        cls.name = cls.__name__

    if DB is None:
        DB = material.database

    if complex:
        material._register_material_complex(cls.name, cls, DB)
    else:
        material._register_material_simple(cls.name, cls, DB)

    return cls

material.register_material = register_material
del register_material

material.simple = lambda mat, **kwargs: material.register_material(mat, complex=False, **kwargs)
material.complex = lambda mat, **kwargs: material.register_material(mat, complex=True, **kwargs)


## ## plask.manager ## ##

def read(source, destination=None):
    if destination is None:
        try:
            destination = _globals_
        except NameError:
            import __main__
            destination = __main__.__dict__
    manager = Manager()
    manager.read(source)
    manager.export(destination)

## ##  ## ##
