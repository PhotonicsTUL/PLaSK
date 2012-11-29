# -*- coding: utf-8 -*-
'''
PLaSK (Photonic Laser Simultion Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with different models: thermal, electrical, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consistent analysis of complete devices.
'''
from .plaskcore import *

## ## plask.material ## ##

materialdb = material.database = material.MaterialsDB.get_default()

def update_factories():
    '''For each material in default database make factory in plask.material'''
    def factory(name):
        return lambda **kwargs: materialdb.get(name, **kwargs)
    for mat in material.database:
        if mat == 'air': continue
        name = mat.split(":")[0]
        if name not in material.__dict__:
            material.__dict__[name] = factory(name)
material.update_factories = update_factories
del update_factories

material.air = materialdb.get("air")
material.Air = lambda: material.air

materialdb.load_all()
material.update_factories()

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

def load(source, destination=None):
    #TODO documentation
    if destination is None:
        try:
            destination = __globals
        except NameError:
            import __main__
            destination = __main__.__dict__
    try:
        manager = destination['__manager__']
    except KeyError:
        manager = Manager()
        destination['__manager__'] = manager
    manager.load(source)
    manager.export(destination)
    material.update_factories() # There might have been some materials in the source file
    # Set default axes if all loaded geometries share the same
    lst = [ manager.geometries[g].axes for g in manager.geometries ]
    same = lst and lst.count(lst[0]) == len(lst)
    if same: config.axes = lst[0]

def run(source):
    '''Load and run the code from the XPL file.

       'source' is the name of the XPL file or open file object.
    '''
    import sys
    env = globals().copy()
    env['plask'] = sys.modules["plask"]
    load(source, env)
    exec(env['__script__'], env)


## ##  ## ##

try:
    from plask.pylab import *
except ImportError:
    from numpy import *
    print_log(LOG_WARNING, "plask.pylab could not be imported. You won't be able to make professionally-looking plots. Install matplotlib to resolve this issue.")
    has_pylab = False
else:
    has_pylab = True

try:
    from plask.hdf5 import *
except ImportError:
    print_log(LOG_WARNING, "plask.hdf5 could not be imported. Your won't be able to save fields to HDF5 files. Install h5py to resolve this issue.")
    has_hdf5 = False
else:
    has_hdf5 = True


## ##  ## ##
#TODO dedeluxe
try:
    from plask.deluxe import *
except ImportError:
    pass
