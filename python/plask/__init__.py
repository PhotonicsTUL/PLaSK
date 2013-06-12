# -*- coding: utf-8 -*-
'''
PLaSK (Photonic Laser Simulation Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with different models: thermal, electrical, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consistent analysis of complete devices.
'''

copyright = "(c) 2013 Lodz University of Technology, Institute of Physics, Photonics Group"


import sys as _sys
import os as _os

_any = any # this buit-in is overriden by numpy

_os.environ["PLASK_PREFIX_PATH"] = _os.sep + _os.path.join(*__file__.split(_os.sep)[:-5])

from ._plask import *
from ._plask import _print_exception, __globals

banner = '''\
PLaSK %s --- Photonic Laser Simulation Kit
%s
''' % (version, copyright)

_sys.path.insert(2, _os.path.join(lib_path, "solvers"))


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
            material.__dict__[name.replace('-', '_')] = factory(name)
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


## ## plask.geometry ## ##

def Stack2D(repeat=None, shift=0., **kwargs):
    '''Stack2D(repeat=None, shift=0)
           Create the stack, optionally repeating it 'repeat' times and with the bottom side
           of the first object at the 'shift' position (in container local coordinates).
           'kwargs' may contain default aligner specification.

           If 'repeat' is None, this function creates SingleStack2D and MultiStack2D otherwise.
    '''
    if repeat is None:
        return geometry.SingleStack2D(shift, **kwargs)
    else:
        return geometry.MultiStack2D(repeat, shift, **kwargs)
geometry.Stack2D = Stack2D
del Stack2D

def Stack3D(repeat=None, shift=0., **kwargs):
    '''Stack3D(repeat=None, shift=0)
           Create the stack, optionally repeating it 'repeat' times and with the bottom side
           of the first object at the 'shift' position (in container local coordinates).
           'kwargs' may contain default aligner specification.

           If 'repeat' is None, this function creates SingleStack3D and MultiStack3D otherwise.
    '''
    if repeat is None:
        return geometry.SingleStack3D(shift, **kwargs)
    else:
        return geometry.MultiStack3D(repeat, shift, **kwargs)
geometry.Stack3D = Stack3D
del Stack3D


## ## plask.manager ## ##

def loadxpl(source, vars={}, sections=None, destination=None):
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
    manager.load(source, vars, sections)
    manager.export(destination)
    material.update_factories() # There might have been some materials in the source file
    # Set default axes if all loaded geometries share the same
    lst = [ g.axes for g in manager.geometrics.values() if isinstance(g, geometry.Geometry) ]
    same = lst and lst.count(lst[0]) == len(lst)
    if same: config.axes = lst[0]

def runxpl(source, vars={}):
    '''Load and run the code from the XPL file.

       'source' is the name of the XPL file or open file object.
       'vars' is the optional dictionary with substitution variables
    '''
    env = globals().copy()
    env['plask'] = _sys.modules["plask"]
    loadxpl(source, vars, env)
    if type(source) == str:
        filename = source
    else:
        try: filename = source.name
        except: filename = "<source>"
    try:
        code = compile(env['__script__'], filename, 'exec')
        exec(code, env)
    except Exception as exc:
        ety, eva, etb = _sys.exc_info()
        _print_exception(ety, eva, etb, env['__manager__'].scriptline, filename, True)


## ##  ## ##

def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to PLaSK logs
    """
    try: lineno += __globals['__manager__'].scriptline
    except NameError: pass
    except KeyError: pass
    print_log(LOG_WARNING, "%s, line %s: %s: %s" % (filename, lineno, category.__name__, message))

import warnings
warnings.showwarning = _showwarning
del warnings


## ##  ## ##

class StepProfile(object):
    """
    Helper callable class for creating any step profile for use in custom providers
    """

    def __init__(self, geometry, dtype=float, default=0.):
        self.steps = {}
        self.geometry = geometry
        self.default = default
        self.dtype = dtype

    def __getitem__(self, key):
        return self.steps[key]

    def __setitem__(self, key, val):
        self.steps[key] = val

    def __delitem__(self, key):
        del self.steps[key]

    def __call__(self, mesh, *args):
        result = ones(len(mesh), self.dtype) * self.default
        for obj,val in self.steps.items():
            result[fromiter( (_any(p in box for box in self.geometry.get_object_bboxes(obj)) for p in mesh), bool, len(mesh) )] = val
        return result


## ##  ## ##

try:
    from plask.pylab import *
except ImportError:
    from numpy import *
    print_log(LOG_WARNING, "plask.pylab could not be imported. You will not be able to make professionally-looking plots. Install matplotlib to resolve this issue.")
    has_pylab = False
else:
    has_pylab = True

try:
    from plask.hdf5 import *
except ImportError:
    print_log(LOG_WARNING, "plask.hdf5 could not be imported. Your will not be able to save fields to HDF5 files. Install h5py to resolve this issue.")
    has_hdf5 = False
else:
    has_hdf5 = True


## ##  ## ##
