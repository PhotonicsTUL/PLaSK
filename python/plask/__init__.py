# -*- coding: utf-8 -*-
'''
PLaSK (Photonic Laser Simulation Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with different models: thermal, electrical, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consistent analysis of complete devices.

Modules
=======

.. autosummary::
   :toctree: plask
   :template: module.rst

   geometry
   mesh
   material
   flow
   phys
   algorithm
'''

copyright = "(c) 2013 Lodz University of Technology, Institute of Physics, Photonics Group"

import sys as _sys
import os as _os

_any = any # this buit-in is overriden by numpy

_os.environ["PLASK_PREFIX_PATH"] = _os.sep + _os.path.join(*__file__.split(_os.sep)[:-5])

from ._plask import *
from ._plask import _print_exception

try:
    from ._plask import __globals
except ImportError:
    pass

banner = '''\
PLaSK %s --- Photonic Laser Simulation Kit
%s
''' % (version, copyright)

## ## plask.material ## ##

material.db = material.MaterialsDB.get_default()

def update_factories():
    '''For each material in default database make factory in ``plask.material``.'''
    def factory(name):
        func = lambda **kwargs: material.db.get(name, **kwargs)
        func.__doc__ = "Create %s material." % name
        return func
    for mat in material.db:
        if mat == 'air': continue
        name = mat.split(":")[0]
        if name not in material.__dict__:
            material.__dict__[name.replace('-', '_')] = factory(name)
material.update_factories = update_factories
del update_factories

material.air = material.db.get("air")
material.Air = lambda: material.air

material.db.load_all()
material.update_factories()

def register_material(cls=None, name=None, is_complex=False, DB=None):
    '''Register a custom Python material.'''

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
        DB = material.db

    if is_complex:
        material._register_material_complex(cls.name, cls, DB)
    else:
        material._register_material_simple(cls.name, cls, DB)

    return cls

material.register_material = register_material
del register_material

material.simple = lambda mat, **kwargs: material.register_material(mat, is_complex=False, **kwargs)
material.simple.__doc__ = """Decorator for simple custom material class."""
material.complex = lambda mat, **kwargs: material.register_material(mat, is_complex=True, **kwargs)
material.complex.__doc__ = """Decorator for complex custom material class."""

material.const = staticmethod

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
    '''Load the XPL file.'''

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
    lst = [ g.axes for g in manager.geometry.values() if isinstance(g, geometry.Geometry) ]
    same = lst and lst.count(lst[0]) == len(lst)
    if same: current_axes = lst[0]

def runxpl(source, vars={}):
    '''Load and run the code from the XPL file.

       'source' is the name of the XPL file or open file object.
       'vars' is the optional dictionary with substitution variables
    '''
    env = globals().copy()
    env['plask'] = _sys.modules["plask"]
    loadxpl(source, vars, destination=env)
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
        _print_exception(ety, eva, etb, env['__manager__'].script_first_line, filename, True)


## ##  ## ##

def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to PLaSK logs
    """
    try: lineno += __globals['__manager__'].script_first_line
    except NameError: pass
    except KeyError: pass
    print_log(LOG_WARNING, "%s, line %s: %s: %s" % (filename, lineno, category.__name__, message))

import warnings
warnings.showwarning = _showwarning
del warnings


## ##  ## ##

class StepProfile(object):
    """
    Step profile for use in custom providers.

    Create a step profile class that can set a constant value of any scalar field
    in an arbitrary geometry object. Typical use of this class is setting an
    arbitrary heat source or step-profile material gain located in a chosen geometry
    object.

    Args:
        geometry: Geometry in which the step-profile is defined.
            It must be known in order to properly map the absolute mesh coordinates
            to the step-profile items.
        default: Default value of the provided field, returned in all non-referenced
            geometry objects.
        dtype: Type of the returned value. Defaults to `None`, in which case it is
            determined by the type of `default`.

    After creation, set the desired values at chosen geometry objects using item
    access [] notation:

    >>> profile[geometry_object] = value

    Then, you may retrieve the provider of a desired type using the normal outXXX
    name:

    >>> solver.inProperty = profile.outProperty

    This way you create a provider of the proper type and  associate it with the
    profile, so each time, the profile is in any way changed, all the receivers
    connected to the provider get notified.

    Example:
        To create a heat source profile that sets some heat at the object named
        `hot`:

        >>> hot = geometry.Rectangle(20,2, 'GaAs')
        >>> cold = geometry.Rectangle(20,10, 'GaAs')
        >>> stack = geometry.Stack2D()
        >>> stack.prepend(hot)
        <plask.geometry.PathHint at 0x47466b0>
        >>> stack.prepend(cold)
        <plask.geometry.PathHint at 0x47469e0>
        >>> geom = geometry.Cylindrical2D(stack)
        >>> profile = StepProfile(geom)
        >>> profile[hot] = 1e7
        >>> receiver = flow.HeatReceiverCyl()
        >>> receiver.connect(profile.outHeat)
        >>> list(receiver(mesh.Rectilinear2D([10], [5, 11])))
        [0.0, 10000000.0]
        >>> receiver.changed
        False
        >>> profile[hot] = 2e7
        >>> receiver.changed
        True
        >>> list(receiver(mesh.Rectilinear2D([10], [5, 11])))
        [0.0, 20000000.0]

    """

    def __init__(self, geometry, default=0., dtype=None):
        self.steps = {}
        self._geometry = geometry
        self._default = default
        self.dtype = dtype if dtype is not None else type(default)
        self.providers = {}

    def __getitem__(self, key):
        present = [step for step in self.steps if key == step]
        if present: return self.steps[present[0]]
        else: return self.steps[key]

    def __setitem__(self, key, val):
        # We use '== operator instead of 'is'
        present = [step for step in self.steps if key == step]
        if present: self.steps[present[0]] = val
        else: self.steps[key] = val
        for prov in self.providers.values(): prov.set_changed()

    def __delitem__(self, key):
        del self.steps[key]
        for prov in self.providers.values(): prov.set_changed()

    def __getattr__(self, name):
        if name[:3] != 'out':
            super(StepProfile, self).__getattr__(name)
        if (name in self.providers):
            return self.providers[name]
        suffix = { geometry.Cartesian2D: '2D',
                   geometry.Cylindrical2D: 'Cyl',
                   geometry.Cartesian3D: '3D' }[type(self._geometry)]
        provider = flow.__dict__[name[3:] + "Provider" + suffix](self)
        self.providers[name] = provider
        return provider

    def __call__(self, mesh, *args):
        result = ones(len(mesh), self.dtype) * self._default
        for xobj,val in self.steps.items():
            obj, pth = xobj if type(xobj) is tuple else (xobj, None)
            obj_iter = (self._geometry.object_contains(obj, pth, p) for p in mesh)
            result[fromiter(obj_iter, bool, len(mesh))] = val
        return result

    @property
    def default(self):
        '''Default value of the profile.

           This value is returned for all mesh points that are located outside any
           of the geometry objects with a specified value.
        '''
        return self._default

    @default.setter
    def default(self, val):
        self._default = val
        for prov in self.providers.values(): prov.set_changed()

    @property
    def geometry(self):
        '''Profile geometry. (read only)'''
        return self._geometry

    def clear_providers(self):
        '''Clear orphaned providers.

           Remove all associated providers that are not used elsewhere.
        '''
        keys = list(self.providers.keys())
        for key in keys:
            if _sys.getrefcount(self.providers[key]) == 2:
                del self.providers[key]

## ##  ## ##

try:
    from .pylab import *
except ImportError:
    from numpy import *
    print_log(LOG_WARNING, "plask.pylab could not be imported. You will not be able to make professionally-looking plots. Install matplotlib to resolve this issue.")
    has_pylab = False
else:
    has_pylab = True

try:
    from .hdf5 import *
except ImportError:
    print_log(LOG_WARNING, "plask.hdf5 could not be imported. Your will not be able to save fields to HDF5 files. Install h5py to resolve this issue.")
    has_hdf5 = False
else:
    has_hdf5 = True

## ##  ## ##

import plask.phys

import plask.algorithm

## ##  ## ##
