# -*- coding: utf-8 -*-
"""
PLaSK (Photonic Laser Simulation Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with different models: thermal, electrical, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consistent analysis of complete devices.
"""

import sys
import os as _os
import weakref as _weakref

_any = any # this buit-in is overriden by numpy

_os.environ["PLASK_PREFIX_PATH"] = _os.sep.join(__file__.split(_os.sep)[:-5])

import _plask
from ._plask import *

try: from ._plask import _print_stack # for debug only
except ImportError: pass

try:
    from ._plask import __globals
except ImportError:
    pass

def print_exc():
    """Print last exception to PLaSK log."""
    _plask._print_exception(sys.exc_type, sys.exc_value, sys.exc_traceback)

## ## plask.material ## ##

material.db = material.MaterialsDB.get_default()

material.get = lambda *args, **kwargs: material.db.get(*args, **kwargs)

def update_factories():
    """For each material in default database make factory in ``plask.material``."""
    def factory(name):
        def func(**kwargs):
            if 'dop' in kwargs:
                kwargs = kwargs.copy()
                dop = kwargs.pop('dop')
                return material.db.get(name+':'+dop, **kwargs)
            else:
                return material.db.get(name, **kwargs)
        func.__doc__ = u"Create material {}.\n\n:rtype: Material".format(name)
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

def load(lib):
    """
    Load material library from file.

    Args:
        lib (str): Library to load without the extension (.so or .dll).
    """
    material.db.load(lib)
    material.update_factories()
material.load = load
del load

class _simple(object):
    """
    Decorator for custom simple material class.

    Args:
        base (str or material.Material): Base class specification.
            It may be either a material object or a string with either
            complete or incomplete specification of the material.
            In either case you must initialize the base in the
            constructor of your material.
    """
    def __init__(self, base=None):
        if isinstance(base, type):
            raise TypeError("material.simple argument is a class (you probably forgot parenthes)")
        self.base = base
    def __call__(self, cls):
        if 'name' not in cls.__dict__: cls.name = cls.__name__
        material._register_material_simple(cls.name, cls, self.base)
        return cls
material.simple = _simple
del _simple

class _alloy(object):
    """
    Decorator for custom complex material (alloy) class.

    Args:
        base (str or material.Material): Base class specification.
            It may be either a material object or a string with either
            complete or incomplete specification of the material.
            In either case you must initialize the base in the
            constructor of your material.
    """
    def __init__(self, base=None):
        if isinstance(base, type):
            raise TypeError("material.alloy argument is a class (you probably forgot parenthes)")
        self.base = base
    def __call__(self, cls):
        if 'name' not in cls.__dict__: cls.name = cls.__name__
        material._register_material_complex(cls.name, cls, self.base)
        return cls
material.alloy = _alloy
material.complex = _alloy
del _alloy

material.const = staticmethod


## ## plask.geometry ## ##

def Stack2D(repeat=None, shift=0., **kwargs):
    """
    Create a 2D container that organizes its items in a vertical stack.

    The bottom side of the first object is located at the `shift` position in
    container local coordinates. Each consecutive object is placed on the top of
    the previous one.

    Args:
        repeat (int): Number of the stack contents repetitions. If None, this
                      function creates SingleStack2D and MultiStack2D otherwise.
        shift (float): Position in the local coordinates of the bottom of the stack.
        alignment (dict): Horizontal alignment specifications. This dictionary can
                          contain only one item. Its key can be ``left``, ``right``,
                          ``#center``, and ``#`` where `#` is the horizontal axis
                          name. The corresponding value is the position of the given
                          edge/center/origin of the item. This alignment can be
                          overriden while adding the objects to the stack.
                          By default the alignment is ``{'left': 0}``.

    See also:
        :class:`plask.geometry.SingleStack2D`, :class:`plask.geometry.MultiStack2D`.
    """
    if repeat is None:
        return geometry.SingleStack2D(shift, **kwargs)
    else:
        return geometry.MultiStack2D(repeat, shift, **kwargs)
geometry.Stack2D = Stack2D
del Stack2D

def Stack3D(repeat=None, shift=0., **kwargs):
    """
    Create a 3D container that organizes its items in a vertical stack.

    The bottom side of the first object is located at the `shift` position in
    container local coordinates. Each consecutive object is placed on the top of
    the previous one. Then the whole stack is repeated *repeat* times.

    Args:
        repeat (int): Number of the stack contents repetitions. If None, this
                      function creates SingleStack3D and MultiStack3D otherwise.
        shift (float): Position in the local coordinates of the bottom of the stack.
        alignments (dict): Horizontal alignments specifications. Keys in this dict
                           can be ``left``, ``right``, ``back``, ``front``,
                           ``#center``, and ``#`` where `#` are the horizontal axis
                           names. The corresponding value is the position of the
                           given edge/center/origin of the item. This alignment can
                           be overriden while adding the objects to the stack.
                           By default the alignment is ``{'left': 0, 'back': 0}``.

    See also:
        :class:`plask.geometry.SingleStack3D`, :class:`plask.geometry.MultiStack3D`.
    """
    if repeat is None:
        return geometry.SingleStack3D(shift, **kwargs)
    else:
        return geometry.MultiStack3D(repeat, shift, **kwargs)
geometry.Stack3D = Stack3D
del Stack3D

def Shelf(repeat=None, shift=0.):
    """
    Create a 2D shelf container that organizes its items one next to another
    (like books on a bookshelf).

    The left side of the first object is located at the `shift` position in
    container local coordinates. Each consecutive object is placed to the right of
    the previous one. All the items are vertically aligned according to its bottom
    edge.

    Args:
        repeat (int): Number of the shelf contents repetitions. If None, this
                      function creates Shelf2D and MultiShelf2D otherwise.
        shift (float): Position in the local coordinates of the left side of the
                       shelf.Classes

    See also:
        :class:`plask.geometry.Shelf2D`, :class:`plask.geometry.MultiShelf2D`.
    """
    if repeat is None:
        return geometry.Shelf2D(shift)
    else:
        return geometry.MultiShelf2D(repeat, shift)
geometry.Shelf = Shelf
del Shelf

geometry.AlignContainer2D = geometry.Align2D
geometry.AlignContainer3D = geometry.Align3D

## ## plask.manager ## ##

def loadxpl(source, vars={}, sections=None, destination=None, update=False):
    """
    Load the XPL file. All sections contents is read into the `destination` scope.

    Args:
        source (str): Name of the XPL file or open file object.
        vars (dict): Optional dictionary with substitution variables. Values
                     specified in the <defines> section of the XPL file are
                     overridden with the one specified in this parameter.
        sections (list): List of section names to read.
        destination (dict): Destination scope. If None, ``globals()`` is used.
        update (bool): If the flag is ``False``, all data got from the previous
                       call to :fun:`loadxpl` are discarded. Set it to ``True``
                       if you want to append some data from another file.
    """

    if destination is None:
        try:
            destination = __globals
        except NameError:
            import __main__
            destination = __main__.__dict__
    if update:
        try:
            manager = destination['__manager__']
        except KeyError:
            destination['__manager__'] = manager = Manager()
    else:
        destination['__manager__'] = manager = Manager()
    manager.load(source, vars, sections)
    manager.export(destination)
    material.update_factories() # There might have been some materials in the source file
    # Set default axes if all loaded geometries share the same
    lst = [ g.axes for g in manager.geometry.values() if isinstance(g, geometry.Geometry) ]
    same = lst and lst.count(lst[0]) == len(lst)
    if same: current_axes = lst[0]

def runxpl(source, vars={}):
    """
    Load and run the code from the XPL file. Unlike :fun:`loadxpl` this function
    does not modify the current global scope.

    Args:
        source (str): Name of the XPL file or open file object.
        vars (dict): Optional dictionary with substitution variables. Values
                     specified in the <defines> section of the XPL file are
                     overridden with the one specified in this parameter.
    """
    env = globals().copy()
    env['plask'] = sys.modules["plask"]
    env.update(vars)
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
        ety, eva, etb = sys.exc_info()
        _plask._print_exception(ety, eva, etb, env['__manager__'].script_first_line, filename, True)


## ##  ## ##

def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to PLaSK logs
    """
    if filename.endswith('.xpl'):
        try:
            lineno += __globals['__manager__'].script_first_line
        except NameError: pass
        except KeyError: pass
    print_log(LOG_WARNING, "{0}, line {1}: {2}: {3}".format(filename, lineno, category.__name__, str(message).replace('\n', ' ')))

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
        >>> list(receiver(mesh.Rectangular2D([10], [5, 11])))
        [0.0, 10000000.0]
        >>> receiver.changed
        False
        >>> profile[hot] = 2e7
        >>> receiver.changed
        True
        >>> list(receiver(mesh.Rectangular2D([10], [5, 11])))
        [0.0, 20000000.0]

    """

    def __init__(self, geometry, default=0., dtype=None):
        self.steps = {}
        self._geometry = geometry
        self._default = default
        self.dtype = dtype if dtype is not None else float if type(default) is int else type(default)
        self.providers = _weakref.WeakValueDictionary()

    def _fix_key(self, key):
        present = [step for step in self.steps if key == step]
        if present: return present[0]
        else: return key

    def __getitem__(self, key):
        return self.steps[self._fix_key(key)]

    def __setitem__(self, key, val):
        self.steps[self._fix_key(key)] = val
        for prov in self.providers.values(): prov.set_changed()

    def __delitem__(self, key):
        del self.steps[self._fix_key(key)]
        for prov in self.providers.values(): prov.set_changed()

    def __getattr__(self, name):
        if name[:3] != 'out':
            super(StepProfile, self).__getattr__(name)
        if name in self.providers:
            return self.providers[name]
        suffix = { geometry.Cartesian2D: '2D',
                   geometry.Cylindrical2D: 'Cyl',
                   geometry.Cartesian3D: '3D' }[type(self._geometry)]
        provider = flow.__dict__[name[3:] + "Provider" + suffix](self)
        self.providers[name] = provider
        return provider

    def __call__(self, mesh, *args):
        if not isinstance(mesh, _plask.mesh.Mesh) and len(args):
            mesh = args[0]
        result = ones(len(mesh), self.dtype) * self._default
        for xobj,val in self.steps.items():
            obj, pth = xobj if type(xobj) is tuple else (xobj, None)
            obj_iter = (self._geometry.object_contains(obj, pth, p) for p in mesh)
            result[fromiter(obj_iter, bool, len(mesh))] = val
        return result

    @property
    def default(self):
        """Default value of the profile.

           This value is returned for all mesh points that are located outside any
           of the geometry objects with a specified value.
        """
        return self._default

    @default.setter
    def default(self, val):
        self._default = val
        for prov in self.providers.values(): prov.set_changed()

    @property
    def geometry(self):
        """Profile geometry. (read only)"""
        return self._geometry

## ##  ## ##

def LAM(mat, lam, T=300.):
    """
    Compute optical wavelength in specified material.

    This is utility function that computes the physical lenght of a single
    wavelength in specified material. Its main purpose is easier design of
    DBR stack.

    If you are using it with custom materials, make sure that it does provide
    :meth:`~plask.material.Material.nr` method.

    Args:
        mat (material.Material or str): Material to compute physical wavelength in.
        lam (float): Free-space wavelength to scale for material `mat`.
        T (float): Temperature at which material refractive index is retrieved.
    """
    return 1e-3 * lam / material.get(mat).nr(lam, T)
## ##  ## ##

try:
    from .pylab import *
except ImportError:
    from numpy import *
    print_log(LOG_WARNING, "plask.pylab could not be imported."
                           " You will not be able to make professionally-looking plots."
                           " Install matplotlib to resolve this issue.")
    has_pylab = False
else:
    has_pylab = True

try:
    from .hdf5 import *
except ImportError:
    print_log(LOG_WARNING, "plask.hdf5 could not be imported."
                           " Your will not be able to save fields to HDF5 files."
                           " Install h5py to resolve this issue.")
    has_hdf5 = False
else:
    has_hdf5 = True

## ##  ## ##

import plask.phys

import plask.algorithm

## ##  ## ##

_plask.__xml__globals.update(globals())
