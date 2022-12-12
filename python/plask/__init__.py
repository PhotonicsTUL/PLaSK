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

"""
PLaSK (Photonic Laser Simulation Kit) is a comprehensive toolkit for simulation
of various micro-scale photonic devices. It is particularly well suited for
analysis of semiconductor lasers, as it allows to perform simulations of various
physical phenomena with different models: thermal, electrical, quantum and optical.
PLaSK takes care of considering mutual interactions between these models and
allows to easily perform complex self-consistent analysis of complete devices.
"""

import sys
import os
import weakref as _weakref

# this buit-ins are overriden by numpy:
_any = any
_sum = sum

_basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_i = sys.path.index(_basepath) if _basepath in sys.path else 0
_basepath = os.path.dirname(_basepath)
if 'PLASK_PREFIX_PATH' not in os.environ:
    os.environ['PLASK_PREFIX_PATH'] = PREFIX_PATH = os.path.dirname(os.path.dirname(_basepath))
else:
    PREFIX_PATH = os.environ['PLASK_PREFIX_PATH']

if os.name == 'nt':
    _path = os.environ.get('PATH', "").split(';')
    _plask_bin_path = os.path.join(PREFIX_PATH, 'bin')
    if _plask_bin_path not in _path:
        _path.append(_plask_bin_path)
        os.environ['PATH'] = ';'.join(_path)


try:
    from . import _plask
    from ._plask import *
except ImportError:  # this seems necessary for Python 3.4
    import _plask
    from _plask import *

try: from ._plask import _print_stack # for debug only
except ImportError: pass


def print_exc():
    """Print last exception to PLaSK log."""
    _plask._print_exception(sys.exc_type, sys.exc_value, sys.exc_traceback)


if 'PLASK_SOLVERS_PATH' in os.environ:
    for _path in os.environ['PLASK_SOLVERS_PATH'].split(';' if os.name == 'nt' else ':'):
        _i += 1
        sys.path.insert(_i, _path)
else:
    sys.path.insert(_i+1, os.path.join(_basepath, 'solvers'))


## ## plask.material ## ##

try:
    from . import _material
except ImportError:
    from ._plask import _material

from . import material



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


class Cylindrical2D(geometry.Cylindrical):
    """obsolete"""
geometry.Cylindrical2D = Cylindrical2D
del Cylindrical2D


## ## plask.manager ## ##

def _geometry_changer(name):
    return lambda cls: Manager._register_geometry_changer(name, cls)
Manager._geometry_changer = _geometry_changer
del _geometry_changer

from .util import _geometry_copy_changers


def loadxpl(source, defs={}, sections=None, destination=None, update=False):
    """
    Load the XPL file. All sections contents is read into the `destination` scope.

    Args:
        source (str): Name of the XPL file or open file object.
        defs (dict): Optional dictionary with substitution variables. Values
                     specified in the <defines> section of the XPL file are
                     overridden with the one specified in this parameter.
        sections (list): List of section names to read.
        destination (dict): Destination scope. If None, ``globals()`` is used.
        update (bool): If the flag is ``False``, all data got from the previous
                       call to :func:`loadxpl` are discarded. Set it to ``True``
                       if you want to append some data from another file.
    """

    if destination is None:
        import __main__
        destination = __main__.__dict__
    if update:
        try:
            manager = destination['__manager__']
        except KeyError:
            destination['__manager__'] = manager = Manager()
    else:
        destination['__manager__'] = manager = Manager()
    manager.load(source, defs, sections)
    manager.export(destination)
    material.update_factories() # There might have been some materials in the source file
    # Set default axes if all loaded geometries share the same
    lst = [ g.axes for g in manager.geo.values() if isinstance(g, geometry.Geometry) ]
    same = lst and lst.count(lst[0]) == len(lst)
    if same: current_axes = lst[0]


def runxpl(source, defs={}):
    """
    Load and run the code from the XPL file. Unlike :func:`loadxpl` this function
    does not modify the current global scope.

    Args:
        source (str): Name of the XPL file or open file object.
        defs (dict): Optional dictionary with substitution variables. Values
                     specified in the <defines> section of the XPL file are
                     overridden with the one specified in this parameter.
    """
    env = globals().copy()
    env['plask'] = sys.modules["plask"]
    env.update(defs)
    loadxpl(source, defs, destination=env)
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
        _plask._print_exception(ety, eva, etb, filename, '<script>', env['__manager__']._scriptline)


## ##  ## ##

def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to PLaSK logs
    """
    print_log(LOG_WARNING, "{0}, line {1}: {2}: {3}".format(filename, lineno, category.__name__, str(message).replace('\n', ' ')))

import warnings
warnings.showwarning = _showwarning
del warnings

## ##  ## ##

class StepProfile:
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
        >>> geom = geometry.Cylindrical(stack)
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
            super().__getattribute__(name)
        if name in self.providers:
            return self.providers[name]
        suffix = { geometry.Cartesian2D: '2D',
                   geometry.Cylindrical: 'Cyl',
                   geometry.Cartesian3D: '3D' }[type(self._geometry)]
        try:
            provider = flow.__dict__[name[3:] + "Provider" + suffix](self)
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))
        self.providers[name] = provider
        return provider

    def __call__(self, mesh, *args):
        if not isinstance(mesh, _plask.mesh.Mesh) and len(args):
            mesh = args[0]
        result = ones(len(mesh), self.dtype) * self._default
        for xobj,val in self.steps.items():
            obj, pth = xobj if type(xobj) is tuple else (xobj, None)
            if pth is not None: pth = geometry.PathHints(pth)
            result[self._geometry.object_contains(obj, pth, mesh)] = val
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

class MaterialField:
    """
    Distribution of materials for a given geometry on a mesh.

    This class creates a ‘field’ of :py:class:`material.Material` objects and
    provides getters to easily obtain its properties as :py:class:`Data` object.

    Args:
        geometry: Geometry for which the materials a retrieved
        mesh: Mesh at which the parameters are retrieved

    Example:
        >>> material_field = MaterialField(your_geometry, your_mesh)
        >>> plot_field(material_field.thermk(300.), comp=0)
    """

    def __init__(self, geometry, mesh):
        self.materials = [geometry.get_material(point) for point in mesh]
        self.mesh = mesh

    def __getattr__(self, item):
        return lambda *args, **kwargs: \
            Data(numpy.array([getattr(m, item)(*args, **kwargs) for m in self.materials]), self.mesh)

    def __len__(self):
        return len(self.materials)

    def __iter__(self):
        return iter(self.materials)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            return self.materials[self.mesh.index(*index)]
        else:
            return self.materials[index]


def get_material_field(self, mesh):
    """
    Distribution of materials for a given geometry on a mesh.

    This class creates a ‘field’ of :py:class:`material.Material` objects and
    provides getters to easily obtain its properties as :py:class:`Data` object.

    Args:
        geometry: Geometry for which the materials a retrieved
        mesh: Mesh at which the parameters are retrieved

    Example:
        >>> material_field = this_geometry.get_material_field(your_mesh)
        >>> plot_field(material_field.thermk(300.), comp=0)
    """
    return MaterialField(self, mesh)

geometry.Geometry.get_material_field = get_material_field
del get_material_field

## ##  ## ##

import plask.phys
wl = phys.wl

## ##  ## ##

for JOBID in 'PLASK_JOBID', 'JOB_ID', 'SLURM_JOB_ID', 'SLURM_JOBID', 'PBS_JOBID', 'LSB_JOBID', 'LOAD_STEP_ID':
    if JOBID in os.environ:
        try: JOBID = int(os.environ[JOBID])
        except ValueError: JOBID = os.environ[JOBID]
        BATCH = True
        break
else:
    from time import time as _time
    JOBID = _time()
    BATCH = False
    del _time

for ARRAYID in 'PLASK_ARRAYID', 'PBS_ARRAYID', 'SLURM_ARRAY_TASK_ID', 'LSB_JOBINDEX', 'SGE_TASK_ID':
    if ARRAYID in os.environ:
        try: ARRAYID = int(os.environ[ARRAYID])
        except ValueError: ARRAYID = os.environ[ARRAYID]
        break
else:
    ARRAYID = None

for PROCID in 'PLASK_PROCID', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID', 'PBS_VNODENUM':
    if PROCID in os.environ:
        try: PROCID = int(os.environ[PROCID])
        except ValueError: PROCID = os.environ[PROCID]
        break
else:
    PROCID = None


## ##  ## ##
_plask.__xpl__globals.update(globals())


## ##  ## ##

# Default colors
MATERIAL_COLORS = {
    'Cu': '#9E807E',
    'Au': '#A6A674',
    'Pt': '#A6A674',
    'In': '#585266',

    'SiO2': '#FFD699',
    'Si': '#BF7300',
    'aSiO2': '#FFDF99',
    'aSi': '#BF8300',

    'AlOx': '#98F2FF',

    'GaAs': '#009e00',
    'AlAs': '#d1f000',
    'InAs': '#d18000',

    'GaN': '#00009e',
    'GaN_bulk': '#000080',
    'AlN': '#00d1f0',
    'InN': '#d10080',

    'GaP': '#915daf',
    'AlP': '#a089b5',
    'InP': '#6218a3',

    'GaSb': '#c17024',
    'AlSb': '#d19359',
    'InSb': '#b56113',

}
