# -*- coding: utf-8 -*-
'''
This is an interface to h5py. It constains useful functions for saving
and loading data to/from HDF5 files.


----------------------------------------------------------------------
'''

import numpy
import h5py
import plask

def save_field(file, field, name='', mode='a'):
    '''Save field to HDF5 file. Call this function as:

           save_field(file, field, name='', mode='a')

       'file' is either a filename to save to or a h5py.File object open for writing.
       'field' is plask.Data object returned by providers/receivers.

       If 'file' is string, a new HDF5 file is opened with mode specified by 'mode'.
       Then both the data and their mesh are written to this file under a group specified
       by 'name' argument. It can contain slashes ('/'), in which case corresponding
       hierarchy is created in HDF5 file.

       Saved data can later be restored by any HDF5-aware application, or by load_field
       function of PLaSK.
    '''
    msh = field.mesh
    mst = type(msh)
    if mst in (plask.mesh.Rectilinear2D, plask.mesh.Regular2D):
        axes = msh.axis1, msh.axis0
    elif mst in (plask.mesh.Rectilinear3D, plask.mesh.Regular3D):
        axes = msh.axis2, msh.axis1, msh.axis0
    else:
        raise TypeError("unsupported mesh type for provided data")

    if type(file) == str:
        file = h5py.File(file, mode)
        close = True
    else:
        close = False

    if name:
        dest = file.create_group(name)
    else:
        dest = file

    data = dest.create_dataset('data', data=field.array)

    n = len(axes)
    mesh = dest.create_group('mesh')
    mesh.attrs['type'] = mst.__name__
    if mst in (plask.mesh.Rectilinear2D, plask.mesh.Rectilinear3D):
        for i,ax in enumerate(axes):
            axis = mesh.create_dataset('axis%d' % (n-1-i), data=numpy.array(ax))
            try:
                data.dims[i].label = plask.config.axes[3-n+i]
                data.dims.create_scale(axis)
                data.dims[i].attach_scale(axis)
            except AttributeError:
                pass
    elif mst in (plask.mesh.Regular2D, plask.mesh.Regular2D):
        dt = numpy.dtype([('start', float), ('stop', float), ('num', int)])
        for i,ax in enumerate(axes):
            axis = mesh.create_dataset('axis%d' % (n-1-i), (1,), dtype=dt)
            axis[0] = ax.start, ax.stop, len(ax)

    if close:
        file.close()


def load_field():
    pass
