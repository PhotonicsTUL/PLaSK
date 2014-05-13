# -*- coding: utf-8 -*-
'''
This is an interface to h5py. It constains useful functions for saving
and loading data to/from HDF5 files.


----------------------------------------------------------------------
'''

import numpy
import h5py
import plask

def save_field(field, file, path='', mode='a'):
    """
    Save field to HDF5 file.

    Args:
        file (str or file): File to save to.
            It should be eiher a filename or a h5py.File object opened for
            writing.
        field (plask.Data): Field to save.
           It should be an object returned by PLaSK provider that contains
           calculated field.
        path (str): HDF5 path (group and dataset name), under which the
           data is saved in the HDF5 file.
        mode (str): Mode used for opening new files.

    If ``file`` is a string, a new HDF5 file is opened with the mode
    specified by ``mode``. Then both the data and its mesh are written to
    this file under the path specified by the ``path`` argument. It can
    contain slashes ('/'), in which case a corresponding hierarchy is created
    in the HDF5 file.

    Saved data can later be restored by any HDF5-aware application, or by
    ``load_field`` function of PLaSK.

    Example:
       You may save data retrieved from a provider to file as follows:

       >>> data = my_solver.outMyData(my_mesh)
       >>> save_field('myfile.h5', data, 'mygroup/mydata', 'a')

       In another PLaSK session, you may retrieve this data and plot it
       or provide to some receiver:

       >>> data = load_field('myfile.h5', 'mygroup/mydata')
       >>> plot_field(data)
       >>> other_solver.inMyData = data
    """
    msh = field.mesh
    mst = type(msh)
    if mst in (plask.mesh.Rectilinear1D, plask.mesh.Regular1D):
        axes = (msh,)
    if mst in (plask.mesh.Rectangular2D):
        axes = msh.axis1, msh.axis0
    elif mst in (plask.mesh.Rectangular3D):
        axes = msh.axis2, msh.axis1, msh.axis0
    else:
        raise TypeError("unsupported mesh type for provided data")

    if type(file) == str:
        file = h5py.File(file, mode)
        close = True
    else:
        close = False

    if path:
        dest = file.create_group(path)
    else:
        dest = file

    data = dest.create_dataset('data', data=field.array)

    n = len(axes)
    mesh = dest.create_group('mesh')
    mesh.attrs['type'] = mst.__name__
    mesh.attrs['ordering'] = msh.ordering
    if mst in (plask.mesh.Rectilinear1D, plask.mesh.Rectilinear2D, plask.mesh.Rectilinear3D):
        for i,ax in enumerate(axes):
            axis = mesh.create_dataset('axis{:d}'.format(n-1-i), data=numpy.array(ax))
            try:
                data.dims[i].label = plask.current_axes[3-n+i]
                data.dims.create_scale(axis)
                data.dims[i].attach_scale(axis)
            except AttributeError:
                pass
    elif mst in (plask.mesh.Regular1D, plask.mesh.Regular2D, plask.mesh.Regular2D):
        dt = numpy.dtype([('start', float), ('stop', float), ('num', int)])
        for i,ax in enumerate(axes):
            axis = mesh.create_dataset('axis{:d}'.format(n-1-i), (1,), dtype=dt)
            axis[0] = ax.start, ax.stop, len(ax)

    if close:
        file.close()


def load_field(file, path=''):
    """
    Load field from HDF5 file.

    Args:
        file (str or file): File to load from.
            It should be eiher a filename or a h5py.File object opened for
            reading.
        path (str): HDF5 path (group and dataset name), under which the
           data is located in the HDF5 file.
    Returns:
        Read plask.Data object.

    If ``file`` is a string, a new HDF5 file is opened for reading. Then both
    the data and its mesh are read from this file from the path specified by
    the ``path`` argument. ``path`` can contain slashes ('/'), in which case
    the corresponding hierarchy in the HDF5 file is used.

    Example:
       You may save data retrieved from a provider to file as follows:

       >>> data = my_solver.outMyData(my_mesh)
       >>> save_field('myfile.h5', data, 'mygroup/mydata', 'a')

       In another PLaSK session, you may retrieve this data and plot it
       or provide to some receiver:

       >>> data = load_field('myfile.h5', 'mygroup/mydata')
       >>> plot_field(data)
       >>> other_solver.inMyData = data
    """
    if type(file) == str:
        file = h5py.File(file, 'r')
        close = True
    else:
        close = False

    mesh = file[path+'/mesh']
    mst = plask.mesh.__dict__[mesh.attrs['type']]
    if mst in (plask.mesh.Regular2D, plask.mesh.Regular3D):
        kwargs = dict([ (k, (v[0][0],v[0][1],int(v[0][2]))) for k,v in mesh.items() ])
    else:
        kwargs = dict(mesh.items())
    msh = mst(**kwargs)

    data = file[path+'/data']
    data = numpy.array(data)
    result = plask.Data(numpy.array(data), msh)

    if close:
        file.close()

    return result
