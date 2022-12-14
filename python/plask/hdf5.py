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

'''
This is an interface to h5py. It contains useful functions for saving
and loading data to/from HDF5 files.


----------------------------------------------------------------------
'''

import numpy
import h5py
import plask

def save_rectangular1d(dest_group, name, mesh, compress=None):
    mesh_type = type(mesh)
    if mesh_type is plask.mesh.Regular:
        axis = dest_group.create_dataset(name, (1,), dtype=numpy.dtype([('start', float), ('stop', float), ('num', int)]))
        axis[0] = mesh.start, mesh.stop, len(mesh)
    else:
        if compress is None:
            axis = dest_group.create_dataset(name, data=numpy.array(mesh))
        else:
            mesh_array = numpy.array(mesh)
            axis = dest_group.create_dataset(name, data=mesh_array, chunks=mesh_array.shape,
                                             compression='gzip', compression_opts=compress)
    axis.attrs['type'] = mesh_type.__name__
    return axis

def load_rectangular1d(src_group, name):
    data = src_group[name]
    mtype = data.attrs['type']
    try:
        mtype = mtype.decode('ascii')
    except (UnicodeDecodeError, AttributeError):
        pass
    mesh_type = plask.mesh.__dict__[mtype]
    if isinstance(data, h5py.Group):
        data = data['points']
    if mesh_type is plask.mesh.Regular:
        return plask.mesh.Regular(data[0][0], data[0][1], int(data[0][2]))
    else:
        return plask.mesh.Ordered(data)


def save_field(field, file, path='', mode='a', compress=None):
    """
    Save field to HDF5 file.

    Args:
        file (str or file): File to save to.
            It should be either a filename or a h5py.File object opened for
            writing.
        field (plask.Data): Field to save.
           It should be an object returned by PLaSK provider that contains
           calculated field.
        path (str): HDF5 path (group and dataset name), under which the
           data is saved in the HDF5 file.
        mode (str): Mode used for opening new files.
        compress (None or int): Compression level. If None, data is not compressed.

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
    if mst is plask.mesh.Rectangular2D:
        axes = msh.axis1, msh.axis0
    elif mst is plask.mesh.Rectangular3D:
        axes = msh.axis2, msh.axis1, msh.axis0
    else:
        if mst in (plask.mesh.Rectilinear, plask.mesh.Regular):
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

    if compress is None:
        data = dest.create_dataset('_data', data=field.array)
    else:
        field_array = field.array
        data = dest.create_dataset('_data', data=field_array, chunks=field_array.shape,
                                   compression='gzip', compression_opts=compress)

    mesh_group = dest.create_group('_mesh')
    if mst in (plask.mesh.Ordered, plask.mesh.Regular):
        axis_dataset = save_rectangular1d(mesh_group, 'points', msh, compress)
        if type(msh) is plask.mesh.Ordered:
            try:
                data.dims[0].label = plask.current_axes[2]
                data.dims.create_scale(axis_dataset)
                data.dims[0].attach_scale(axis_dataset)
            except AttributeError:
                pass
    elif mst in (plask.mesh.Rectangular2D, plask.mesh.Rectangular3D):
        n = len(axes)
        mesh_group.attrs['type'] = mst.__name__
        #mesh_group.attrs['ordering'] = msh.ordering
        for i,ax in enumerate(axes):
            axis_dataset = save_rectangular1d(mesh_group, 'axis{:d}'.format(n-1-i), ax, compress)
            if type(ax) is plask.mesh.Ordered:
                try:
                    data.dims[i].label = plask.current_axes[3-n+i]
                    data.dims.create_scale(axis_dataset)
                    data.dims[i].attach_scale(axis_dataset)
                except AttributeError:
                    pass

    if close:
        file.close()


def load_field(file, path='/'):
    """
    Load field from HDF5 file.

    Args:
        file (str or file): File to load from.
            It should be either a filename or a h5py.File object opened for
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

    group = file[path]
    data = '_data'
    try:
        try:
            mesh = group['_mesh']
        except KeyError:
            mesh = group['mesh']
            data = 'data'
        mtype = mesh.attrs['type']
        try:
            mtype = mtype.decode('ascii')
        except (UnicodeDecodeError, AttributeError):
            pass
    except KeyError:
        raise TypeError('Group {} is not a PLaSK field'.format(path))
    if mtype in ('Rectilinear2D', 'Regular2D'): mtype = 'Rectangular2D'
    elif mtype in ('Rectilinear3D', 'Regular3D'): mtype = 'Rectangular3D'
    mst = plask.mesh.__dict__[mtype]

    if mst in (plask.mesh.Regular, plask.mesh.Ordered):
        msh = load_rectangular1d(mesh, 'points')
    elif mst in (plask.mesh.Rectangular2D, plask.mesh.Rectangular3D):
        msh = mst(*tuple(load_rectangular1d(mesh, axis) for axis in mesh))

    try:
        data = group[data]
    except KeyError:
        raise TypeError('Group {} is not a PLaSK field'.format(path))
    data = numpy.array(data)
    result = plask.Data(data, msh)

    if close:
        file.close()

    return result
