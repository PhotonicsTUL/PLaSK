# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import plask

from thermal.static import Static2D, StaticCyl, Static3D
from electrical.shockley import Shockley2D, ShockleyCyl, Shockley3D


# Get unique suffix for savefiles
if plask.BATCH:
    _suffix = "-{}".format(plask.JOBID)
else:
    import time as _time
    _suffix = _time.strftime("-%Y%m%d-%H%M", _time.localtime(plask.JOBID))


def _h5_open(filename, group):
    import h5py
    if filename is None:
        import sys
        filename = sys.argv[0]
        if filename.endswith('.py'): filename = filename[:-3]
        elif filename.endswith('.xpl'): filename = filename[:-4]
        elif filename == '': filename = 'console'
        filename += _suffix + '.h5'
    if type(filename) is h5py.File:
        h5file = filename
        filename = h5file.filename
    else:
        h5file = h5py.File(filename, 'a')
    orig_group = group; idx = 1
    while group in h5file:
        group = "{}-{}".format(orig_group, idx)
        idx += 1
    if group is not orig_group:
        plask.print_log(plask.LOG_WARNING,
                        "Group '{}' exists in HDF5 file '{}'. Saving to group '{}'" \
                            .format(orig_group, filename, group))
    return h5file, group


class ThermoElectric(plask.Solver):

    def __init__(self, name):
        super(ThermoElectric, self).__init__(name)
        self.thermal = self.Thermal(name)
        self.electrical = self.Electrical(name)
        self.electrical.inTemperature = self.thermal.outTemperature
        self.thermal.inHeat = self.electrical.outHeat

        self.outTemperature = self.thermal.outTemperature
        self.outHeatFlux = self.thermal.outHeatFlux
        self.outThermalConductivity = self.thermal.outThermalConductivity
        self.outVoltage = self.electrical.outVoltage
        self.outCurrentDensity = self.electrical.outCurrentDensity
        self.outHeat = self.electrical.outHeat
        self.outConductivity = self.electrical.outConductivity

        self.tfreq = 6

    def load_xml(self, xml, manager):
        for tag in xml:
            if tag == 'geometry':
                self.thermal.geometry = manager.geo[tag['thermal']]
                self.electrical.geometry = manager.geo[tag['electrical']]
            elif tag == 'mesh':
                self.thermal.mesh = manager.msh[tag['thermal']]
                self.electrical.mesh = manager.msh[tag['electrical']]
            elif tag == 'junction':
                if 'pnjcond' in tag: self.electrical.pnjcond = tag['pnjcond']
                for key, val in tag.attrs.items():
                    if key.startswith('beta') or key.startswith('js'):
                        setattr(self.electrical, key, val)
            elif tag == 'contacts':
                if 'pcond' in tag: self.electrical.pcond = tag['pcond']
                if 'ncond' in tag: self.electrical.ncond = tag['ncond']
            elif tag == 'loop':
                self.tfreq = tag.get('tfreq', self.tfreq)
                if 'maxterr' in tag: self.thermal.maxerr = tag['maxterr']
                if 'maxcerr' in tag: self.electrical.maxerr = tag['maxcerr']
            elif tag == 'tmatrix':
                if 'itererr' in tag: self.thermal.itererr = tag['itererr']
                if 'iterlim' in tag: self.thermal.iterlim = tag['iterlim']
                if 'logfreq' in tag: self.thermal.logfreq = tag['logfreq']
            elif tag == 'ematrix':
                if 'itererr' in tag: self.electrical.itererr = tag['itererr']
                if 'iterlim' in tag: self.electrical.iterlim = tag['iterlim']
                if 'logfreq' in tag: self.electrical.logfreq = tag['logfreq']
            elif tag == 'temperature':
                self.thermal.temperature_boundary.read_from_xml(tag, manager)
            elif tag == 'heatflux':
                self.thermal.heatflux_boundary.read_from_xml(tag, manager)
            elif tag == 'convection':
                self.thermal.convection_boundary.read_from_xml(tag, manager)
            elif tag == 'radiation':
                self.thermal.radiation_boundary.read_from_xml(tag, manager)
            elif tag == 'voltage':
                self.electrical.voltage_boundary.read_from_xml(tag, manager)

    def on_initialize(self):
        pass

    def on_invalidate(self):
        self.thermal.invalidate()
        self.electrical.invalidate()

    def compute(self, save=True, noinit=False):
        """
        Run calculations.

        In the beginning the solvers are invalidated and next, the thermo-
        electric algorithm is executed until both solvers converge to the
        value specified in their configuration in the `maxerr` property.

        Args:
            save (bool or str): If `True` the computed fields are saved to the
                HDF5 file named after the script name with the suffix denoting
                either the batch job id or the current time if no batch system
                is used. The filename can be overridden by setting this paramete
                as a string.
            noinit (bool): If this flas is set, solvers are not invalidated
                           in the beginning of the computations.
        """
        if not noinit:
            self.thermal.invalidate()
            self.electrical.invalidate()

        self.initialize()

        verr = 2. * self.electrical.maxerr
        terr = 2. * self.thermal.maxerr
        while terr > self.thermal.maxerr or verr > self.electrical.maxerr:
            verr = self.electrical.compute(self.tfreq)
            terr = self.thermal.compute(1)

        if save:
            self.save(None if save is True else save)

    def save(self, filename=None, group='ThermoElectric'):
        """
        Save the computation results to the HDF5 file.

        Args:
            filename (str): The file name to save to.
                If omitted, the file name is generated automatically based on
                the script name with suffix denoting either the batch job id or
                the current time if no batch system is used.

            group (str): HDF5 group to save the data under.
        """
        h5file, group = _h5_open(filename, group)
        tmesh = self.thermal.mesh
        vmesh = self.electrical.mesh
        jmesh = vmesh.get_midpoints()
        temp = self.thermal.outTemperature(tmesh)
        volt = self.electrical.outVoltage(vmesh)
        curr = self.electrical.outCurrentDensity(jmesh)
        plask.save_field(temp, h5file, group + '/Temperature')
        plask.save_field(volt, h5file, group + '/Potential')
        plask.save_field(curr, h5file, group + '/CurrentDensity')
        h5file.close()

    def plot_temperature(self, geometry_color='0.75', mesh_color=None, **kwargs):
        """
        Plot computed temperature to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification for
                the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            kwargs: Keyword arguments passed to the plot function.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        """
        field = self.thermal.outTemperature(self.thermal.mesh)
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Temperature [K]")
        if geometry_color is not None:
            plask.plot_geometry(self.thermal.geometry, color=geometry_color)
        if mesh_color is not None:
            plask.plot_mesh(self.thermal.mesh, color=mesh_color)
        plask.window_title("Temperature")

    def plot_voltage(self, geometry_color='0.75', mesh_color=None, **kwargs):
        """
        Plot computed voltage to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            kwargs: Keyword arguments passed to the :func:`plask.plot_field`.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        """
        field = self.electrical.outVoltage(self.electrical.mesh)
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Voltage [V]")
        if geometry_color is not None:
            plask.plot_geometry(self.electrical.geometry, color=geometry_color)
        if mesh_color is not None:
            plask.plot_mesh(self.electrical.mesh, color=mesh_color)
        plask.window_title("Voltage")

    def plot_vertical_voltage(self, at=0., **kwargs):
        """
        Plot computed voltage along the vertical axis

        Args:
            at (float): Horizontal position of the axis at which the voltage
                        is plotted.

            kwargs: Keyword arguments passed to the plot function.
        """
        if isinstance(self.electrical.geometry, plask.geometry.Cartesian3D):
            try:
                at0, at1 = at
            except TypeError:
                at0 = at1 = at
            mesh = plask.mesh.Rectangular2D(plask.mesh.Ordered([at0]), plask.mesh.Ordered([at1]),
                                            self.electrical.mesh.axis2)
        else:
            mesh = plask.mesh.Rectangular2D(plask.mesh.Ordered([at]), self.electrical.mesh.axis1)
        field = self.electrical.outVoltage(mesh)
        plask.plot(mesh.axis1, field, **kwargs)
        plask.xlabel(u"${}$ [\xb5m]".format(plask.config.axes[-1]))
        plask.ylabel("Voltage [V]")
        plask.window_title("Voltage")

    def plot_junction_current(self, refine=16, bounds=True, interpolation='linear', label=None, **kwargs):
        """
        Plot current density at the active region.

        Args:
            refine (int): Number of points in the plot between each two points
                          in the computational mesh.
            bounds (bool): If *True* then the geometry objects boundaries are
                           plotted.

            interpolation (str): Interpolation used when retrieving current density.

            label (str or sequence): Label for each junction. It can be a sequence
                                     consecutive labels for each junction, or a string
                                     in which case the same label is used for each
                                     junction. If omitted automatic label is generated.

            kwargs: Keyword arguments passed to the plot function.
        """
        # A little magic to get junction position first
        points = self.electrical.mesh.get_midpoints()
        geom = self.electrical.geometry.item
        yy = plask.unique(list(points.index1(i) for i,p in enumerate(points)
                          if geom.has_role('junction', p) or geom.has_role('active', p)))
        yy = [int(y) for y in yy]
        if len(yy) == 0:
            raise ValueError("no junction defined")
        act = []
        start = yy[0]
        axis1 = self.electrical.mesh.axis1
        for i, y in enumerate(yy):
            if y > yy[i-1] + 1:
                act.append(0.5 * (axis1[start] + axis1[yy[i-1] + 1]))
                start = y
        act.append(0.5 * (axis1[start] + axis1[yy[-1] + 1]))

        axis = plask.concatenate([
            plask.linspace(x, self.electrical.mesh.axis0[i+1], refine+1)
            for i,x in enumerate(list(self.electrical.mesh.axis0)[:-1])
        ])

        for i, y in enumerate(act):
            msh = plask.mesh.Rectangular2D(axis, plask.mesh.Ordered([y]))
            curr = self.electrical.outCurrentDensity(msh, interpolation).array[:,0,1]
            s = sum(curr)
            if label is None:
                lab = "Junction {:d}".format(i + 1)
            elif isinstance(label, tuple) or isinstance(label, tuple):
                lab = label[i]
            else:
                lab = label
            plask.plot(msh.axis0, curr if s > 0 else -curr,
                       label=lab, **kwargs)
        if len(act) > 1:
            plask.legend(loc='best')
        plask.xlabel(u"${}$ [\xb5m]".format(plask.config.axes[-2]))
        plask.ylabel(u"Current Density [kA/cm\xb2]")
        if bounds:
            simplemesh = plask.mesh.Rectangular2D.SimpleGenerator()\
                (self.electrical.geometry.item)
            for x in simplemesh.axis0:
                plask.axvline(x,
                              linestyle=plask.rc.grid.linestyle, linewidth=plask.rc.grid.linewidth,
                              color=plask.rc.grid.color, alpha=plask.rc.grid.alpha)
        plask.window_title("Current Density")


class ThermoElectric2D(ThermoElectric):
    Thermal = Static2D
    Electrical = Shockley2D


class ThermoElectricCyl(ThermoElectric):
    Thermal = StaticCyl
    Electrical = ShockleyCyl


class ThermoElectric3D(ThermoElectric):
    Thermal = Static3D
    Electrical = Shockley3D


