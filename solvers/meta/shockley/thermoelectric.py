#coding: utf8
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

import thermal.static
import electrical.shockley


class attribute(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)


# Get unique suffix for savefiles
if plask.BATCH:
    suffix = "-{}".format(plask.JOBID)
else:
    import time as _time
    suffix = _time.strftime("-%Y%m%d-%H%M", _time.localtime(plask.JOBID))


def h5open(filename, group):
    import h5py
    if filename is None:
        import sys
        filename = sys.argv[0]
        if filename.endswith('.py'): filename = filename[:-3]
        elif filename.endswith('.xpl'): filename = filename[:-4]
        elif filename == '': filename = 'console'
        filename += suffix + '.h5'
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

        self.tfreq = 6

    def _read_attr(self, tag, attr, solver, pyattr=None):
        if pyattr is None: pyattr = attr
        if attr in tag:
            setattr(solver, pyattr, tag[attr])

    def load_xpl(self, xpl, manager):
        for tag in xpl:
            self._parse_xpl(tag, manager)

    def _parse_xpl(self, tag, manager):
        if tag == 'geometry':
            self.thermal.geometry = manager.geo[tag.get_str('thermal')]
            self.electrical.geometry = manager.geo[tag.get_str('electrical')]
        elif tag == 'mesh':
            self.thermal.mesh = manager.msh[tag.get_str('thermal')]
            self.electrical.mesh = manager.msh[tag.get_str('electrical')]
        elif tag == 'junction':
            self._read_attr(tag, 'pnjcond', self.electrical)
            for key, val in tag.attrs.items():
                if key.startswith('beta') or key.startswith('js'):
                    setattr(self.electrical, key, val)
        elif tag == 'contacts':
            self._read_attr(tag, 'pcond', self.electrical)
            self._read_attr(tag, 'ncond', self.electrical)
        elif tag == 'loop':
            self.tfreq = tag.get('tfreq', self.tfreq)
            self._read_attr(tag, 'inittemp', self.thermal, 'inittemp')
            self._read_attr(tag, 'maxterr', self.thermal, 'maxerr')
            self._read_attr(tag, 'maxcerr', self.electrical, 'maxerr')
        elif tag == 'tmatrix':
            self._read_attr(tag, 'itererr', self.thermal)
            self._read_attr(tag, 'iterlim', self.thermal)
            self._read_attr(tag, 'logfreq', self.thermal)
        elif tag == 'ematrix':
            self._read_attr(tag, 'itererr', self.electrical)
            self._read_attr(tag, 'iterlim', self.electrical)
            self._read_attr(tag, 'logfreq', self.electrical)
        elif tag == 'temperature':
            self.thermal.temperature_boundary.read_from_xpl(tag, manager)
        elif tag == 'heatflux':
            self.thermal.heatflux_boundary.read_from_xpl(tag, manager)
        elif tag == 'convection':
            self.thermal.convection_boundary.read_from_xpl(tag, manager)
        elif tag == 'radiation':
            self.thermal.radiation_boundary.read_from_xpl(tag, manager)
        elif tag == 'voltage':
            self.electrical.voltage_boundary.read_from_xpl(tag, manager)

    def on_initialize(self):
        pass

    def on_invalidate(self):
        self.thermal.invalidate()
        self.electrical.invalidate()

    def compute(self, save=True, invalidate=True):
        """
        Run calculations.

        In the beginning the solvers are invalidated and next, the thermo-
        electric algorithm is executed until both solvers converge to the
        value specified in their configuration in the `maxerr` property.

        Args:
            save (bool or str): If `True` the computed fields are saved to the
                HDF5 file named after the script name with the suffix denoting
                either the batch job id or the current time if no batch system
                is used. The filename can be overridden by setting this parameter
                as a string.
            invalidate (bool): If this flag is set, solvers are invalidated
                               in the beginning of the computations.
        """
        if invalidate:
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

    def get_total_current(self, nact=0):
        """
        Get total current flowing through active region [mA]
        """
        return self.electrical.get_total_current(nact)

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
        h5file, group = h5open(filename, group)
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

    def plot_temperature(self, geometry_color='0.75', mesh_color=None, geometry_alpha=0.35, mesh_alpha=0.15, **kwargs):
        """
        Plot computed temperature to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification for
                the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            mesh_alpha (float): Mesh opacity (1 — fully opaque, 0 – invisible).

            kwargs: Keyword arguments passed to the plot function.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        """
        field = self.thermal.outTemperature(self.thermal.mesh)
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Temperature [K]")
        if geometry_color is not None:
            plask.plot_geometry(self.thermal.geometry, color=geometry_color, alpha=geometry_alpha)
        if mesh_color is not None:
            plask.plot_mesh(self.thermal.mesh, color=mesh_color, alpha=mesh_alpha)
        plask.window_title("Temperature")

    def plot_voltage(self, geometry_color='0.75', mesh_color=None, geometry_alpha=0.35, mesh_alpha=0.15, **kwargs):
        """
        Plot computed voltage to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            mesh_alpha (float): Mesh opacity (1 — fully opaque, 0 – invisible).

            kwargs: Keyword arguments passed to the :func:`plask.plot_field`.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        """
        field = self.electrical.outVoltage(self.electrical.mesh)
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Voltage [V]")
        if geometry_color is not None:
            plask.plot_geometry(self.electrical.geometry, color=geometry_color, alpha=geometry_alpha)
        if mesh_color is not None:
            plask.plot_mesh(self.electrical.mesh, color=mesh_color, alpha=mesh_alpha)
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
    """
    Thermo-electric calculations solver without the optical part.

    This solver performs under-threshold thermo-electrical computations.
    It computes electric current flow and tempereture distribution in a self-
    consistent loop until desired convergence is reached.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    Thermal = thermal.static.Static2D
    Electrical = electrical.shockley.Shockley2D

    outTemperature = property(lambda self: self.thermal.outTemperature,
                              doc=Thermal.outTemperature.__doc__)

    outHeatFlux = property(lambda self: self.thermal.outHeatFlux,
                           doc=Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=Thermal.outThermalConductivity.__doc__)

    outVoltage = property(lambda self: self.electrical.outVoltage,
                          doc=Electrical.outVoltage.__doc__)

    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=Electrical.outCurrentDensity.__doc__)

    outHeat = property(lambda self: self.electrical.outHeat,
                       doc=Electrical.outHeat.__doc__)

    outConductivity = property(lambda self: self.electrical.outConductivity,
                               doc=Electrical.outConductivity.__doc__)

    thermal = attribute(Thermal.__name__+"()")
    """
    :class:`thermal.static.Static2D` solver used for thermal calculations.
    """

    electrical = attribute(Electrical.__name__+"()")
    """
    :class:`electrical.shockley.Shockley2D` solver used for electrical calculations.
    """

    tfreq = 6.0
    """
    Number of electrical iterations per single thermal step.
    
    As temperature tends to converge faster, it is reasonable to repeat thermal
    solution less frequently.
    """


class ThermoElectricCyl(ThermoElectric):
    """
    Thermo-electric calculations solver without the optical part.

    This solver performs under-threshold thermo-electrical computations.
    It computes electric current flow and tempereture distribution in a self-
    consistent loop until desired convergence is reached.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    Thermal = thermal.static.StaticCyl
    Electrical = electrical.shockley.ShockleyCyl

    outTemperature = property(lambda self: self.thermal.outTemperature,
                              doc=Thermal.outTemperature.__doc__)

    outHeatFlux = property(lambda self: self.thermal.outHeatFlux,
                           doc=Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=Thermal.outThermalConductivity.__doc__)

    outVoltage = property(lambda self: self.electrical.outVoltage,
                          doc=Electrical.outVoltage.__doc__)

    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=Electrical.outCurrentDensity.__doc__)

    outHeat = property(lambda self: self.electrical.outHeat,
                       doc=Electrical.outHeat.__doc__)

    outConductivity = property(lambda self: self.electrical.outConductivity,
                               doc=Electrical.outConductivity.__doc__)

    thermal = attribute(Thermal.__name__+"()")
    """
    :class:`thermal.static.Static2D` solver used for thermal calculations.
    """

    electrical = attribute(Electrical.__name__+"()")
    """
    :class:`electrical.shockley.Shockley2D` solver used for electrical calculations.
    """

    tfreq = 6.0
    """
    Number of electrical iterations per single thermal step.
    
    As temperature tends to converge faster, it is reasonable to repeat thermal
    solution less frequently.
    """


class ThermoElectric3D(ThermoElectric):
    """
    Thermo-electric calculations solver without the optical part.

    This solver performs under-threshold thermo-electrical computations.
    It computes electric current flow and tempereture distribution in a self-
    consistent loop until desired convergence is reached.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    Thermal = thermal.static.Static3D
    Electrical = electrical.shockley.Shockley3D

    outTemperature = property(lambda self: self.thermal.outTemperature,
                              doc=Thermal.outTemperature.__doc__)

    outHeatFlux = property(lambda self: self.thermal.outHeatFlux,
                           doc=Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=Thermal.outThermalConductivity.__doc__)

    outVoltage = property(lambda self: self.electrical.outVoltage,
                          doc=Electrical.outVoltage.__doc__)

    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=Electrical.outCurrentDensity.__doc__)

    outHeat = property(lambda self: self.electrical.outHeat,
                       doc=Electrical.outHeat.__doc__)

    outConductivity = property(lambda self: self.electrical.outConductivity,
                               doc=Electrical.outConductivity.__doc__)

    thermal = attribute(Thermal.__name__+"()")
    """
    :class:`thermal.static.Static3D` solver used for thermal calculations.
    """

    electrical = attribute(Electrical.__name__+"()")
    """
    :class:`electrical.shockley.Shockley3D` solver used for electrical calculations.
    """

    tfreq = 6.0
    """
    Number of electrical iterations per single thermal step.
    
    As temperature tends to converge faster, it is reasonable to repeat thermal
    solution less frequently.
    """


__all__ = 'ThermoElectric2D', 'ThermoElectricCyl', 'ThermoElectric3D'