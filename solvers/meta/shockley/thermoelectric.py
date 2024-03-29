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

# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology

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
        from os.path import basename, abspath
        filename = sys.argv[0]
        if filename.endswith('.py'): filename = filename[:-3]
        elif filename.endswith('.xpl'): filename = filename[:-4]
        elif filename == '': filename = 'console'
        filename = abspath(basename(filename))
        filename += suffix + '.h5'
    if type(filename) is h5py.File:
        h5file = filename
        filename = h5file.filename
        close = False
    else:
        h5file = h5py.File(filename, 'a')
        close = True
    group_parts = [g for g in group.split('/') if g]
    orig_group = group = '/'.join(group_parts)
    idx = 1
    while group in h5file:
        group = '/'.join(["{}-{}".format(group_parts[0], idx)] + group_parts[1:])
        idx += 1
    if group is not orig_group:
        plask.print_log('warning',
                        "Group '/{}' exists in HDF5 file '{}'. Saving to group '/{}'" \
                        .format(orig_group, filename, group))
    group = '/' + group
    return h5file, group, filename, close


class ThermoElectric(plask.Solver):

    _Thermal = None
    _Electrical = None

    def __init__(self, name):
        super().__init__(name)
        self.thermal = self._Thermal(name)
        self.electrical = self._Electrical(name)
        self.__reconnect()

        self.tfreq = 6

    def __reconnect(self):
        self.electrical.inTemperature = self.thermal.outTemperature
        self.thermal.inHeat = self.electrical.outHeat

    def reconnect(self):
        """
        Reconnect all internal solvers.

        This method should be called if some of the internal solvers were changed manually.
        """
        self.__reconnect()

    def _read_attr(self, tag, attr, solver, type=None, pyattr=None):
        if pyattr is None: pyattr = attr
        if attr in tag:
            try:
                if type is not None:
                    val = type(tag[attr])
                else:
                    val = tag[attr]
            except ValueError:
                raise plask.XMLError("{}: {} attribute {} has illegal value '{}'".format(
                    tag, type.__name__.title(), attr, tag[attr]))
            else:
                setattr(solver, pyattr, val)

    def load_xpl(self, xpl, manager):
        for tag in xpl:
            self._parse_xpl(tag, manager)

    def _parse_fem_tag(self, tag, manager, solver):
        self._read_attr(tag, 'algorithm', solver)
        for it in tag:
            if it == 'iterative':
                self._read_attr(it, 'accelerator', solver.iterative)
                self._read_attr(it, 'preconditioner', solver.iterative)
                self._read_attr(it, 'noconv', solver.iterative)
                self._read_attr(it, 'maxit', solver.iterative, int)
                self._read_attr(it, 'maxerr', solver.iterative, float)
                self._read_attr(it, 'nfact', solver.iterative, int)
                self._read_attr(it, 'omega', solver.iterative, float)
                self._read_attr(it, 'ndeg', solver.iterative, int)
                self._read_attr(it, 'lvfill', solver.iterative, int)
                self._read_attr(it, 'ltrunc', solver.iterative, int)
                self._read_attr(it, 'nsave', solver.iterative, int)
                self._read_attr(it, 'nrestart', solver.iterative, int)
            else:
                raise plask.XMLError("{}: Unrecognized tag '{}'".format(it, it.name))

    def _parse_xpl(self, tag, manager):
        if tag == 'geometry':
            self.thermal.geometry = tag.getitem(manager.geo, 'thermal')
            self.electrical.geometry = tag.getitem(manager.geo, 'electrical')
        elif tag == 'mesh':
            self.thermal.mesh = tag.getitem(manager.msh, 'thermal')
            self._read_attr(tag, 'empty-elements', self.electrical, str, 'empty_elements')
            self.electrical.mesh = tag.getitem(manager.msh, 'electrical')
        elif tag == 'junction':
            self._read_attr(tag, 'pnjcond', self.electrical, float)
            for key, val in tag.attrs.items():
                if key.startswith('beta') or key.startswith('js'):
                    try:
                        val = float(val)
                    except ValueError:
                        raise plask.XMLError("{}: Float attribute '{}' has illegal value '{}'".format(tag, key, val))
                    setattr(self.electrical, key, val)
        elif tag == 'contacts':
            self._read_attr(tag, 'pcond', self.electrical, float)
            self._read_attr(tag, 'ncond', self.electrical, float)
        elif tag == 'loop':
            self.tfreq = int(tag.get('tfreq', self.tfreq))
            self._read_attr(tag, 'inittemp', self.thermal, float, 'inittemp')
            self._read_attr(tag, 'maxterr', self.thermal, float, 'maxerr')
            self._read_attr(tag, 'maxcerr', self.electrical, float, 'maxerr')
        elif tag == 'tmatrix':
            self._parse_fem_tag(tag, manager, self.thermal)
        elif tag == 'ematrix':
            self._parse_fem_tag(tag, manager, self.electrical)
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
        else:
            raise plask.XMLError("{}: Unrecognized tag '{}'".format(tag, tag.name))

    def on_initialize(self):
        self.thermal.initialize()
        self.electrical.initialize()

    def on_invalidate(self):
        self.thermal.invalidate()
        self.electrical.invalidate()

    def compute(self, save=True, invalidate=True, group='ThermoElectric'):
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

            group (str): HDF5 group to save the data under.
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

        infolines = self._get_info()
        plask.print_log('important', "Thermoelectric Computations Finished")
        for line in infolines:
            plask.print_log('important', "  " + line)

        if save:
            self.save(None if save is True else save, group)

    def get_total_current(self, nact=0):
        """
        Get total current flowing through active region (mA)
        """
        return self.electrical.get_total_current(nact)

    @staticmethod
    def _iter_levels(geometry, mesh, *required):
        if isinstance(mesh, (plask.mesh.Mesh1D, plask.ndarray, list, tuple)):
            hor = mesh,
            Mesh = plask.mesh.Rectangular2D
        elif isinstance(mesh, plask.mesh.Rectangular2D):
            hor = mesh.axis0,
            Mesh = plask.mesh.Rectangular2D
        elif isinstance(mesh, plask.mesh.Rectangular3D):
            hor = mesh.axis0, mesh.axis1
            Mesh = plask.mesh.Rectangular3D
        else:
            return

        points = Mesh.SimpleGenerator()(geometry).elements.mesh
        levels = {}
        for p in points:
            roles = geometry.get_roles(p)
            suffix = None
            for role in roles:
                for prefix in ('active', 'junction'):
                    if role.startswith(prefix):
                        suffix = role[len(prefix):]
                        break
            if suffix is not None and (not required or any(role in required for role in roles)):
                levels[suffix] = p[-1]

        for name, v in levels.items():
            axs = hor + ([v],)
            mesh2 = Mesh(*axs)
            yield name, mesh2

    def _save_thermoelectric(self, h5file, group):
        tmesh = self.thermal.mesh
        vmesh = self.electrical.mesh
        jmesh = vmesh.elements.mesh
        temp = self.thermal.outTemperature(tmesh)
        volt = self.electrical.outVoltage(vmesh)
        curr = self.electrical.outCurrentDensity(jmesh)
        plask.save_field(temp, h5file, group + '/Temperature')
        plask.save_field(volt, h5file, group + '/Potential')
        plask.save_field(curr, h5file, group + '/CurrentDensity')
        for name, jmesh2 in self._iter_levels(self.electrical.geometry, jmesh):
            curr2 = self.electrical.outCurrentDensity(jmesh2)
            plask.save_field(curr2, h5file, group + '/Junction' + name + 'CurrentDensity')

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
        h5file, group, filename, close = h5open(filename, group)
        self._save_thermoelectric(h5file, group)
        if close:
            h5file.close()
        plask.print_log('info', "Fields saved to file '{}' in group '{}'".format(filename, group))
        return filename

    def get_temperature(self):
        """
        Get temperature on a thermal mesh.
        """
        return self.thermal.outTemperature(self.thermal.mesh)

    def get_voltage(self):
        """
        Get voltage on an electrical mesh.
        """
        return self.electrical.outVoltage(self.electrical.mesh)

    def get_vertical_voltage(self, at=0):
        """
        Get computed voltage along the vertical axis.

        Args:
            at (float): Horizontal position of the axis at which the voltage
                        is plotted.
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
        return field

    def get_junction_currents(self, refine=16, interpolation='linear'):
        """
        Get current densities at the active regions.

        Args:
            refine (int): Number of points in the plot between each two points
                          in the computational mesh.

            interpolation (str): Interpolation used when retrieving current density.

        Return:
            dict: Dictionary of junction current density data.
                  Keys are the junction number.
        """
        axis = plask.concatenate([
            plask.linspace(x, self.electrical.mesh.axis0[i+1], refine+1)
            for i,x in enumerate(list(self.electrical.mesh.axis0)[:-1])
        ])

        result = {}
        for lb, msh in self._iter_levels(self.electrical.geometry, axis):
            if not lb:
                lb = 0
            else:
                try: lb = int(lb)
                except ValueError: pass
            result[lb] = self.electrical.outCurrentDensity(msh, interpolation).array[:,0,1]
        return result

    def plot_temperature(self, geometry_color='0.75', mesh_color=None, geometry_alpha=0.35, mesh_alpha=0.15,
                         geometry_lw=1.0, mesh_lw=1.0, **kwargs):
        """
        Plot computed temperature to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification for
                the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            mesh_alpha (float): Mesh opacity (1 — fully opaque, 0 – invisible).

            geometry_lw (float): Line width for geometry.

            mesh_lw (float): Line width for mesh.

            **kwargs: Keyword arguments passed to the plot function.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        """
        field = self.get_temperature()
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Temperature (K)")
        if geometry_color is not None:
            plask.plot_geometry(self.thermal.geometry, color=geometry_color, alpha=geometry_alpha, lw=geometry_lw)
        if mesh_color is not None:
            plask.plot_mesh(self.thermal.mesh, color=mesh_color, alpha=mesh_alpha, lw=mesh_lw)
        plask.window_title("Temperature")

    def plot_voltage(self, geometry_color='0.75', mesh_color=None, geometry_alpha=0.35, mesh_alpha=0.15,
                     geometry_lw=1.0, mesh_lw=1.0, **kwargs):
        """
        Plot computed voltage to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            mesh_alpha (float): Mesh opacity (1 — fully opaque, 0 – invisible).

            geometry_lw (float): Line width for geometry.

            mesh_lw (float): Line width for mesh.

            **kwargs: Keyword arguments passed to the :func:`plask.plot_field`.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        """
        field = self.get_voltage()
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Voltage (V)")
        if geometry_color is not None:
            plask.plot_geometry(self.electrical.geometry, color=geometry_color, alpha=geometry_alpha, lw=geometry_lw)
        if mesh_color is not None:
            plask.plot_mesh(self.electrical.mesh, color=mesh_color, alpha=mesh_alpha, lw=mesh_lw)
        plask.window_title("Voltage")

    def plot_vertical_voltage(self, at=0., **kwargs):
        """
        Plot computed voltage along the vertical axis.

        Args:
            at (float): Horizontal position of the axis at which the voltage
                        is plotted.

            **kwargs: Keyword arguments passed to the plot function.
        """
        field = self.get_vertical_voltage(at)
        plask.plot(field.mesh.axis1, field, **kwargs)
        plask.xlabel(u"${}$ (µm)".format(plask.config.axes[-1]))
        plask.ylabel("Voltage (V)")
        plask.window_title("Voltage")

    def _plot_hbounds(self, solver):
        simplemesh = plask.mesh.Rectangular2D.SimpleGenerator()(solver.geometry.item)
        for x in simplemesh.axis0:
            plask.axvline(x,
                          linestyle=plask.rc.grid.linestyle, linewidth=plask.rc.grid.linewidth,
                          color=plask.rc.grid.color, alpha=plask.rc.grid.alpha)

    def plot_junction_current(self, refine=16, bounds=True, interpolation='linear', label=None, **kwargs):
        """
        Plot current density at the active region.

        Args:
            refine (int): Number of points in the plot between each two points
                          in the computational mesh.

            bounds (bool): If *True* then the geometry objects boundaries are
                           plotted.

            interpolation (str): Interpolation used when retrieving current density.

            label (str or sequence): Label for each junction. It can be a sequence of
                                     consecutive labels for each junction, or a string
                                     in which case the same label is used for each
                                     junction. If omitted automatic label is generated.

            **kwargs: Keyword arguments passed to the plot function.
        """
        axis = plask.concatenate([
            plask.linspace(x, self.electrical.mesh.axis0[i+1], refine+1)
            for i,x in enumerate(list(self.electrical.mesh.axis0)[:-1])
        ])

        i = 0
        for i, (lb, msh) in enumerate(self._iter_levels(self.electrical.geometry, axis)):
            curr = self.electrical.outCurrentDensity(msh, interpolation).array[:,0,1]
            s = sum(curr)
            if label is None:
                lab = "Junction {:s}".format(lb)
            elif isinstance(label, tuple) or isinstance(label, tuple):
                lab = label[i]
            else:
                lab = label
            plask.plot(msh.axis0, curr if s > 0 else -curr,
                       label=lab, **kwargs)
        if i > 0:
            plask.legend(loc='best')
        plask.xlabel(u"${}$ (µm)".format(plask.config.axes[-2]))
        plask.ylabel(u"Current Density (kA/cm\xb2)")
        if bounds:
            self._plot_hbounds(self.electrical)
        plask.window_title("Current Density")

    def _get_defines_info(self):
        try:
            import __main__
            defines = ["  {} = {}".format(key, __main__.DEF[key]) for key in __main__.__overrites__]
        except (NameError, KeyError, AttributeError):
            defines = []
        if defines:
            defines = ["Temporary defines:"] + defines
        return defines

    def _get_info(self):
        info = self._get_defines_info()
        try:
            info.append("Total current (mA):            {:8.3f}".format(self.get_total_current()))
        except:
            pass
        try:
            info.append("Maximum temperature (K):       {:8.3f}".format(max(self.get_temperature())))
        except:
            pass
        return info


class ThermoElectric2D(ThermoElectric):
    """
    Thermo-electric calculations solver without the optical part.

    This solver performs under-threshold thermo-electrical computations.
    It computes electric current flow and temperature distribution in a self-
    consistent loop until desired convergence is reached.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    _Thermal = thermal.static.Static2D
    _Electrical = electrical.shockley.Shockley2D

    outTemperature = property(lambda self: self.thermal.outTemperature,
                              doc=_Thermal.outTemperature.__doc__)

    outHeatFlux = property(lambda self: self.thermal.outHeatFlux,
                           doc=_Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=_Thermal.outThermalConductivity.__doc__)

    outVoltage = property(lambda self: self.electrical.outVoltage,
                          doc=_Electrical.outVoltage.__doc__)

    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=_Electrical.outCurrentDensity.__doc__)

    outHeat = property(lambda self: self.electrical.outHeat,
                       doc=_Electrical.outHeat.__doc__)

    outConductivity = property(lambda self: self.electrical.outConductivity,
                               doc=_Electrical.outConductivity.__doc__)

    thermal = attribute(_Thermal.__name__+"()")
    """
    :class:`thermal.static.Static2D` solver used for thermal calculations.
    """

    electrical = attribute(_Electrical.__name__+"()")
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
    It computes electric current flow and temperature distribution in a self-
    consistent loop until desired convergence is reached.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    _Thermal = thermal.static.StaticCyl
    _Electrical = electrical.shockley.ShockleyCyl

    outTemperature = property(lambda self: self.thermal.outTemperature,
                              doc=_Thermal.outTemperature.__doc__)

    outHeatFlux = property(lambda self: self.thermal.outHeatFlux,
                           doc=_Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=_Thermal.outThermalConductivity.__doc__)

    outVoltage = property(lambda self: self.electrical.outVoltage,
                          doc=_Electrical.outVoltage.__doc__)

    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=_Electrical.outCurrentDensity.__doc__)

    outHeat = property(lambda self: self.electrical.outHeat,
                       doc=_Electrical.outHeat.__doc__)

    outConductivity = property(lambda self: self.electrical.outConductivity,
                               doc=_Electrical.outConductivity.__doc__)

    thermal = attribute(_Thermal.__name__+"()")
    """
    :class:`thermal.static.Static2D` solver used for thermal calculations.
    """

    electrical = attribute(_Electrical.__name__+"()")
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
    It computes electric current flow and temperature distribution in a self-
    consistent loop until desired convergence is reached.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    _Thermal = thermal.static.Static3D
    _Electrical = electrical.shockley.Shockley3D

    outTemperature = property(lambda self: self.thermal.outTemperature,
                              doc=_Thermal.outTemperature.__doc__)

    outHeatFlux = property(lambda self: self.thermal.outHeatFlux,
                           doc=_Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=_Thermal.outThermalConductivity.__doc__)

    outVoltage = property(lambda self: self.electrical.outVoltage,
                          doc=_Electrical.outVoltage.__doc__)

    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=_Electrical.outCurrentDensity.__doc__)

    outHeat = property(lambda self: self.electrical.outHeat,
                       doc=_Electrical.outHeat.__doc__)

    outConductivity = property(lambda self: self.electrical.outConductivity,
                               doc=_Electrical.outConductivity.__doc__)

    thermal = attribute(_Thermal.__name__+"()")
    """
    :class:`thermal.static.Static3D` solver used for thermal calculations.
    """

    electrical = attribute(_Electrical.__name__+"()")
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
