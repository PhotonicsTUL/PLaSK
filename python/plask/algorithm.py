# -*- coding: utf-8 -*-
'''
This module contains the set of high-level PLaSK algorithm typically used
in computations of semiconductor lasers.

TODO doc

'''
import os
import plask

# Get unique suffix for savefiles
if 'JOB_ID' in os.environ:
    _suffix = "-" + os.environ['JOB_ID']
elif 'PBS_JOBID' in os.environ:
    _suffix = "-" + os.environ['PBS_JOBID']
else:
    import time
    _suffix = time.strftime("-%Y%m%d-%H%M", time.gmtime())


class ThermoElectric(object):
    '''
    Algorithm for thermo-electric calculations without the optical part.

    This algorithm performs under-threshold thermo-electrical computations.
    It computes electric current flow and tempereture distribution in a self-
    consistent loop until desired convergence is reached.

        ``ThermoElectric(thermal, electrical, tfreq=6)``

    :param solver thermal:
        Configured thermal solver. In must have ``outTemperature`` provider and
        ``inHeat`` receiver. Temperature computations are done with ``compute``
        method, which takes maximum number of iterations as input and returns
        maximum convergence error.
    :param solver electrical:
        Configured electrical solver. It must have ``outHeat`` provider and
        ``inTemperature`` receiver. Computations are done with ``compute`` method,
        which takes maximum number of iterations as input and returns maximum
        convergence error. Solver specific parameters (e.g. ``beta``) should
        already be set before execution of the algorithm.
    :param integer tfreq:
        Number of electrical iterations per single thermal step. As temperature
        tends to converge faster, it is reasonable to repeat thermal solution
        less frequently.

    Solvers specified on construction of this algorithm are automatically
    connected. Then the computations can be executed using `run` method, after
    which the results may be save to the HDF5 file with `save` or presented
    visually using ``plot_...`` methods.

    '''

    def __init__(self, thermal, electrical, tfreq=6):
        electrical.inTemperature = thermal.outTemperature
        thermal.inHeat = electrical.outHeat
        self.thermal = thermal
        self.electrical = electrical
        self.tfreq = tfreq


    def run(self, save=True):
        '''
        Execute the algorithm.

        In the beginning the solvers are invalidated and next, the thermo-
        electric algorithm is executed until both solvers converge to the
        value specified in their configuration in the `maxerr` property.

        :param save:
            If `True` the computed fields are saved to the HDF5 file named
            after the script name with the suffix denoting either the batch job
            id or the current time if no batch system is used. The filename can
            be overriden by setting this parameted as a string.
        :type save: bool or str
        '''
        self.thermal.invalidate()
        self.electrical.invalidate()
        self.electrical.invalidate()

        verr = 2. * self.electrical.maxerr
        terr = 2. * self.thermal.maxerr
        while terr > self.thermal.maxerr or verr > self.electrical.maxerr:
            verr = self.electrical.compute(self.tfreq)
            terr = self.thermal.compute(1)

        if save:
            self.save(None if save is True else save)

    def save(self, filename=None, group='ThermoElectric'):
        '''
        Save the comutation results to the HDF5 file.

        :param str filename:
            The file name to save to. If omitted, the file name is generated
            automatically based on the script name with suffix denoting either
            the batch job id or the current time if no batch system is used.
        :param str group:
            HDF5 group to save the data under.
        '''
        if filename is None:
            import sys
            filename = sys.argv[0]
            if filename.endswith('.py'): filename = filename[:-3]
            elif filename.endswith('.xpl'): filename = filename[:-4]
            filename += _suffix + '.h5'
        tmesh = self.thermal.mesh
        vmesh = self.thermal.mesh
        jmesh = vmesh.get_midpoints()
        temp = self.thermal.outTemperature(tmesh)
        volt = self.electrical.outVoltage(vmesh)
        curr = self.electrical.outCurrentDensity(jmesh)
        import h5py
        h5file = h5py.File(filename, 'a')
        plask.save_field(temp, h5file, group + '/Temperature')
        plask.save_field(volt, h5file, group + '/Potential')
        plask.save_field(curr, h5file, group + '/CurrentDensity')


    def plot_temperature(self, geometry_color='w', mesh_color=None):
        '''
        Plot computed temperature to the current axes.

        :param geometry_color:
            Matplotlib color specification of the geometry. If None, structure
            is not plotted.
        :type geometry_color: str or None
        :param mesh_color:
            Matplotlib color specification of the mesh. If None, the mesh is
            not plotted.
        :type mesh_color: str or None

        .. seealso::

            plask.plot_field : Plot any field obtained from receivers
        '''
        field = self.thermal.outTemperature(self.thermal.mesh)
        plask.plot_field(field)
        cbar = plask.colorbar(use_gridspec=True)
        plask.xlabel(u"$%s$ [\xb5m]" % plask.config.axes[-2])
        plask.ylabel(u"$%s$ [\xb5m]" % plask.config.axes[-1])
        cbar.set_label("Temperature [K]")
        if geometry_color is not None:
            plask.plot_geometry(self.thermal.geometry, color=geometry_color)
        if mesh_color is not None:
            plask.plot_mesh(self.thermal.mesh, color=mesh_color)
        plask.gcf().canvas.set_window_title("Temperature")


    def plot_voltage(self, geometry_color='w', mesh_color=None):
        '''
        Plot computed voltage to the current axes.

        :param geometry_color:
            Matplotlib color specification of the geometry. If None, structure
            is not plotted.
        :type geometry_color: str or None
        :param mesh_color:
            Matplotlib color specification of the mesh. If None, the mesh is
            not plotted.
        :type mesh_color: str or None

        .. seealso::

            plask.plot_field : Plot any field obtained from receivers
        '''
        field = self.electrical.outVoltage(self.electrical.mesh)
        plask.plot_field(field)
        cbar = plask.colorbar(use_gridspec=True)
        plask.xlabel(u"$%s$ [\xb5m]" % plask.config.axes[-2])
        plask.ylabel(u"$%s$ [\xb5m]" % plask.config.axes[-1])
        cbar.set_label("Voltage [V]")
        if geometry_color is not None:
            plask.plot_geometry(self.electrical.geometry, color=geometry_color)
        if mesh_color is not None:
            plask.plot_mesh(self.electrical.mesh, color=mesh_color)
        plask.gcf().canvas.set_window_title("Voltage")


    def plot_vertical_voltage(self, at=0.):
        '''
        Plot computed voltage along the vertical axis

        :param float at:
            Horizontal position of the axis at which the voltage is plotted.
        '''
        mesh = plask.mesh.Rectilinear2D([at], self.electrical.mesh.axis1)
        field = self.electrical.outVoltage(mesh)
        plask.plot(mesh.axis1, field)
        plask.xlabel(u"$%s$ [\xb5m]" % plask.config.axes[-1])
        plask.ylabel("Voltage [V]")
        plask.gcf().canvas.set_window_title("Voltage")


    def plot_junction_current(self, refine=16):
        '''
        Plot current density at the active region.

        Parameters
        ----------
        refine : integer, optional
            Numeber of points in the plot between each two points in
            the computational mesh.
        '''
        # A little magic to get junction position first
        points = self.electrical.mesh.get_midpoints()
        geom = self.electrical.geometry.item
        yy = plask.unique(points.index1(i) for i,p in enumerate(points)
                          if geom.has_role('junction', p)
                          or geom.has_role('active', p))
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
            msh = plask.mesh.Rectilinear2D(axis, [y])
            curr = self.electrical.outCurrentDensity(msh, 'spline').array[:,0,1]
            s = sum(curr)
            plask.plot(msh.axis0, curr if s > 0 else -curr,
                       label="Junction %d" % (i + 1))
        if len(act) > 1:
            plask.legend(loc='best')
        plask.xlabel(u"$%s$ [\xb5m]" % plask.config.axes[-2])
        plask.ylabel(u"Current Density [kA/cm\xb2]")
        plask.gcf().canvas.set_window_title("Current Density")



class ThresholdSearch(ThermoElectric):
    '''
    Algorithm for threshold search of semiconductor laser.

    This algorithm performs thermo-electrical computations followed by
    determination ot threshold current and optical analysis in order to
    determine the threshold of a semiconductor laser. The search is
    performed by scipy root finding algorithm in order to determine
    the volatage and electric current ensuring no optical loss in the
    laser cavity.

    ThresholdSearch(thermal, electrical, diffusion, gain, optical,
                    ivolt, vmin, vmax, approx_mode, mode='auto', tfreq=6,
                    quick=False)

    Parameters
    ----------
    thermal : thermal solver
        Configured thermal solver. In must have `outTemperature` provider and
        `inHeat` receiver. Temperature computations are done with `compute`
        method, which takes maximum number of iterations as input and returns
        maximum convergence error.
    electrical : electrical solver
        Configured electrical solver. It must have providers `outHeat` and
        `outCurrentDensity` as well as `inTemperature` receiver. Computations
        are done with `compute` method, which takes maximum number of iterations
        as input and returns maximum convergence error. Solver specific parameters
        (e.g. `beta`) should already be set before execution of the algorithm.
    diffusion : diffusion solver or None
        Configured solver computing carriers diffusion. It must have one provider:
        `outCarriersConcentration` and two receivers: `inTemperature` and
        `inCurrentDensity`. Under-threshold computations are done with `compute`
        method. If this parameter is None then it is assumed that its functionality
        is already ensured by the electrical solvers, which in such a case should
        have its own `outCarriersConcentration` provider.
    gain : gain solver
        Configured gain solver. TODO
    optical : optical solver
        Configured optical solver. It is required to have `inTemperature` and
        `inGain` receivers and `outLightIntensity` provider that is necessary only
        for plotting electromagnetic field profile. This solver needs to have
        `find_mode` method if `quick` is false or `get_detrminant` and `set_mode`
        methods is `quick` is true. TODO
    ivolt : integer
        Index in the `voltage_boundary` boundary conditions list in the
        `electrical` solver that
    vmin : float
        TODO
    vmax : float
        TODO
    approx_mode : float
        Approximation of the optical mode (either the effective index or
        the wavelength) needed for optical computation.
    mode : string, optional
        TODO: 'neff', 'wavelength' or 'auto'
    tfreq : integer, optional
        Number of electrical iterations per single thermal step. As temperature
        tends to converge faster, it is reasonable to repeat thermal solution
        less frequently.
    quick : bool, optional
        If this parameter is True, the algorithm tries to find the threshold
        the easy way i.e. by computing only the optical determinant in each
        iteration.

    Solvers specified on construction of this algorithm are automatically
    connected. Then the computations can be executed using `run` method, after
    which the results may be save to the HDF5 file with `save` or presented
    visually using `plot_`... methods.

    '''

    def __init__(self, thermal, electrical, diffusion, gain, optical,
                 ivolt, vmin, vmax, approx_mode, mode='auto', tfreq=6,
                 quick=False):
        ThermoElectric.__init__(self, thermal, electrical, tfreq)
        if diffusion is not None:
            diffusion.inTemperature = thermal.outTemperature
            diffusion.inCurrentDensity = electrical.outCurrentDensity
            self.diffusion = diffusion
        else:
            self.diffusion = electrical
        self.gain = gain
        self.optical = optical
        self.ab = vmin,vmax
        self.ivb = ivolt
        self.optstart = approx_mode
        self.quick = quick
        if mode == 'auto':
            if type(otpical.geometry) == plask.geometry.Cartesian2D:
                self.mode = 'neff'
            elif type(otpical.geometry) == plask.geometry.Cylindrical2D:
                self.mode = 'wavelength'
            else:
                raise TypeError("unable to determine the mode automatically in 3D")
        else:
            self.mode = mode
        if self.mode not in ('neff', 'wavelength'):
            raise ValueError("wrong mode ('neff', 'wavelength', or 'auto' allowed)")

    def run(self):
        '''
        Execute the algorithm.

        In the beginning the solvers are invalidated and next, the self-
        consistent loop of thermal, electrical, gain, and optical calculations
        are run within the root-finding algorithm until the mode is found
        with zero optical losses.

        Returns
        -------
        float
            The voltage set to `ivolt` boundary condition for the threshold.
            The threshold current can be then obtained by calling

            >>> electrical.get_total_current()
            123.0
        '''
        pass
