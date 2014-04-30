# -*- coding: utf-8 -*-
'''
High-level PLaSK algorithms for computations of semiconductor lasers.

TODO doc

'''
import os
import plask

try:
    import scipy
except ImportError:
    plask.print_log(plask.LOG_WARNING, "scipy could not be imported."
                                       " You will not be able to run some algorithms."
                                       " Install scipy to resolve this issue.")
else:
    import scipy.optimize

# Get unique suffix for savefiles
if 'JOB_ID' in os.environ:
    _suffix = "-" + os.environ['JOB_ID']
elif 'PBS_JOBID' in os.environ:
    _suffix = "-" + os.environ['PBS_JOBID']
else:
    import time
    _suffix = time.strftime("-%Y%m%d-%H%M", time.localtime())


def _h5_open(filename, group):
    import h5py
    if filename is None:
        import sys
        filename = sys.argv[0]
        if filename.endswith('.py'): filename = filename[:-3]
        elif filename.endswith('.xpl'): filename = filename[:-4]
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


class ThermoElectric(object):
    '''
    Algorithm for thermo-electric calculations without the optical part.

    This algorithm performs under-threshold thermo-electrical computations.
    It computes electric current flow and tempereture distribution in a self-
    consistent loop until desired convergence is reached.

    Args:
        thermal (solver): Configured thermal solver.
            It must have ``outTemperature`` provider and ``inHeat`` receiver.
            Temperature computations are done with ``compute`` method, which
            takes maximum number of iterations as input and returns maximum
            convergence error.

        electrical (solver): Configured electrical solver.
            It must have ``outHeat`` provider and ``inTemperature`` receiver.
            Computations are done with ``compute`` method, which takes maximum
            number of iterations as input and returns maximum convergence error.
            Solver specific parameters (e.g. ``beta``) should already be set
            before execution of the algorithm.

        tfreq (int): Number of electrical iterations per single thermal step.
            As temperature tends to converge faster, it is reasonable to repeat
            thermal solution less frequently.

        connect (bool): If True, solvers are automatically connected by the
            algorithm in its constructor.

    Solvers specified on construction of this algorithm are automatically
    connected if ``connect`` parameter is *True*. Then the computations can be
    executed using `run` method, after which the results may be save to the
    HDF5 file with `save` or presented visually using ``plot_...`` methods.
    If ``save`` parameter of the :meth:`run` method is *True* the fields are
    saved automatically after the computations. The file name is based on the
    name of the executed script with suffix denoting either the launch time or
    the identifier of a batch job if a batch system (like OpenPBS or SGE) is used.

    Example:

        The typical usage scenario of this algorithm is as follows:

        >>> task = algorithm.ThermoElectric(therm, electr)
        >>> task.run()
        >>> task.plot_junction_current()

        The code above launches the computations, assuming that `therm` is
        a configured thermal solver and ``electr`` an electrical one. Then the
        current density distribution along the junction is plotted.
    '''

    def __init__(self, thermal, electrical, tfreq=6, connect=True):
        electrical.inTemperature = thermal.outTemperature
        thermal.inHeat = electrical.outHeat
        self.thermal = thermal
        self.electrical = electrical
        self.tfreq = tfreq
        if connect:
            self.electrical.inTemperature = self.thermal.outTemperature
            self.thermal.inHeat = self.electrical.outHeat

    def run(self, save=True):
        '''
        Execute the algorithm.

        In the beginning the solvers are invalidated and next, the thermo-
        electric algorithm is executed until both solvers converge to the
        value specified in their configuration in the `maxerr` property.

        Args:
            save (bool or str): If `True` the computed fields are saved to the
                HDF5 file named after the script name with the suffix denoting
                either the batch job id or the current time if no batch system
                is used. The filename can be overriden by setting this parameted
                as a string.
        '''
        self.thermal.invalidate()
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
        Save the computation results to the HDF5 file.

        Args:
            filename (str): The file name to save to.
                If omitted, the file name is generated automatically based on
                the script name with suffix denoting either the batch job id or
                the current time if no batch system is used.

            group (str): HDF5 group to save the data under.
        '''
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


    def plot_temperature(self, geometry_color='w', mesh_color=None, **kwargs):
        '''
        Plot computed temperature to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification for
                the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.


            **kwargs: Keyword arguments passed to the plot function.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        '''
        field = self.thermal.outTemperature(self.thermal.mesh)
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Temperature [K]")
        if geometry_color is not None:
            plask.plot_geometry(self.thermal.geometry, color=geometry_color)
        if mesh_color is not None:
            plask.plot_mesh(self.thermal.mesh, color=mesh_color)
        plask.gcf().canvas.set_window_title("Temperature")


    def plot_voltage(self, geometry_color='w', mesh_color=None, **kwargs):
        '''
        Plot computed voltage to the current axes.

        Args:
            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            mesh_color (str or ``None``): Matplotlib color specification for
                the mesh. If ``None``, the mesh is not plotted.

            **kwargs: Keyword arguments passed to the :func:`plask.plot_field`.

        See also:
            :func:`plask.plot_field` : Plot any field obtained from receivers
        '''
        field = self.electrical.outVoltage(self.electrical.mesh)
        plask.plot_field(field, **kwargs)
        cbar = plask.colorbar(use_gridspec=True)
        cbar.set_label("Voltage [V]")
        if geometry_color is not None:
            plask.plot_geometry(self.electrical.geometry, color=geometry_color)
        if mesh_color is not None:
            plask.plot_mesh(self.electrical.mesh, color=mesh_color)
        plask.gcf().canvas.set_window_title("Voltage")


    def plot_vertical_voltage(self, at=0., **kwargs):
        '''
        Plot computed voltage along the vertical axis

        Args:
            at (float): Horizontal position of the axis at which the voltage
                        is plotted.

            **kwargs: Keyword arguments passed to the plot function.
        '''
        mesh = plask.mesh.Rectilinear2D([at], self.electrical.mesh.axis1)
        field = self.electrical.outVoltage(mesh)
        plask.plot(mesh.axis1, field, **kwargs)
        plask.xlabel(u"${}$ [\xb5m]".format(plask.config.axes[-1]))
        plask.ylabel("Voltage [V]")
        plask.gcf().canvas.set_window_title("Voltage")


    def plot_junction_current(self, refine=16, bounds=True, **kwargs):
        '''
        Plot current density at the active region.

        Args:
            refine (int): Number of points in the plot between each two points
                          in the computational mesh.
            bounds (bool): If *True* then the geometry objects boundaries are
                           plotted.

            **kwargs: Keyword arguments passed to the plot function.
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
                       label="Junction {:d}".format(i + 1), **kwargs)
        if len(act) > 1:
            plask.legend(loc='best')
        plask.xlabel(u"${}$ [\xb5m]".format(plask.config.axes[-2]))
        plask.ylabel(u"Current Density [kA/cm\xb2]")
        if bounds:
            simplemesh = plask.mesh.Rectilinear2D.SimpleGenerator()(
                             self.electrical.geometry.item
                         )
            for x in simplemesh.axis0:
                plask.axvline(x, ls=":", color="k")
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

    Args:
        thermal (solver): Configured thermal solver.
            It must have ``outTemperature`` provider and ``inHeat`` receiver.
            Temperature computations are done with ``compute`` method, which
            takes maximum number of iterations as input and returns maximum
            convergence error.

        electrical (solver): Configured electrical solver.
            It must have providers ``outHeat`` and ``outCurrentDensity`` as
            well as ``inTemperature`` receiver. Computations are done with
            ``compute`` method, which takes maximum number of iterations as
            input and returns maximum convergence error. Solver specific
            parameters (e.g. ``beta``) should already be set before execution
            of the algorithm.

        diffusion (solver or ``None``): Configured diffusion solver.
            It must have one provider ``outCarriersConcentration`` and two
            receivers: ``inTemperature`` and ``inCurrentDensity``. Under-
            threshold computations are done with ``compute_threshold`` method.
            If this parameter is ``None`` then it is assumed that its
            functionality is already ensured by the electrical solvers, which
            in such a case should have its own ``outCarriersConcentration``
            provider.

        gain (solver): Configured gain solver.
            It must have one provider ``outGain`` and two receivers:
            ``inTemperature`` and ``inCarriersConcentration``. This solver
            should be able to provide gain for a given wavelength without
            explicitly calling any computations. Hence, if you need to call
            any method to initialize the solver, do it before running this
            algorithm.

        optical (solver): Configured optical solver.
            It is required to have ``inTemperature`` and ``inGain`` receivers
            and ``outLightIntensity`` provider that is necessary only for
            plotting electromagnetic field profile. This solver needs to have
            ``find_mode`` method if ``quick`` is false or ``get_determinant``
            and ``set_mode`` methods is ``quick`` is true. Furhtermore it needs
            to have an attribute ``modes`` that is a list of the found modes.

        ivolt (int): Index in the ``voltage_boundary`` boundary conditions list
            in the ``electrical`` solver that is varied in order to find the
            threshold.

        vmin (float): Lower voltage limit for the threshold search. This should
             be the voltage value below the threshold.

        vmax (float): Upper voltage limit for the threshold search. This should
             be the voltage value above the threshold.

        approx_mode (float): Approximation of the optical mode
            (either the effective index or the wavelength) needed for optical
            computations.

        loss (str): Loss parameter returned by optical computations.
            Threshold is assumed to be found if either this value (for 'loss')
            or its imaginary part (for 'neff' or 'wavelength') is zero.

        tfreq (int): Number of electrical iterations per single thermal step.
            As temperature tends to converge faster, it is reasonable to repeat
            thermal solution less frequently.

        vtol (float): Tolerance on voltage in the root search.

        invalidate (bool): If this parameter is *True*, thermal and electrical
            solvers are invalidated in each root search iteration.

        quick (bool): If this parameter is *True*, the algorithm tries to find
            the threshold the easy way i.e. by computing only the optical
            determinant in each iteration.

        connect (bool): If True, solvers are automatically connected by the
            alogrithm in its constructor.

    Solvers specified on construction of this algorithm are automatically
    connected if ``connect`` parameter is *True*. Then the computations can be
    executed using `run` method, after which the results may be save to the
    HDF5 file with `save` or presented visually using ``plot_...`` methods.
    If ``save`` parameter of the :meth:`run` method is *True* the fields are
    saved automatically after the computations. The file name is based on the
    name of the executed script with suffix denoting either the launch time or
    the identifier of a batch job if a batch system (like OpenPBS or SGE) is used.

    Example:

        The typical usage scenario of this algorithm is as follows:

        >>> task = algorithm.ThresholdSearch(therm, electr, diff, gain, optic,
                1, 0., 3., 980.)
        >>> task.run()
        >>> I_th = electrical.get_total_current()
        >>> task.plot_gain()

        The code above launches the computations, assuming that `therm` is
        a configured thermal solver and ``electr`` an electrical one. Then the
        current density distribution along the junction is plotted.

    '''

    def __init__(self, thermal, electrical, diffusion, gain, optical,
                 ivolt, vmin, vmax, approx_mode, loss='auto', tfreq=6,
                 vtol=1e-6, invalidate=False, quick=False, connect=True):
        ThermoElectric.__init__(self, thermal, electrical, tfreq, connect)
        if diffusion is not None:
            self.diffusion = diffusion
            if connect:
                diffusion.inTemperature = thermal.outTemperature
                diffusion.inCurrentDensity = electrical.outCurrentDensity
        else:
            self.diffusion = electrical
        if connect:
            gain.inTemperature = thermal.outTemperature
            gain.inCarriersConcentration = diffusion.outCarriersConcentration
            optical.inTemperature = thermal.outTemperature
            optical.inGain = gain.outGain
        self.gain = gain
        self.optical = optical
        self.vmin = vmin
        self.vmax = vmax
        self.ivb = ivolt
        self.vtol = vtol
        self.optstart = approx_mode
        self.invalidate = invalidate
        self.threshold_current = None
        if quick:
            raise NotImplemented('Quick threshold search not implemented')
        if loss == 'auto':
            if type(optical.geometry) == plask.geometry.Cartesian2D:
                self.loss = 'neff'
            elif type(optical.geometry) == plask.geometry.Cylindrical2D:
                self.loss = 'loss'
            else:
                raise TypeError("Unable to automatically determine the mode of "
                                "optical calculations in 3D")
        else:
            self.loss = loss
        if self.loss not in ('neff', 'wavelength', 'loss'):
            raise ValueError("Wrong mode ('loss', 'neff', 'wavelength', or 'auto' allowed)")

    def run(self, save=True):
        '''
        Execute the algorithm.

        In the beginning the solvers are invalidated and next, the self-
        consistent loop of thermal, electrical, gain, and optical calculations
        are run within the root-finding algorithm until the mode is found
        with zero optical losses.

        Args:
            save (bool or str): If `True` the computed fields are saved to the
                HDF5 file named after the script name with the suffix denoting
                either the batch job id or the current time if no batch system
                is used. The filename can be overriden by setting this parameted
                as a string.

        Returns:
            The voltage set to ``ivolt`` boundary condition for the threshold.
            The threshold current can be then obtained by calling:

            >>> electrical.get_total_current()
            123.0
        '''

        self.thermal.invalidate()
        self.electrical.invalidate()
        self.diffusion.invalidate()
        self.optical.invalidate()

        def func(volt):
            '''Function to search zero of'''
            self.electrical.voltage_boundary[self.ivb].value = volt
            if self.invalidate:
                self.thermal.invalidate()
                self.electrical.invalidate()
            verr = 2. * self.electrical.maxerr
            terr = 2. * self.thermal.maxerr
            while terr > self.thermal.maxerr or verr > self.electrical.maxerr:
                verr = self.electrical.compute(self.tfreq)
                terr = self.thermal.compute(1)
            try: self.diffusion.compute_threshold()
            except AttributeError: pass
            nm = self.optical.find_mode(self.optstart)
            val = self.optical.modes[nm].__getattribute__(self.loss)
            if self.loss != 'loss': val = val.imag
            return val

        result = scipy.optimize.brentq(func, self.vmin, self.vmax, xtol=self.vtol)
        self.threshold_current = self.electrical.get_total_current()

        if save:
            self.save(None if save is True else save)

        return result

    def save(self, filename=None, group='ThresholdSearch'):
        '''
        Save the computation results to the HDF5 file.

        Args:
            filename (str): The file name to save to.
                If omitted, the file name is generated automatically based on
                the script name with suffix denoting either the batch job id or
                the current time if no batch system is used.

            group (str): HDF5 group to save the data under.
        '''
        h5file, group = _h5_open(filename, group)
        ThermoElectric.save(self, h5file, group)


