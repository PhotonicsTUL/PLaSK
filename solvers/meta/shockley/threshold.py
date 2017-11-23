# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology

import numpy as np

import electrical.diffusion
import electrical.shockley
import gain.freecarrier
import plask
import thermal.static

try:
    import scipy
except ImportError:
    plask.print_log('warning', "scipy could not be imported."
                               " You will not be able to run some algorithms."
                               " Install scipy to resolve this issue.")
else:
    import scipy.optimize

from .thermoelectric import attribute, h5open, ThermoElectric
from . import _doc


class ThresholdSearch(ThermoElectric):

    _OPTICAL_ROOTS = {}

    Diffusion = None
    Gain = None
    Optical = None
    _optarg = 'lam'
    _lam0 = 'lam0'

    tfreq = 6.0
    vmin = None
    vmax = None
    optical_resolution = (800, 600)
    vtol = 1e-5
    quick = False
    maxiter = 50
    skip_thermal = False

    def __init__(self, name):
        super(ThresholdSearch, self).__init__(name)
        self.diffusion = self.Diffusion(name)
        self.gain = self.Gain(name)
        self.optical = self.Optical(name)
        self.diffusion.inTemperature = self.thermal.outTemperature
        self.diffusion.inCurrentDensity = self.electrical.outCurrentDensity
        self.gain.inTemperature = self.thermal.outTemperature
        self.gain.inCarriersConcentration = self.diffusion.outCarriersConcentration
        self.optical.inTemperature = self.thermal.outTemperature
        self.optical.inGain = self.gain.outGain
        self.threshold_voltage = None
        self.threshold_current = None
        self._invalidate = None
        self.modeno = None

    def _parse_xpl(self, tag, manager):
        if tag == 'root':
            self.vmin = tag.get('vmin', self.vmin)
            self.vmax = tag.get('vmax', self.vmax)
            self.ivb = tag['bcond']
            self.vtol = tag.get('vtol', self.vtol)
            self.maxiter = tag.get('maxiter', self.maxiter)
            self.quick = tag.get('quick', self.quick)
        elif tag == 'diffusion':
            self._read_attr(tag, 'fem-method', self.diffusion, str, 'fem_method')
            self._read_attr(tag, 'accuracy', self.diffusion, float)
            self._read_attr(tag, 'abs-accuracy', self.diffusion, float, 'abs_accuracy')
            self._read_attr(tag, 'maxiters', self.diffusion, int)
            self._read_attr(tag, 'maxrefines', self.diffusion, int)
            self._read_attr(tag, 'interpolation', self.diffusion, str)
        elif tag == 'gain':
            self._read_attr(tag, 'lifetime', self.gain, float)
            self._read_attr(tag, 'matrix-elem', self.gain, float, 'matrix_element')
            self._read_attr(tag, 'strained', self.gain, bool)
        elif tag.name in self._OPTICAL_ROOTS:
            root = getattr(self.optical, self._OPTICAL_ROOTS[tag.name])
            self._read_attr(tag, 'method', root, str)
            self._read_attr(tag, 'tolx', root, float)
            self._read_attr(tag, 'tolf-min', root, float, 'tolf_min')
            self._read_attr(tag, 'tolf-max', root, float, 'tolf_max')
            self._read_attr(tag, 'maxstep', root, float)
            self._read_attr(tag, 'maxiter', root, int)
            self._read_attr(tag, 'alpha', root, float)
            self._read_attr(tag, 'lambda', root, float)
            self._read_attr(tag, 'initial-range', root, tuple, 'initial_range')
        elif tag == 'output':
            self.optical_resolution = tag.get('optical-res-x', self.optical_resolution[0]), \
                                      tag.get('optical-res-y', self.optical_resolution[1])
        else:
            if tag == 'geometry':
                self.optical.geometry = self.diffusion.geometry = self.gain.geometry = \
                    tag.getitem(manager.geo, 'optical')
            elif tag == 'mesh':
                if 'diffusion' in tag: self.diffusion.mesh = tag.getitem(manager.msh, 'diffusion')
                if 'optical' in tag: self.optical.mesh = tag.getitem(manager.msh, 'optical')
                if 'gain' in tag: self.gain.mesh = tag.getitem(manager.msh, 'gain')
            elif tag == 'loop':
                self.skip_thermal = tag.get('skip-thermal', self.skip_thermal)
                self._read_attr(tag, 'inittemp', self.gain, float, 'T0')
            super(ThresholdSearch, self)._parse_xpl(tag, manager)

    def on_initialize(self):
        if not self.skip_thermal:
            self.thermal.initialize()
        self.electrical.initialize()
        self.diffusion.initialize()
        self.gain.initialize()
        self.optical.initialize()

    def on_invalidate(self):
        if not self.skip_thermal:
            self.thermal.invalidate()
        self.electrical.invalidate()
        self.diffusion.invalidate()
        self.optical.invalidate()

    def _optargs(self):
        return {}

    def get_lam(self):
        raise NotImplemented('get_lam')

    def compute_thermoelectric(self):
        """
        Perform thermo-electric calculations.

        This method may be called manually to perform thermo-electric calculations.
        Afterwards, one may investigate gain spectrum or verify settings of the optical
        solver.
        """
        self.initialize()
        if self.skip_thermal:
            self.electrical.compute()
        else:
            if self._invalidate:
                self.thermal.invalidate()
                self.electrical.invalidate()
                self.diffusion.invalidate()
                self.gain.invalidate()
            verr = 2. * self.electrical.maxerr
            terr = 2. * self.thermal.maxerr
            while terr > self.thermal.maxerr or verr > self.electrical.maxerr:
                verr = self.electrical.compute(self.tfreq)
                terr = self.thermal.compute(1)
        self.diffusion.compute_threshold()

    def step(self, volt):
        """
        Function performing one step of the threshold search.

        Args:
            volt (float): Voltage on a specified boundary condition [V].

        Returns:
            float: Loss of a specified mode
        """
        plask.print_log('detail', "ThresholdSearch: V = {0:.4f} V".format(volt))
        self.electrical.voltage_boundary[self.ivb].value = volt
        self.compute_thermoelectric()
        self.optical.invalidate()
        optstart = getattr(self, 'get_'+self._optarg)()
        if self._optarg is None:
            self.modeno = self.optical.find_mode(optstart, **self._optargs())
        else:
            _optargs = self._optargs().copy()
            _optargs[self._optarg] = optstart
            self.modeno = self.optical.find_mode(**_optargs)
        val = self.optical.modes[self.modeno].loss
        plask.print_log('result', "ThresholdSearch: V = {:.4f} V, loss = {:g} / cm".format(volt, val))
        return val

    def _quickstep(self, arg):
        """
        Function performing one step of the quick threshold search.

        Args:
            arg (array): Array containing voltage on a specified boundary condition [V] and wavelength.

        Returns:
            array: Imaginary and real part of a specified mode
        """
        volt, lam = arg * self._quickscale
        plask.print_log('detail', "ThresholdSearch: V = {0:.4f} V, lam = {1:g} nm".format(volt, lam))
        self.electrical.voltage_boundary[self.ivb].value = volt
        self.compute_thermoelectric()
        if self._optarg is None:
            det = self.optical.get_determinant(lam, **self._optargs())
        else:
            _optargs = self._optargs().copy()
            _optargs[self._optarg] = lam
            det = self.optical.get_determinant(**_optargs)
        plask.print_log('result', "ThresholdSearch: V = {0:.4f} V, lam = {1:g} nm, det = {2.real}{2.imag:+g}j"
                        .format(volt, lam, det))
        return np.array([det.real, det.imag])

    def get_optical_determinant(self, lam):
        """
        Function computing determinant of the optical solver.

        Args:
             lam (float or array): Wavelength to compute the determinant for [nm].

        Returns:
            float or array: Optical determinant.
        """
        self.compute_thermoelectric()
        if self._optarg is None:
            return self.optical.get_determinant(lam, **self._optargs())
        else:
            _optargs = self._optargs().copy()
            _optargs[self._optarg] = lam
            return self.optical.get_determinant(**_optargs)

    def _plot_in_junction(self, func, axis, bounds, kwargs, label):
        if axis is None: axis = self.optical.mesh
        i = 0
        for i, (lb, msh) in enumerate(self._get_levels(self.diffusion.geometry, axis, 'QW', 'QD', 'gain')):
            values = np.array(func(msh))
            if label is None:
                lab = "Junction {:s}".format(lb)
            elif isinstance(label, tuple) or isinstance(label, tuple):
                lab = label[i]
            else:
                lab = label
            plask.plot(msh.axis0, values, label=lab, **kwargs)
        if i > 1:
            plask.legend(loc='best')
        plask.xlabel(u"${}$ [\xb5m]".format(plask.config.axes[-2]))
        if bounds:
            self._plot_hbounds(self.optical)

    def plot_junction_concentration(self, bounds=True, interpolation='linear', label=None, **kwargs):
        """
        Plot carriers concentration at the active region.

        Args:
            bounds (bool): If *True* then the geometry objects boundaries are
                           plotted.

            interpolation (str): Interpolation used when retrieving current density.

            label (str or sequence): Label for each junction. It can be a sequence of
                                     consecutive labels for each junction, or a string
                                     in which case the same label is used for each
                                     junction. If omitted automatic label is generated.

            **kwargs: Keyword arguments passed to the plot function.
        """
        self._plot_in_junction(lambda msh: self.diffusion.outCarriersConcentration(msh, interpolation),
                               self.diffusion.mesh, bounds, kwargs, label)
        plask.ylabel(u"Carriers Concentration [1/cm\xb3]")
        plask.window_title("Carriers Concentration")

    def plot_junction_gain(self, axis=None, bounds=True, interpolation='linear', label=None, **kwargs):
        """
        Plot gain at the active region.

        Args:
            axis (mesh or sequence): Points along horizontal axis to plot gain at.
                                     Defaults to thr optical mesh.

            bounds (bool): If *True* then the geometry objects boundaries are
                           plotted.

            interpolation (str): Interpolation used when retrieving current density.

            label (str or sequence): Label for each junction. It can be a sequence of
                                     consecutive labels for each junction, or a string
                                     in which case the same label is used for each
                                     junction. If omitted automatic label is generated.

            **kwargs: Keyword arguments passed to the plot function.
        """
        lam = getattr(self.optical, self._lam0).real
        if lam is None: lam = self.get_lam().real
        self._plot_in_junction(lambda msh: self.gain.outGain(msh, lam, interpolation),
                               axis, bounds, kwargs, label)
        plask.ylabel(u"Gain [1/cm]")
        plask.window_title("Gain Profile")

    def plot_optical_determinant(self, lams, **kwargs):
        """
        Function plotting determinant of the optical solver.

        Args:
            lams (array): Wavelengths to plot the determinant for [nm].

            **kwargs: Keyword arguments passed to the plot function.
        """
        vals = self.get_optical_determinant(lams)
        plask.plot(lams, abs(vals))
        plask.yscale('log')
        plask.xlabel("Wavelength [nm]")
        plask.ylabel("Determinant [ar.u.]")

    def compute(self, save=True, invalidate=False, group='ThresholdSearch'):
        """
        Execute the algorithm.

        In the beginning the solvers are invalidated and next, the self-
        consistent loop of thermal, electrical, gain, and optical calculations
        are run within the root-finding algorithm until the mode is found
        with zero optical losses.

        Args:
            save (bool or str): If `True` the computed fields are saved to the
                HDF5 file named after the script name with the suffix denoting
                either the batch job id or the current time if no batch system
                is used. The filename can be overridden by setting this parameter
                as a string.

            invalidate (bool): If this flag is set, solvers are invalidated
                               in the beginning of the computations.

            group (str): HDF5 group to save the data under.

        Returns:
            The voltage set to ``ivolt`` boundary condition for the threshold.
            The threshold current can be then obtained by calling:

            >>> solver.get_total_current()
            123.0
        """

        if invalidate:
            if not self.skip_thermal:
                self.thermal.invalidate()
            self.electrical.invalidate()
            self.diffusion.invalidate()
            self.optical.invalidate()
        self._invalidate = invalidate
        self.initialize()

        if (self.vmin is None) != (self.vmax is None):
            raise ValueError("Both 'vmin' and 'vmax' must be either None or a float")
        if self.vmin is None:
            volt = self.electrical.voltage_boundary[self.ivb].value
            if self.quick:
                self.compute_thermoelectric()
                lam = self.get_lam().real
                self._quickscale = np.array([volt, lam])
                result = scipy.optimize.root(self._quickstep, np.array([1., 1.]),
                                             tol=1e-3 * min(self.vtol/volt, 1e-6),
                                             options=dict(maxfev=2*self.maxiter, eps=1e-6))
                if not result.status:
                    raise plask.ComputationError(result.message)
                self.threshold_voltage, lam = result.x * self._quickscale
                if self._optarg is None:
                    self.modeno = self.optical.set_mode(lam, **self._optargs())
                else:
                    _optargs = self._optargs().copy()
                    _optargs[self._optarg] = lam
                    self.modeno = self.optical.set_mode(**_optargs)
            else:
                self.threshold_voltage = scipy.optimize.newton(self.step, volt,
                                                               tol=self.vtol/volt, maxiter=self.maxiter)
        else:
            if self.quick:
                raise RuntimeError("Quick computation only allowed with a single starting point")
            self.threshold_voltage = scipy.optimize.brentq(self.step, self.vmin, self.vmax,
                                                           xtol=self.vtol * 2. / (self.vmin+self.vmax),
                                                           maxiter=self.maxiter, disp=True)

        self.threshold_current = abs(self.electrical.get_total_current())

        infolines = self._get_info()
        plask.print_log('important', "Threshold Search Finished")
        for line in infolines:
            plask.print_log('important', "  " + line)

        if save:
            filename = self.save(None if save is True else save)
            if filename.endswith('.h5'): filename = filename[:-3]
            plask.print_log('info', "Results saved to file '{}.txt'".format(filename))
            with open(filename+'.txt', 'a') as out:
                out.writelines(line + '\n' for line in infolines)
                out.write("\n")

        return self.threshold_voltage

    def _get_info(self):
        return self._get_defines_info() + [
            "Threshold voltage [V]:     {:8.3f}".format(self.threshold_voltage),
            "Threshold current [mA]:    {:8.3f}".format(self.threshold_current),
            "Maximum temperature [K]:   {:8.3f}".format(max(self.thermal.outTemperature(self.thermal.mesh)))
        ]

    def save(self, filename=None, group='ThresholdSearch', optical_resolution=None):
        """
        Save the computation results to the HDF5 file.

        Args:
            filename (str): The file name to save to.
                If omitted, the file name is generated automatically based on
                the script name with suffix denoting either the batch job id or
                the current time if no batch system is used.

            group (str): HDF5 group to save the data under.

            optical_resolution (tuple of ints): Number of points in horizontal and vertical directions
                for optical field.
        """
        if optical_resolution is None: optical_resolution = self.optical_resolution
        h5file, group, filename, close = h5open(filename, group)
        self._save_thermoelectric(h5file, group)
        levels = list(self._get_levels(self.diffusion.geometry, self.diffusion.mesh, 'QW', 'gain'))
        for no, mesh in levels:
            value = self.diffusion.outCarriersConcentration(mesh)
            plask.save_field(value, h5file, group + '/Junction'+no+'CarriersConcentration')
        for no, mesh in levels:
            lam = getattr(self.optical, self._lam0).real
            if lam is None: lam = self.get_lam().real
            value = self.gain.outGain(mesh, lam)
            plask.save_field(value, h5file, group + '/Junction'+no+'Gain')
        if self.modeno is not None:
            obox = self.optical.geometry.bbox
            oaxis = plask.mesh.Regular(obox.left, obox.right, optical_resolution[0])
            omesh = plask.mesh.Rectangular2D(oaxis,
                                             plask.mesh.Regular(obox.bottom, obox.top, optical_resolution[1]))
            ofield = self.optical.outLightMagnitude(self.modeno, omesh)
            plask.save_field(ofield/max(ofield), h5file, group + '/LightMagnitude')
            nrfield = self.optical.outRefractiveIndex(omesh)
            plask.save_field(nrfield, h5file, group + '/RefractiveIndex')
            rmesh = next(self._get_levels(self.optical.geometry, oaxis, 'QW', 'QD', 'gain'))[1]
            orfield = self.optical.outLightMagnitude(self.modeno, rmesh)
            plask.save_field(orfield/max(orfield), h5file, group + '/HorizontalLightMagnitude')

        if close:
            h5file.close()
        plask.print_log('info', "Fields saved to file '{}'".format(filename))
        return filename

    def plot_optical_field(self, resolution=None, geometry_color='0.75', geometry_alpha=0.35, **kwargs):
        """
        Plot computed optical mode field at threshold.

        Args:
            resolution (tuple of ints): Number of points in horizontal and vertical directions.

            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            **kwargs: Keyword arguments passed to the plot function.
        """
        if resolution is None: resolution = self.optical_resolution
        box = self.optical.geometry.bbox
        intensity_mesh = plask.mesh.Rectangular2D(plask.mesh.Regular(box.left, box.right, resolution[0]),
                                                  plask.mesh.Regular(box.bottom, box.top, resolution[1]))
        field = self.optical.outLightMagnitude(self.modeno, intensity_mesh)
        plask.plot_field(field, **kwargs)
        plask.plot_geometry(self.optical.geometry, color=geometry_color, alpha=geometry_alpha)
        plask.gcf().canvas.set_window_title("Light Intensity")

    def plot_optical_field_horizontal(self, resolution=None, bounds=True, interpolation='linear', **kwargs):
        """
        Plot horizontal distribution of the computed optical mode field at threshold.

        Args:
            resolution (int): Number of points in horizontal direction.

            bounds (bool): If *True* then the geometry objects boundaries are
                           plotted.

            interpolation (str): Interpolation used when retrieving current density.

            **kwargs: Keyword arguments passed to the plot function.
        """
        if resolution is None:
            resolution = self.optical_resolution[0]
        box = self.optical.geometry.bbox
        raxis = plask.mesh.Regular(box.left, box.right, resolution)
        rmesh = next(self._get_levels(self.optical.geometry, raxis, 'QW', 'QD', 'gain'))[1]
        field = self.optical.outLightMagnitude(self.modeno, rmesh, interpolation)
        color = kwargs.pop('color', None)
        if color is None:
            try:
                color = plask.rcParams['axes.prop_cycle']
            except KeyError:
                color = plask.rcParams['axes.color_cycle'][1]
            else:
                color = color.by_key()['color'][1]
        plask.plot_profile(field/max(field), color=color, **kwargs)
        if bounds:
            self._plot_hbounds(self.optical)
        plask.ylabel("Light Intensity [arb.u.]")
        plask.gcf().canvas.set_window_title("Radial Light Intensity")

    def plot_optical_field_vertical(self, pos=0.01, offset=0.5, resolution=None, interpolation='linear', **kwargs):
        """
        Plot vertical distribution of the computed optical mode field at threshold and
        refractive index profile.

        Args:
            resolution (int): Number of points in horizontal direction.

            pos (float): Horizontal position to get the field at.

            offset (float): Distance above and below geometry boundary to include into
                            the plot.

            interpolation (str): Interpolation used when retrieving current density.

            **kwargs: Keyword arguments passed to the plot function.
        """
        if resolution is None:
            resolution = self.optical_resolution[1]
        box = self.optical.geometry.bbox
        zaxis = plask.mesh.Regular(box.bottom-offset, box.top+offset, 10*resolution)
        zmesh = plask.mesh.Rectangular2D([pos], zaxis)
        try:
            cc = plask.rcParams['axes.prop_cycle']
        except KeyError:
            cc = plask.rcParams['axes.color_cycle']
        else:
            cc = cc.by_key()['color']
        color2 = cc[0]
        color = kwargs.pop('color', None)
        if color is None:
            color = cc[1]
        ax1 = plask.gca()
        field = self.optical.outLightMagnitude(self.modeno, zmesh, interpolation)
        plask.plot_profile(field/max(field), color=color, **kwargs)
        plask.ylabel("Light Intensity [arb.u.]")
        ax2 = plask.twinx()
        plask.plot_profile(self.optical.outRefractiveIndex(zmesh).real, comp=0, color=color2)
        plask.ylabel("Refractive Index")
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)
        ax2.patch.set_visible(True)
        plask.xlim(zmesh.axis1[0], zmesh.axis1[-1])
        plask.gcf().canvas.set_window_title("Vertical Light Intensity")


class ThresholdSearchCyl(ThresholdSearch):
    """
    Solver for threshold search of semiconductor laser.

    This solver performs thermo-electrical computations followed by
    determination ot threshold current and optical analysis in order to
    determine the threshold of a semiconductor laser. The search is
    performed by ``scipy`` root finding algorithm in order to determine
    the voltage and electric current ensuring no optical loss in the
    laser cavity.

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
    Diffusion = electrical.diffusion.DiffusionCyl
    Gain = gain.freecarrier.FreeCarrierCyl

    _OPTICAL_ROOTS = {'optical-root': 'root', 'optical-stripe-root': 'stripe_root'}

    outTemperature = property(lambda self: self.thermal.outTemperature, doc=Thermal.outTemperature.__doc__)
    outHeatFlux = property(lambda self: self.thermal.outHeatFlux, doc=Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=Thermal.outThermalConductivity.__doc__)
    outVoltage = property(lambda self: self.electrical.outVoltage, doc=Electrical.outVoltage.__doc__)
    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=Electrical.outCurrentDensity.__doc__)
    outHeat = property(lambda self: self.electrical.outHeat, doc=Electrical.outHeat.__doc__)
    outConductivity = property(lambda self: self.electrical.outConductivity, doc=Electrical.outConductivity.__doc__)
    outCarriersConcentration = property(lambda self: self.diffusion.outCarriersConcentration,
                                        doc=Diffusion.outCarriersConcentration.__doc__)
    outGain = property(lambda self: self.gain.outGain, doc=Gain.outGain.__doc__)
    outLightMagnitude = property(lambda self: self.optical.outLightMagnitude, doc=_doc.outLightMagnitude)
    outLoss = property(lambda self: self.optical.outLoss, doc=_doc.outLoss)
    outWavelength = property(lambda self: self.optical.outWavelength, doc=_doc.outWavelength)
    outRefractiveIndex = property(lambda self: self.optical.outRefractiveIndex, doc=_doc.outRefractiveIndex)
    outLightE = property(lambda self: self.optical.outLightE, doc=_doc.outLightE)

    thermal = attribute(Thermal.__name__+"()")
    ":class:`thermal.static.StaticCyl` solver used for thermal calculations."

    electrical = attribute(Electrical.__name__+"()")
    ":class:`electrical.shockley.ShockleyCyl` solver used for electrical calculations."

    diffusion = attribute(Diffusion.__name__+"()")
    ":class:`electrical.diffusion.DiffusionCyl` solver used for electrical calculations."

    gain = attribute(Gain.__name__+"()")
    ":class:`gain.freecarrier.FreeCarrierCyl` solver used for gain calculations."

    optical = attribute("EffectiveFrequencyCyl()")
    ":class:`optical.effective.EffectiveFrequencyCyl` solver used for optical calculations."

    tfreq = 6.0
    """
    Number of electrical iterations per single thermal step.

    As temperature tends to converge faster, it is reasonable to repeat thermal
    solution less frequently.
    """

    vmin = None
    """
    Minimum voltage to search threshold for.

    It should be below the threshold.
    """

    vmax = None
    """
    Maximum voltage to search threshold for.

    It should be above the threshold.
    """

    vtol = 1e-5
    "Tolerance on voltage in the root search."

    maxiter = 50
    "Maximum number of root finding iterations."

    maxlam = attribute("optical.lam0")
    "Maximum wavelength considered for the optical mode search."

    dlam = 0.02
    """
    Wavelength step.

    Step, by which the wavelength is swept while searching for the approximate mode.
    """

    lpm = 0
    """
    Angular mode number $m$.

    0 for LP0x, 1 for LP1x, etc.
    """

    lpn = 1
    """
    Radial mode number $n$.

    1 for LPx1, 2 for LPx2, etc.
    """

    optical_resolution = (800, 600)
    """
    Number of points along the horizontal and vertical axes for the saved
    and plotted optical field.
    """

    skip_thermal = False
    """
    Skip thermal computations.

    The structure is assumed to have a constant temperature.
    This can be used to look for the threshold under pulse laser operation.
    """

    def __init__(self, name=''):
        from optical.effective import EffectiveFrequencyCyl
        self.Optical = EffectiveFrequencyCyl
        super(ThresholdSearchCyl, self).__init__(name)
        self.maxlam = None

    # def on_initialize(self):
    #     super(ThresholdSearchCyl, self).on_initialize()

    def get_lam(self):
        """
        Get approximate wavelength for optical computations.

        This method returns approximate wavelength for optical computations.
        By default if browses the wavelength range starting from :attr:`maxlam`,
        decreasing it by :attr:`dlam` until radial mode :attr:`lpn` is found.

        You can override this method to use custom mode approximation.

        Example:
             >>> solver = ThresholdSearchCyl()
             >>> solver.get_lam = lambda: 980.
             >>> solver.compute()
        """

        lam = self.maxlam if self.maxlam is not None else self.optical.lam0
        n = 0
        prev = 0.
        decr = False
        while n < self.lpn and lam.real > 0.:
            curr = abs(self.optical.get_determinant(lam=lam, m=self.lpm))
            if decr and curr > prev:
                n += 1
            decr = curr < prev
            prev = curr
            lam -= self.dlam
        if n == self.lpn:
            return lam + 2. * self.dlam
        raise ValueError("Approximation of mode LP{0.lpm}{0.lpn} not found".format(self))

    def _optargs(self):
        return dict(m=self.lpm)

    def _parse_xpl(self, tag, manager):
        if tag == 'optical':
            self._read_attr(tag, 'lam0', self.optical, float)
            self._read_attr(tag, 'vlam', self.optical, float)
            self._read_attr(tag, 'vat', self.optical, float)
            maxlam = tag.get('maxlam')
            if maxlam is not None:
                try:
                    self.maxlam = complex(maxlam)
                except ValueError:
                    raise plask.XMLError("{}: attribute maxlam has illegal value '{}'".format(tag, maxlam))
            self.dlam = float(tag.get('dlam', self.dlam))
            self.lpm = int(tag.get('m', self.lpm))
            self.lpn = int(tag.get('n', self.lpn))
        else:
            if tag == 'optical-root':
                self._read_attr(tag, 'determinant', self.optical, str, 'determinant_mode')
            super(ThresholdSearchCyl, self)._parse_xpl(tag, manager)

    def get_vert_optical_determinant(self, vlam):
        """
        Function computing ‘vertical determinant’ of the optical solver.

        Args:
             vlam (float or array): ‘Vertical wavelength’ to compute the vertical
                                     determinant for [nm].

        Returns:
            float or array: Optical vertical determinant.
        """
        self.compute_thermoelectric()
        if self._optarg is None:
            return self.optical.get_determinant(vlam, **self._optargs())
        else:
            _optargs = self._optargs().copy()
            _optargs[self._optarg] = vlam
            return self.optical.get_determinant(**_optargs)

    def plot_vert_optical_determinant(self, vlams, **kwargs):
        """
        Function plotting ‘vertical determinant’ of the optical solver.

        Args:
            vlams (array): ‘Vertical wavelengths’ to plot the determinant for [nm].

            **kwargs: Keyword arguments passed to the plot function.
        """
        vals = self.get_vert_optical_determinant(vlams)
        plask.plot(vlams, abs(vals))
        plask.yscale('log')
        plask.xlabel("Vertical Wavelength [nm]")
        plask.ylabel("Determinant [ar.u.]")

    def _get_info(self):
        return super(ThresholdSearchCyl, self)._get_info() + [
            "LP{}{} mode wavelength [nm]: {:8.3f}".format(self.lpm, self.lpn, self.optical.modes[self.modeno].lam.real)
        ]


class ThresholdSearchBesselCyl(ThresholdSearch):
    """
    Solver for threshold search of semiconductor laser with vector optical solver.

    This solver performs thermo-electrical computations followed by
    determination ot threshold current and optical analysis in order to
    determine the threshold of a semiconductor laser. The search is
    performed by ``scipy`` root finding algorithm in order to determine
    the voltage and electric current ensuring no optical loss in the
    laser cavity.

    This solver uses vector optical solver :class:`~plask.optical.slab.BesselCyl`.

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
    Diffusion = electrical.diffusion.DiffusionCyl
    Gain = gain.freecarrier.FreeCarrierCyl

    _OPTICAL_ROOTS = {'optical-root': 'root'}

    outTemperature = property(lambda self: self.thermal.outTemperature, doc=Thermal.outTemperature.__doc__)
    outHeatFlux = property(lambda self: self.thermal.outHeatFlux, doc=Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=Thermal.outThermalConductivity.__doc__)
    outVoltage = property(lambda self: self.electrical.outVoltage, doc=Electrical.outVoltage.__doc__)
    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=Electrical.outCurrentDensity.__doc__)
    outHeat = property(lambda self: self.electrical.outHeat, doc=Electrical.outHeat.__doc__)
    outConductivity = property(lambda self: self.electrical.outConductivity, doc=Electrical.outConductivity.__doc__)
    outCarriersConcentration = property(lambda self: self.diffusion.outCarriersConcentration,
                                        doc=Diffusion.outCarriersConcentration.__doc__)
    outGain = property(lambda self: self.gain.outGain, doc=Gain.outGain.__doc__)
    outLightMagnitude = property(lambda self: self.optical.outLightMagnitude, doc=_doc.outLightMagnitude)
    outLoss = property(lambda self: self.optical.outLoss, doc=_doc.outLoss)
    outWavelength = property(lambda self: self.optical.outWavelength, doc=_doc.outWavelength)
    outRefractiveIndex = property(lambda self: self.optical.outRefractiveIndex, doc=_doc.outRefractiveIndex)
    outLightE = property(lambda self: self.optical.outLightE, doc=_doc.outLightE)

    thermal = attribute(Thermal.__name__ + "()")
    ":class:`thermal.static.StaticCyl` solver used for thermal calculations."

    electrical = attribute(Electrical.__name__ + "()")
    ":class:`electrical.shockley.ShockleyCyl` solver used for electrical calculations."

    diffusion = attribute(Diffusion.__name__ + "()")
    ":class:`electrical.diffusion.DiffusionCyl` solver used for electrical calculations."

    gain = attribute(Gain.__name__ + "()")
    ":class:`gain.freecarrier.FreeCarrierCyl` solver used for gain calculations."

    optical = attribute("BesselCyl()")
    ":class:`optical.slab.BesselFrequencyCyl` solver used for optical calculations."

    tfreq = 6.0
    """
    Number of electrical iterations per single thermal step.

    As temperature tends to converge faster, it is reasonable to repeat thermal
    solution less frequently.
    """

    vmin = None
    """
    Minimum voltage to search threshold for.

    It should be below the threshold.
    """

    vmax = None
    """
    Maximum voltage to search threshold for.

    It should be above the threshold.
    """

    vtol = 1e-5
    "Tolerance on voltage in the root search."

    maxiter = 50
    "Maximum number of root finding iterations."

    maxlam = attribute("optical.lam0")
    "Maximum wavelength considered for the optical mode search."

    dlam = 0.05
    """
    Wavelength step.

    Step, by which the wavelength is swept while searching for the approximate mode.
    """

    hem = 1
    """
      Angular mode number $m$.

      1 for HE1x, 2 for HE2x, etc.
    """

    hen = 1
    """
      Radial mode number $n$.

      1 for HEx1, 2 for HEx2, etc.
    """

    lam = None
    """
    Initial wavelength for optical search.

    If this value is set, the computations are started from this value. If this
    value is set, the radial mode number :attr:`hen` is ignored.

    Note that it is safer to leave this empty and allow the solver to look for it
    automatically, however, it may increase the time of optical computations.
    """

    optical_resolution = (800, 600)
    """
    Number of points along the horizontal and vertical axes for the saved
    and plotted optical field.
    """

    skip_thermal = False
    """
    Skip thermal computations.

    The structure is assumed to have a constant temperature.
    This can be used to look for the threshold under pulse laser operation.
    """

    def __init__(self, name=''):
        import optical.slab
        self.Optical = optical.slab.BesselCyl
        super(ThresholdSearchBesselCyl, self).__init__(name)
        self.maxlam = None

    # def on_initialize(self):
    #     super(ThresholdSearchBesselCyl, self).on_initialize()

    # def on_invalidate(self):
    #     super(ThresholdSearchBesselCyl, self).on_invalidate()

    def get_lam(self):
        """
        Get approximate wavelength for optical computations.

        This method returns approximate wavelength for optical computations.
        By default if browses the wavelength range starting from :attr:`maxlam`,
        decreasing it by :attr:`dlam` until radial mode :attr:`hen` is found.

        You can override this method to use custom mode approximation.

        Example:
             >>> solver = ThresholdSearchBesselCyl()
             >>> solver.get_lam = lambda: 980.
             >>> solver.compute()
        """

        if self.lam is not None:
            return self.lam

        lam = self.maxlam if self.maxlam is not None else self.optical.lam0
        n = 0
        prev = 0.
        decr = False
        while n < self.hen and lam.real > 0.:
            curr = abs(self.optical.get_determinant(lam=lam, m=self.hem))
            if decr and curr > prev:
                n += 1
            decr = curr < prev
            prev = curr
            lam -= self.dlam
        if n == self.hen:
            return lam + 2. * self.dlam
        raise ValueError("Approximation of mode HE{0.hem}{0.hen} not found".format(self))

    def _optargs(self):
        return dict(m=self.hem)

    def _parse_xpl(self, tag, manager):
        if tag == 'optical':
            self._read_attr(tag, 'lam0', self.optical, float)
            self._read_attr(tag, 'update-gain', self.optical, bool, 'update_gain')
            self._read_attr(tag, 'domain', self.optical, str, 'domain')
            self._read_attr(tag, 'size', self.optical, int, 'size')
            self._read_attr(tag, 'group-layers', self.optical, bool, 'group_layers')
            self._read_attr(tag, 'k-method', self.optical, str, 'kmethod')
            self._read_attr(tag, 'k-scale', self.optical, float, 'kscale')
            self._read_attr(tag, 'transfer', self.optical, str, 'transfer')
            maxlam = tag.get('maxlam')
            if maxlam is not None:
                try:
                    self.maxlam = complex(maxlam)
                except ValueError:
                    raise plask.XMLError("{}: attribute maxlam has illegal value '{}'".format(tag, maxlam))
            self.dlam = float(tag.get('dlam', self.dlam))
            self.lam = tag.get('lam', self.lam)
            if self.lam is not None: self.lam = complex(self.lam)
            self.hem = int(tag.get('m', self.hem))
            self.hen = int(tag.get('n', self.hen))
        elif tag == 'optical-interface':
            attrs = {key: val for (key, val) in ((key, tag.get(key)) for key in ('position', 'object', 'path'))
                     if val is not None}
            if len(attrs) > 1 and (len(attrs) > 2 or 'position' in attrs):
                raise plask.XMLError("{}: conflicting attributes '{}'".format(tag, "' and '".join(attrs.keys())))
            elif 'position' in attrs:
                self.optical.set_interface(attrs['position'])
            elif 'object' in attrs:
                path = attrs.get('path')
                if path is not None:
                    self.optical.set_interface(manager.geo[attrs['object']], manager.pth[path])
                else:
                    self.optical.set_interface(manager.geo[attrs['object']])
        elif tag == 'optical-vpml':
            self._read_attr(tag, 'factor', self.optical.vpml, complex, 'factor')
            self._read_attr(tag, 'dist', self.optical.vpml, float, 'dist')
            self._read_attr(tag, 'size', self.optical.vpml, float, 'size')
        elif tag == 'optical-pml':
            self._read_attr(tag, 'factor', self.optical.pml, complex, 'factor')
            self._read_attr(tag, 'shape', self.optical.pml, float, 'shape')
            self._read_attr(tag, 'dist', self.optical.pml, float, 'dist')
            self._read_attr(tag, 'size', self.optical.pml, float, 'size')
        else:
            super(ThresholdSearchBesselCyl, self)._parse_xpl(tag, manager)

    def _get_info(self):
        return super(ThresholdSearchBesselCyl, self)._get_info() + [
            "HE{}{} mode wavelength [nm]: {:8.3f}".format(self.hem, self.hen, self.optical.modes[self.modeno].lam.real)
        ]


class ThresholdSearch2D(ThresholdSearch):
    """
    Solver for threshold search of semiconductor laser.

    This solver performs thermo-electrical computations followed by
    determination ot threshold current and optical analysis in order to
    determine the threshold of a semiconductor laser. The search is
    performed by ``scipy`` root finding algorithm in order to determine
    the voltage and electric current ensuring no optical loss in the
    laser cavity.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`compute` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    _optarg = 'neff'
    _lam0 = 'wavelength'

    Thermal = thermal.static.Static2D
    Electrical = electrical.shockley.Shockley2D
    Diffusion = electrical.diffusion.Diffusion2D
    Gain = gain.freecarrier.FreeCarrier2D

    _OPTICAL_ROOTS = {'optical-root': 'root', 'optical-stripe-root': 'stripe_root'}

    outTemperature = property(lambda self: self.thermal.outTemperature, doc=Thermal.outTemperature.__doc__)
    outHeatFlux = property(lambda self: self.thermal.outHeatFlux, doc=Thermal.outHeatFlux.__doc__)

    outThermalConductivity = property(lambda self: self.thermal.outThermalConductivity,
                                      doc=Thermal.outThermalConductivity.__doc__)
    outVoltage = property(lambda self: self.electrical.outVoltage, doc=Electrical.outVoltage.__doc__)
    outCurrentDensity = property(lambda self: self.electrical.outCurrentDensity,
                                 doc=Electrical.outCurrentDensity.__doc__)
    outHeat = property(lambda self: self.electrical.outHeat, doc=Electrical.outHeat.__doc__)
    outConductivity = property(lambda self: self.electrical.outConductivity, doc=Electrical.outConductivity.__doc__)
    outCarriersConcentration = property(lambda self: self.diffusion.outCarriersConcentration,
                                        doc=Diffusion.outCarriersConcentration.__doc__)
    outGain = property(lambda self: self.gain.outGain, doc=Gain.outGain.__doc__)
    outLightMagnitude = property(lambda self: self.optical.outLightMagnitude, doc=_doc.outLightMagnitude)
    outLoss = property(lambda self: self.optical.outLoss, doc=_doc.outLoss)
    outNeff = property(lambda self: self.optical.outNeff, doc=_doc.outNeff)
    outRefractiveIndex = property(lambda self: self.optical.outRefractiveIndex, doc=_doc.outRefractiveIndex)
    outLightE = property(lambda self: self.optical.outLightE, doc=_doc.outLightE)

    thermal = attribute(Thermal.__name__+"()")
    ":class:`thermal.static.StaticCyl` solver used for thermal calculations."

    electrical = attribute(Electrical.__name__+"()")
    ":class:`electrical.shockley.ShockleyCyl` solver used for electrical calculations."

    diffusion = attribute(Diffusion.__name__+"()")
    ":class:`electrical.diffusion.DiffusionCyl` solver used for electrical calculations."

    gain = attribute(Gain.__name__+"()")
    ":class:`gain.freecarrier.FreeCarrierCyl` solver used for gain calculations."

    optical = attribute("EffectiveFrequencyCyl()")
    ":class:`optical.effective.EffectiveFrequencyCyl` solver used for optical calculations."

    tfreq = 6.0
    """
    Number of electrical iterations per single thermal step.

    As temperature tends to converge faster, it is reasonable to repeat thermal
    solution less frequently.
    """

    vmin = None
    """
    Minimum voltage to search threshold for.

    It should be below the threshold.
    """

    vmax = None
    """
    Maximum voltage to search threshold for.

    It should be above the threshold.
    """

    vtol = 1e-5
    "Tolerance on voltage in the root search."

    maxiter = 50
    "Maximum number of root finding iterations."

    wavelength = None
    "Emission wavelength [nm]."

    dneff = 0.02
    """
    Effective index step.

    Step, by which the effective index is swept while searching for the approximate mode.
    """

    mn = 1
    """
    Lateral mode number $n$.
    """

    optical_resolution = (800, 600)
    """
    Number of points along the horizontal and vertical axes for the saved
    and plotted optical field.
    """

    skip_thermal = False
    """
    Skip thermal computations.

    The structure is assumed to have a constant temperature.
    This can be used to look for the threshold under pulse laser operation.
    """

    def __init__(self, name=''):
        from optical.effective import EffectiveIndex2D
        self.Optical = EffectiveIndex2D
        super(ThresholdSearch2D, self).__init__(name)

    def on_initialize(self):
        super(ThresholdSearch2D, self).on_initialize()
        points = plask.mesh.Rectangular2D.SimpleGenerator()(self.optical.geometry).get_midpoints()
        self._maxneff = max(self.optical.geometry.get_material(point).Nr(self.optical.wavelength.real).real
                            for point in points)

    def get_neff(self):
        """
        Get approximate effective index for optical computations.

        This method returns approximate wavelength for optical computations.
        By default if browses the wavelength range starting from :attr:`maxneff`,
        decreasing it by :attr:`dneff` until lateral mode :attr:`mn` is found.

        You can override this method to use custom mode approximation.

        Example:
             >>> solver = ThresholdSearch2D()
             >>> solver.get_neff = lambda: 3.5
             >>> solver.compute()
        """

        neff = self._maxneff
        n = 0
        prev = 0.
        decr = False
        while n < self.mn and neff.real > 0.:
            curr = abs(self.optical.get_determinant(neff))
            if decr and curr > prev:
                n += 1
            decr = curr < prev
            prev = curr
            neff -= self.dneff
        if n == self.mn:
            return neff + 2. * self.dneff
        raise ValueError("Mode approximation not found")

    def _parse_xpl(self, tag, manager):
        if tag == 'optical':
            self._read_attr(tag, 'wavelength', self.optical, float)
            self._read_attr(tag, 'polarization', self.optical, float)
            self._read_attr(tag, 'vneff', self.optical, float)
            self._read_attr(tag, 'vat', self.optical, float)
            self.dneff = float(tag.get('dneff', self.dneff))
            self.mn = int(tag.get('mn', self.mn))
        else:
            if tag == 'optical-root':
                self._read_attr(tag, 'determinant', self.optical, str)
            super(ThresholdSearch2D, self)._parse_xpl(tag, manager)

    def get_optical_determinant(self, neff):
        """
        Function computing determinant of the optical solver.

        Args:
             neff (float or array): Effective index to compute the determinant for.

        Returns:
            float or array: Optical determinant.
        """
        self.compute_thermoelectric()
        if self._optarg is None:
            return self.optical.get_determinant(neff, **self._optargs())
        else:
            _optargs = self._optargs().copy()
            _optargs[self._optarg] = neff
            return self.optical.get_determinant(**_optargs)

    def plot_optical_determinant(self, neffs, **kwargs):
        """
        Function plotting determinant of the optical solver.

        Args:
            neffs (array): Array of effective indices to plot the determinant for.

            **kwargs: Keyword arguments passed to the plot function.
        """
        vals = self.get_optical_determinant(neffs)
        plask.plot(neffs, abs(vals))
        plask.yscale('log')
        plask.xlabel("Effective Index")
        plask.ylabel("Determinant [ar.u.]")

    def get_vert_optical_determinant(self, vneff):
        """
        Function computing ‘vertical determinant’ of the optical solver.

        Args:
             vneff (float or array): Effective index to compute the vertical
                                     determinant for.

        Returns:
            float or array: Optical determinant.
        """
        self.compute_thermoelectric()
        if self._optarg is None:
            return self.optical.get_determinant(vneff, **self._optargs())
        else:
            _optargs = self._optargs().copy()
            _optargs[self._optarg] = vneff
            return self.optical.get_determinant(**_optargs)

    def plot_vert_optical_determinant(self, vneffs, **kwargs):
        """
        Function plotting ‘vertical determinant’ of the optical solver.

        Args:
            vneffs (array): Array of effective indices to plot the vertical
                            determinant for.

            **kwargs: Keyword arguments passed to the plot function.
        """
        vals = self.get_vert_optical_determinant(vneffs)
        plask.plot(vneffs, abs(vals))
        plask.yscale('log')
        plask.xlabel("Vertical Effective Index")
        plask.ylabel("Determinant [ar.u.]")

    def _get_info(self):
        return super(ThresholdSearch2D, self)._get_info() + [
            "Effective index: {:8.3f}".format(self.optical.modes[self.modeno].neff.real)
        ]


__all__ = 'ThresholdSearchCyl', 'ThresholdSearchBesselCyl', 'ThresholdSearch2D'
