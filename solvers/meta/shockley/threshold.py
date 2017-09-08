# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology

import electrical.diffusion
import electrical.shockley
import gain.freecarrier
import thermal.static

import plask

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

    tfreq = 6.0
    vmin = None
    vmax = None
    optical_resolution = (800, 600)
    vtol = 1e-5
    quick = False
    maxiter = 50

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

    def _parse_xpl(self, tag, manager):
        if tag == 'root':
            self.vmin = tag.get('vmin', self.vmin)
            self.vmax = tag.get('vmax', self.vmax)
            self.ivb = tag['bcond']
            self.vtol = tag.get('vtol', self.vtol)
            self.maxiter = tag.get('maxiter', self.maxiter)
            # self.quick = tag.get('quick', self.quick)
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
        elif tag in self._OPTICAL_ROOTS:
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
                self.diffusion.mesh = tag.getitem(manager.msh, 'diffusion')
                if 'optical' in tag: self.optical.mesh = tag.getitem(manager.msh, 'optical')
                if 'gain' in tag: self.gain.mesh = tag.getitem(manager.msh, 'gain')
            elif tag == 'loop':
                self._read_attr(tag, 'inittemp', self.gain, float, 'T0')
            super(ThresholdSearch, self)._parse_xpl(tag, manager)

    def on_initialize(self):
        super(ThresholdSearch, self).on_initialize()
        self.diffusion.initialize()
        self.gain.initialize()
        self.optical.initialize()
        # if quick:
        #     raise NotImplemented('Quick threshold search not implemented')

    def on_invalidate(self):
        super(ThresholdSearch, self).on_invalidate()
        self.diffusion.invalidate()
        self.optical.invalidate()

    def __optargs(self):
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
        self.electrical.voltage_boundary[self.ivb].value = volt
        self.compute_thermoelectric()
        self.optical.invalidate()
        optstart = self.get_lam()
        if self._optarg is None:
            self.modeno = self.optical.find_mode(optstart, **self.__optargs())
        else:
            _optargs = self.__optargs().copy()
            _optargs[self._optarg] = optstart
            self.modeno = self.optical.find_mode(**_optargs)
        val = self.optical.modes[self.modeno].loss
        plask.print_log('result', "ThresholdSearch: V = {:.4f} V, loss = {:g} / cm".format(volt, val))
        return val

    def optical_determinant(self, lam):
        """
        Function computing determinant of the optical solver.

        Args:
             lam (float of array): Wavelength to compute the determinant for [V].

        Returns:
            float or array: Optical determinant.
        """
        self.compute_thermoelectric()
        if self._optarg is None:
            return self.optical.get_determinant(lam, **self.__optargs())
        else:
            _optargs = self.__optargs().copy()
            _optargs[self._optarg] = lam
            return self.optical.get_determinant(**_optargs)

    def compute(self, save=True, invalidate=False):
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

        Returns:
            The voltage set to ``ivolt`` boundary condition for the threshold.
            The threshold current can be then obtained by calling:

            >>> solver.get_total_current()
            123.0
        """

        if invalidate:
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
            self.threshold_voltage = scipy.optimize.newton(self.step, volt, tol=self.vtol, maxiter=self.maxiter)
        else:
            self.threshold_voltage = scipy.optimize.brentq(self.step, self.vmin, self.vmax, xtol=self.vtol,
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
        try:
            import __main__
            defines = ["  {} = {}".format(key, __main__.DEF[key]) for key in __main__.__overrites__]
        except (NameError, KeyError):
            defines = []
        if defines:
            defines = ["Temporary defines:"] + defines
        return defines + [
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
        h5file, group, filename = h5open(filename, group)
        self._save_thermoelectric(h5file, group)
        levels = list(self._get_levels(self.diffusion.geometry, self.diffusion.mesh, 'QW', 'gain'))
        for no, mesh in levels:
            value = self.diffusion.outCarriersConcentration(mesh)
            plask.save_field(value, h5file, group + '/Junction'+no+'CarriersConcentration')
        for no, mesh in levels:
            value = self.gain.outGain(mesh, self.optical.lam0.real)
            plask.save_field(value, h5file, group + '/Junction'+no+'Gain')
        obox = self.optical.geometry.bbox
        omesh = plask.mesh.Rectangular2D(plask.mesh.Regular(obox.left, obox.right, optical_resolution[0]),
                                         plask.mesh.Regular(obox.bottom, obox.top, optical_resolution[1]))
        ofield = self.optical.outLightMagnitude(self.modeno, omesh)
        plask.save_field(ofield, h5file, group + '/LightMagnitude')
        nrfield = self.optical.outRefractiveIndex(omesh)
        plask.save_field(nrfield, h5file, group + '/RefractiveIndex')
        h5file.close()
        plask.print_log('info', "Fields saved to file '{}'".format(filename))
        return filename

    def plot_optical_field(self, resolution=(800, 600), geometry_color='0.75', geometry_alpha=0.35, **kwargs):
        """
        Plot computed optical mode field at threshold.

        Args:
            resolution (tuple of ints): Number of points in horizontal and vertical directions.

            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            kwargs: Keyword arguments passed to the plot function.
        """
        if resolution is None: resolution = self.optical_resolution
        box = self.optical.geometry.bbox
        intensity_mesh = plask.mesh.Rectangular2D(plask.mesh.Regular(box.left, box.right, resolution[0]),
                                                  plask.mesh.Regular(box.bottom, box.top, resolution[1]))
        field = self.optical.outLightMagnitude(self.modeno, intensity_mesh)
        plask.plot_field(field, **kwargs)
        plask.plot_geometry(self.optical.geometry, color=geometry_color, alpha=geometry_alpha)
        plask.gcf().canvas.set_window_title("Light Intensity")


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
    outElectricField = property(lambda self: self.optical.outElectricField, doc=_doc.outElectricField)

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
      
    Step, by which the wavelength is sweep while searching for the approximate mode.
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

    def __optargs(self):
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
            super(ThresholdSearchCyl, self)._parse_xpl(tag, manager)

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
    outElectricField = property(lambda self: self.optical.outElectricField, doc=_doc.outElectricField)

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

    Step, by which the wavelength is sweep while searching for the approximate mode.
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

    def __optargs(self):
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
            self.maxlam = complex(tag.get('maxlam', self.maxlam))
            self.dlam = float(tag.get('dlam', self.dlam))
            self.lam = tag.get('lam', self.lam)
            if self.lam is not None: self.lam = complex(self.lam)
            self.hem = int(tag.get('m', self.hem))
            self.hen = int(tag.get('n', self.hen))
        elif tag == 'optical-interface':
            attrs = {key: val for (key, val) in ((key, tag.get(key)) for key in ('index', 'position', 'object', 'path'))
                     if val is not None}
            if len(attrs) > 1 and (len(attrs) > 2 or 'index' in attrs or 'position' in attrs):
                raise plask.XMLError("{}: conflicting attributes '{}'".format(tag, "' and '".join(attrs.keys())))
            if 'index' in attrs:
                self.optical.interface = attrs['index']
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


__all__ = 'ThresholdSearchCyl', 'ThresholdSearchBesselCyl'
