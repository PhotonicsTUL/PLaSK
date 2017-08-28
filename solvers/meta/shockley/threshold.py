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

import collections

import plask
from plask import *

import thermal.static
import electrical.shockley
import electrical.diffusion
import gain.freecarrier
import optical.effective

try:
    import scipy
except ImportError:
    plask.print_log(plask.LOG_WARNING, "scipy could not be imported."
                                       " You will not be able to run some algorithms."
                                       " Install scipy to resolve this issue.")
else:
    import scipy.optimize

from .thermoelectric import attribute, suffix, h5open, ThermoElectric


class ThresholdSearch(ThermoElectric):

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
        self.optname = None
        self.optstart = None
        self.optargs = {}
        self.vtol = 1e-5
        self.quick = False
        self.threshold_current = None

    def _parse_xpl(self, tag, manager):
        if tag == 'root':
            self.ivb = tag['bcond']
            self.vtol = tag.get('vtol', self.vtol)
            # self.quick = tag.get('quick', self.quick)
        elif tag == 'diffusion':
            self._read_attr(tag, 'fem-method', self.diffusion, 'fem_method')
            self._read_attr(tag, 'accuracy', self.diffusion)
            self._read_attr(tag, 'abs-accuracy', self.diffusion, 'abs_accuracy')
            self._read_attr(tag, 'maxiters', self.diffusion)
            self._read_attr(tag, 'maxrefines', self.diffusion)
            self._read_attr(tag, 'interpolation', self.diffusion)
        elif tag == 'gain':
            self._read_attr(tag, 'lifetime', self.gain)
            self._read_attr(tag, 'matrix-elem', self.gain, 'matrix_element')
            self._read_attr(tag, 'strained', self.gain)
        elif tag in ('optical-root', 'optical-stripe-root'):
            name = {'optical-root': 'root', 'optical-stripe-root': 'stripe_root'}[tag.name]
            root = getattr(self.optical, name)
            self._read_attr(tag, 'method', root)
            self._read_attr(tag, 'tolx', root)
            self._read_attr(tag, 'tolf-min', root, 'tolf_min')
            self._read_attr(tag, 'tolf-max', root, 'tolf_max')
            self._read_attr(tag, 'maxstep', root)
            self._read_attr(tag, 'maxiter', root)
            self._read_attr(tag, 'alpha', root)
            self._read_attr(tag, 'lambda', root)
            self._read_attr(tag, 'initial-range', root, 'initial_range')
        else:
            if tag == 'geometry':
                self.diffusion.geometry = self.gain.geometry = manager.geo[tag['electrical']]
                self.optical.geometry = manager.geo[tag['optical']]
            elif tag == 'mesh':
                self.diffusion.mesh = manager.msh[tag['diffusion']]
                if 'optical' in tag: self.optical.mesh = manager.msh[tag['optical']]
                if 'gain' in tag: self.gain.mesh = manager.msh[tag['gain']]
            elif tag == 'loop':
                self._read_attr(tag, 'inittemp', self.gain, 'T0')
            super(ThresholdSearch, self)._parse_xpl(tag, manager)

    def on_initialize(self):
        super(ThresholdSearch, self).on_initialize()
        # if quick:
        #     raise NotImplemented('Quick threshold search not implemented')

    def on_invalidate(self):
        super(ThresholdSearch, self).on_invalidate()
        self.diffusion.invalidate()
        self.optical.invalidate()

    def compute(self, start, save=True, invalidate=True, **kwargs):
        """
        Execute the algorithm.

        In the beginning the solvers are invalidated and next, the self-
        consistent loop of thermal, electrical, gain, and optical calculations
        are run within the root-finding algorithm until the mode is found
        with zero optical losses.

        Args:
            start (float or tuple of 2 floats): Voltage staring point or voltage
                limits for the threshold search.
            save (bool or str): If `True` the computed fields are saved to the
                HDF5 file named after the script name with the suffix denoting
                either the batch job id or the current time if no batch system
                is used. The filename can be overridden by setting this parameter
                as a string.
            invalidate (bool): If this flag is set, solvers are invalidated
                               in the beginning of the computations.
            **optargs (dict): Other arguments for optical solver's ``find_mode``
                              and ``get_determinant`` methods.

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

        self.initialize()

        optargs = self.optargs.copy()
        optargs.update(kwargs)

        def func(volt):
            """Function to search zero of"""
            self.electrical.voltage_boundary[self.ivb].value = volt
            if invalidate:
                self.thermal.invalidate()
                self.electrical.invalidate()
                self.diffusion.invalidate()
                self.gain.invalidate()
                self.optical.invalidate()
            verr = 2. * self.electrical.maxerr
            terr = 2. * self.thermal.maxerr
            while terr > self.thermal.maxerr or verr > self.electrical.maxerr:
                verr = self.electrical.compute(self.tfreq)
                terr = self.thermal.compute(1)
            try: self.diffusion.compute_threshold()
            except AttributeError: pass
            optstart = self.optstart() if isinstance(self.optstart, collections.Callable) else self.optstart
            if self.optname is None:
                self.modeno = self.optical.find_mode(optstart, **optargs)
            else:
                optargs[self.optname] = optstart
                self.modeno = self.optical.find_mode(**optargs)
            val = self.optical.modes[self.modeno].loss
            plask.print_log('result', "ThresholdSearch: V = {:.4f} V, loss = {:g} / cm".format(volt, val))
            return val

        try:
            vmin, vmax = start
        except TypeError:
            result = scipy.optimize.newton(func, start, tol=self.vtol)
        else:
            result = scipy.optimize.brentq(func, vmin, vmax, xtol=self.vtol)
        self.threshold_current = self.electrical.get_total_current()

        if save:
            self.save(None if save is True else save)

        return result

    def save(self, filename=None, group='ThresholdSearch'):
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
        super(ThresholdSearch, self).save(h5file, group)

    def plot_optical_field(self, hpoints=800, vpoints=600, geometry_color='0.75', geometry_alpha=0.35, **kwargs):
        """
        Plot computed optical mode field at threshold.

        Args:
            hpoints (int): Number of points to plot in vertical plot direction.

            vpoints (int): Number of points to plot in horizontal plot direction.

            geometry_color (str or ``None``): Matplotlib color specification
                for the geometry. If ``None``, structure is not plotted.

            geometry_alpha (float): Geometry opacity (1 — fully opaque, 0 – invisible).

            kwargs: Keyword arguments passed to the plot function.
        """

        box = self.optical.geometry.bbox
        intensity_mesh = plask.mesh.Rectangular2D(plask.mesh.Regular(box.left, box.right, hpoints),
                                                  plask.mesh.Regular(box.bottom, box.top, vpoints))
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
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`run` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    Thermal = thermal.static.StaticCyl
    Electrical = electrical.shockley.ShockleyCyl
    Diffusion = electrical.diffusion.DiffusionCyl
    Gain = gain.freecarrier.FreeCarrierCyl
    Optical = optical.effective.EffectiveFrequencyCyl

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

    def __init__(self, name=''):
        super(ThresholdSearchCyl, self).__init__(name)
        self.optname = 'lam'

    def on_initialize(self):
        super(ThresholdSearchCyl, self).on_initialize()
        self.optical.lam0 = self.optstart() if isinstance(self.optstart, collections.Callable) else self.optstart

    def _parse_xpl(self, tag, manager):
        if tag == 'optical':
            self.optstart = tag['start']
            if 'm' in tag: self.optargs['m'] = tag['m']
            self._read_attr(tag, 'vlam', self.optical)
            self._read_attr(tag, 'vat', self.optical)
        else:
            super(ThresholdSearchCyl, self)._parse_xpl(tag, manager)


__all__ = 'ThresholdSearchCyl',
