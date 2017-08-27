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

try:
    import scipy
except ImportError:
    plask.print_log(plask.LOG_WARNING, "scipy could not be imported."
                                       " You will not be able to run some algorithms."
                                       " Install scipy to resolve this issue.")
else:
    import scipy.optimize

from .thermoelectric import suffix, h5open, ThermoElectric


class ThresholdSearch(ThermoElectric):
    """
    Solver for threshold search of semiconductor laser.

    This solver performs thermo-electrical computations followed by
    determination ot threshold current and optical analysis in order to
    determine the threshold of a semiconductor laser. The search is
    performed by scipy root finding algorithm in order to determine
    the volatage and electric current ensuring no optical loss in the
    laser cavity.

    The computations can be executed using `compute` method, after which
    the results may be save to the HDF5 file with `save` or presented visually
    using ``plot_...`` methods. If ``save`` parameter of the :meth:`run` method
    is *True* the fields are saved automatically after the computations.
    The file name is based on the name of the executed script with suffix denoting
    either the launch time or the identifier of a batch job if a batch system
    (like SLURM, OpenPBS, or SGE) is used.
    """

    def __init__(self, thermal, electrical, diffusion, gain, optical,
                 ivolt, vstart, optstart, optname=None, optargs=None, loss='auto', tfreq=6,
                 vtol=1e-4, invalidate=False, quick=False, connect=False):
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
        self.vstart = vstart
        self.ivb = ivolt
        self.vtol = vtol
        self.optstart = optstart
        self.optname = optname
        self.optargs = optargs if optargs is not None else {}
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

    def run(self, save=True, noinit=False):
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
                is used. The filename can be overriden by setting this parameted
                as a string.
            noinit (bool): If this flas is set, solvers are not invalidated
                           in the beginning of the computations.

        Returns:
            The voltage set to ``ivolt`` boundary condition for the threshold.
            The threshold current can be then obtained by calling:

            >>> electrical.get_total_current()
            123.0
        """

        if not noinit:
            self.thermal.invalidate()
            self.electrical.invalidate()
            self.diffusion.invalidate()
            self.optical.invalidate()

        def func(volt):
            """Function to search zero of"""
            self.electrical.voltage_boundary[self.ivb].value = volt
            if self.invalidate:
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
            optstart = self.optstart() if isinstance(self.optstart, _collections.Callable) else self.optstart
            if self.optname is None:
                self.modeno = self.optical.find_mode(optstart, **self.optargs)
            else:
                okwargs = {self.optname: optstart}
                okwargs.update(self.optargs)
                self.modeno = self.optical.find_mode(**okwargs)
            val = self.optical.modes[self.modeno].__getattribute__(self.loss)
            if self.loss != 'loss':
                if self.loss == 'wavelength':
                    val = (4e7*_np.pi / val).imag
                else:
                    val = val.imag
            plask.print_log('result', "ThresholdSearch: V = {:.4f} V, loss = {:g}{}"
                            .format(volt, val, '' if self.loss == 'neff' else ' / cm'))
            return val

        try:
            vmin, vmax = self.vstart
        except TypeError:
            result = scipy.optimize.newton(func, self.vstart, tol=self.vtol)
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
        h5file, group = _h5_open(filename, group)
        ThermoElectric.save(self, h5file, group)

    def plot_optical_field(self, hpoints=800, vpoints=600, geometry_color=(0.75, 0.75, 0.75, 0.35), **kwargs):
        """
        Plot computed optical mode field at threshold.

        Args:
            hpoints (int): Number of points to plot in vertical plot direction.

            vpoints (int): Number of points to plot in horizontal plot direction.

            kwargs: Keyword arguments passed to the plot function.
        """

        box = self.optical.geometry.bbox
        intensity_mesh = plask.mesh.Rectangular2D(plask.mesh.Regular(box.left, box.right, hpoints),
                                                  plask.mesh.Regular(box.bottom, box.top, vpoints))
        field = self.optical.outLightMagnitude(self.modeno, intensity_mesh)
        plask.plot_field(field)
        plask.plot_geometry(self.optical.geometry, color=geometry_color)
        plask.gcf().canvas.set_window_title("Light Intensity")
