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

from matplotlib.ticker import MaxNLocator
import plask

from ..geometry.plot_widget import PlotWidget as GeometryPlotWidget, NavigationToolbar as GeometryNavigationToolbar


class PlotWidget(GeometryPlotWidget):

    def __init__(self, controller=None, parent=None, picker=None):
        super(PlotWidget, self).__init__(controller, parent, picker)

    def update_plot(self, toplot, set_limits, plane='12'):
        # self.figure.clear()
        self.axes.cla()
        self.selectors = []
        self.axes.minorticks_on()
        if toplot is not None:
            xlim, ylim = self.axes.get_xlim(), self.axes.set_ylim()
            self.axes.grid(which='major', ls='-', lw=1, alpha=0.4, color='0.5')
            self.axes.grid(which='minor', ls='-', lw=1, alpha=0.1, color='0.5')
            self.axes.axhline(0., ls='-', color='k', alpha=0.4, zorder=3)
            self.axes.axvline(0., ls='-', color='k', alpha=0.4, zorder=3)
            margin = 0.1 if set_limits else None

            plask.plot_mesh(axes=self.axes, mesh=toplot, margin=margin, zorder=1.5, plane=plane, lw=1.2,
                            color="#00aa00")

            for ax in self.axes.xaxis, self.axes.yaxis:
                ax.set_major_locator(MaxNLocator(nbins=10, steps=(1, 10)))
                ax.set_minor_locator(MaxNLocator(nbins=100, steps=(1, 10)))
            if not set_limits:
                self.axes.set_xlim(xlim)
                self.axes.set_ylim(ylim)
            self.axes.set_aspect('equal' if self.aspect_locked else 'auto')
            self.canvas.draw()
            self.plane = plane


