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

from ...qt.QtCore import *

import plask
from ...utils.config import CONFIG
from ...utils.matplotlib import PlotWidgetBase, BwColor


class PlotWidget(PlotWidgetBase):

    class NavigationToolbar(PlotWidgetBase.NavigationToolbar):

        toolitems = (
            ('Geometry:', 'Select geometry for mesh preview', None, 'select_geometry', ((), 0)),
            (None, None, None, None, None),
            ('Plot', 'Plot mesh preview', 'draw-brush', 'plot', None),
            ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', False),
            (None, None, None, None, None),
            ('Home', 'Reset original view', 'go-home', 'home', None),
            ('Back', 'Back to  previous view', 'go-previous', 'back', None),
            ('Forward', 'Forward to next view', 'go-next', 'forward', None),
            (None, None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'transform-move', 'pan', False),
            ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False),
            (None, None, None, None, None),
            ('Aspect', 'Set equal aspect ratio for both axes', 'system-lock-screen', 'aspect', False),
            (None, None, None, None, None),
            ('Plane:', 'Select longitudinal-transverse plane', None, 'select_plane',
             (('tran-long', 'long-vert', 'tran-vert'), 2)),
        )

        def __init__(self, *args, **kwargs):
            super(PlotWidget.NavigationToolbar, self).__init__(*args, **kwargs)
            self._actions['plot'].setShortcut(Qt.ALT + Qt.Key_P)
            self.disable_planes(('long','tran','vert'))

        def select_geometry(self, *args):
            # if self.controller._current_controller is not None:
            #     self.controller._current_controller.geometry_name = self.widgets['select_geometry'].currentText()
            if self.controller.plot_auto_refresh:
                self.controller.plot()
            else:
                self.controller.show_update_required()

        def home(self):
            if self.controller.plotted_geometry is not None:
                box = self.controller.plotted_geometry.bbox
                self.parent.zoom_bbox(box)

        def select_plane(self, index):
            super(PlotWidget.NavigationToolbar, self).select_plane(index)
            if self.controller.plot_auto_refresh: self.controller.plot()
            else: self.controller.show_update_required()

    def __init__(self, controller=None, parent=None):
        super(PlotWidget, self).__init__(controller, parent)
        self.get_color = BwColor(self.axes)
        # self.layout().setContentsMargins(0, 9, 6, 2)

    def update_plot(self, mesh, geometry, set_limits, plane='12'):
        updater = self.plot_updater(set_limits, plane)
        margin = next(updater)
        if mesh is not None:
            plask.plot_mesh(axes=self.axes, mesh=mesh, margin=margin, zorder=1.5, plane=plane,
                            lw=CONFIG['mesh/line_width'], color=CONFIG['mesh/mesh_color'])
            try:
                if geometry is not None:
                    plask.plot_geometry(axes=self.axes, geometry=geometry, fill=True, margin=margin, zorder=1,
                                    plane=plane, lw=1.5, get_color=self.get_color)
            finally:
                try: next(updater)
                except StopIteration: pass