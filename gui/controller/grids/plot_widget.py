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


from matplotlib.ticker import MaxNLocator

from ...qt.QtCore import *
from ...qt.QtWidgets import *

import plask
from ...utils.config import CONFIG
from ...utils.matplotlib import PlotWidgetBase, BwColor


class PlotWidget(PlotWidgetBase):
    class NavigationToolbar(PlotWidgetBase.NavigationToolbar):

        toolitems = (
            ('Geometry:', 'Select geometry for mesh preview', None, 'select_geometry', ((), 0), 'plot_geometry'),
            (None, None, None, None, None, None),
            ('Plot', 'Plot mesh preview', 'draw-brush', 'plot', None, 'plot_plot'),
            ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', False, 'plot_refresh'),
            (None, None, None, None, None, None),
            ('Home', 'Zoom to whole geometry', 'go-home', 'home', None, 'plot_home'),
            ('Back', 'Back to previous view', 'go-previous', 'back', None, 'plot_back'),
            ('Forward', 'Forward to next view', 'go-next', 'forward', None, 'plot_forward'),
            (None, None, None, None, None, None),
            ('Save', 'Save the figure', 'document-save', 'save_figure', None, 'plot_save'),
            (None, None, None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'transform-move', 'pan', False, 'plot_pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False, 'plot_zoom'),
            (None, None, None, None, None, None),
            ('Aspect', 'Set equal aspect ratio for both axes', 'system-lock-screen', 'aspect', False, 'plot_aspect'),
            (None, None, None, None, None, None),
            ('Plane:', 'Select longitudinal-transverse plane', None, 'select_plane',
             (('tran-long', 'long-vert', 'tran-vert'), 2), 'plot_plane'),
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.disable_planes(('long', 'tran', 'vert'))

        def select_geometry(self, *args):
            if self.controller._current_controller is not None:
                geometry_name = self.widgets['select_geometry'].currentText()
                self.controller._current_controller.model.geometry_name = None  # to force update
                try:
                    dim = max(self.controller._current_controller.model.dim, 2)
                except AttributeError:
                    pass
                else:
                    if dim == 3:
                        self.enable_planes(self.controller.geometry_axes_names.get(geometry_name, ('long', 'tran', 'vert')))
                    else:
                        self.disable_planes(self.controller.geometry_axes_names.get(geometry_name, ('long', 'tran', 'vert')))
            self.controller.update_current_mesh()
            if self.controller.plot_auto_refresh:
                self.controller.plot()
            else:
                self.controller.show_update_required()

        def home(self):
            if self.controller.current_geometry is not None:
                box = self.controller.current_geometry.bbox
                self.parent().zoom_bbox(box)

        def select_plane(self, index):
            super().select_plane(index)
            self.controller.need_reset_plot = True
            if self.controller.plot_auto_refresh: self.controller.plot()
            else: self.controller.show_update_required()

    def __init__(self, controller=None, parent=None):
        super().__init__(controller, parent)

        self.info = QLabel()
        self.info.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed))
        layout = self.layout()
        layout.addWidget(self.info)
        self.info.setVisible(False)

        colors = CONFIG['geometry/material_colors'].copy()
        self.get_color = BwColor(colors, self.axes)
        # self.layout().setContentsMargins(0, 9, 6, 2)
        self.clear()

    def update_plot(self, mesh, geometry, set_limits, plane='12'):
        updater = self.plot_updater(set_limits, plane)
        margin = next(updater)
        if mesh is not None:
            plask.plot_mesh(
                axes=self.axes,
                mesh=mesh,
                margin=margin,
                zorder=1.5,
                plane=plane,
                lw=CONFIG['mesh/line_width'],
                color=CONFIG['mesh/mesh_color']
            )
            try:
                if geometry is not None:
                    plask.plot_geometry(
                        axes=self.axes,
                        geometry=geometry,
                        fill=True,
                        margin=margin,
                        zorder=1,
                        plane=plane,
                        lw=1.5,
                        get_color=self.get_color,
                        mirror=False,
                        periods=False,
                        edges=CONFIG['geometry/show_edges'],
                        edge_alpha=float(CONFIG['geometry/edges_alpha']),
                        edge_lw=0 if geometry.dims == 2 else 1.5
                    )
            finally:
                try:
                    next(updater)
                except StopIteration:
                    pass
