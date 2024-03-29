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

import matplotlib
from matplotlib.colors import ColorConverter

import plask
from plask._plot_geometry import plane_to_axes
from ...utils.config import CONFIG
from ...utils.matplotlib import PlotWidgetBase, PlotWidgetWithInfoBase

to_rgba = ColorConverter().to_rgba


class PlotWidget(PlotWidgetWithInfoBase):

    class NavigationToolbar(PlotWidgetBase.NavigationToolbar):

        toolitems = (
            ('Plot', 'Plot selected geometry object', 'draw-brush', 'plot', None, 'plot_plot'),
            ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', True, 'plot_refresh'),
            (None, None, None, None, None, None),
            ('Home', 'Zoom to whole geometry', 'go-home', 'home', None, 'plot_home'),
            ('Back', 'Back to previous view', 'go-previous', 'back', None, 'plot_back'),
            ('Forward', 'Forward to next view', 'go-next', 'forward', None, 'plot_forward'),
            ('Zoom Selected', 'Zoom to selected object', 'zoom-fit-best', 'zoom_current', None, 'plot_zoom_current'),
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

        def home(self):
            if self.controller is not None:
                self.controller.zoom_to_root()

        def zoom_current(self):
            if self.controller is not None:
                self.controller.zoom_to_current()

        def select_plane(self, index):
            super().select_plane(index)
            plotted_tree_element = self.controller.plotted_tree_element
            if plotted_tree_element is not None and getattr(plotted_tree_element, 'dim') == 3:
                self.controller.plot_element(plotted_tree_element, set_limits=True)

    def __init__(self, controller=None, parent=None, picker=None):
        super().__init__(controller, parent)
        colors = CONFIG['geometry/material_colors'].copy()
        self.get_color = plask.ColorFromDict(colors, self.axes)
        self.picker = picker

    def clean_selectors(self):
        for artist in self.selectors:
            artist.remove()
        self.selectors = []

    def add_selector(self, artist, clipbox=None):
        self.axes.add_patch(artist)
        if clipbox is not None:
            artist.set_clip_box(clipbox)
        self.selectors.append(artist)

    def show_selection(self, root, selected, clean=True):
        if selected is None or root is None: return
        bboxes = root.get_object_bboxes(selected)
        if not bboxes: return
        ax = plane_to_axes(self.plane, 2 if isinstance(bboxes[0], plask.geometry.Box2D) else 3)
        positions = root.get_object_positions(selected)
        box_color = to_rgba(CONFIG['geometry/selected_color'], alpha=float(CONFIG['geometry/selected_alpha']))
        box_lw = float(CONFIG['geometry/selected_width'])
        show_origin = CONFIG['geometry/show_origin']
        origin_color = CONFIG['geometry/origin_color']
        origin_alpha = float(CONFIG['geometry/origin_alpha'])
        origin_lw = float(CONFIG['geometry/origin_width'])
        origin_size = float(CONFIG['geometry/origin_size'])
        for bbox, pos in zip(bboxes, positions):
            x, y = bbox.lower[ax[0]], bbox.lower[ax[1]]
            dx, dy = bbox.upper[ax[0]] - x, bbox.upper[ax[1]] - y
            if dx >= 0 and dy >= 0:
                rect = matplotlib.patches.Rectangle((x, y), dx, dy, zorder=100.0, fill=False, ec=box_color, lw=box_lw)
                self.add_selector(rect)
                if show_origin:
                    origin = matplotlib.lines.Line2D((pos[ax[0]],), (pos[ax[1]],),
                                                     zorder=101.0,
                                                     marker='+',
                                                     mec=origin_color,
                                                     alpha=origin_alpha,
                                                     mew=origin_lw,
                                                     ms=origin_size)
                    self.axes.add_line(origin)
                    self.selectors.append(origin)
        guidelines = self.guidelines.get(selected, ())
        for guideline in guidelines:
            self.add_selector(guideline)

    def update_plot(self, geometry, set_limits, plane='12'):
        updater = self.plot_updater(set_limits, plane)
        for margin in updater:
            if geometry is not None:
                _, self.guidelines = plask.plot_geometry(
                    axes=self.axes,
                    geometry=geometry,
                    fill=True,
                    margin=margin,
                    zorder=1,
                    plane=plane,
                    lw=1.5,
                    picker=self.picker,
                    get_color=self.get_color,
                    extra=dict(
                        ec=to_rgba(CONFIG['geometry/extra_color'], alpha=float(CONFIG['geometry/extra_alpha'])),
                        lw=float(CONFIG['geometry/extra_width'])
                    ),
                    mirror=False,
                    periods=False,
                    edges=CONFIG['geometry/show_edges'],
                    edge_alpha=float(CONFIG['geometry/edges_alpha']),
                    edge_lw=0 if geometry.dims == 2 else 1.5
                )
