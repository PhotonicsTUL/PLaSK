# coding: utf8
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
import weakref

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import plask
from plask._plot_geometry import plane_to_axes

from .qsignals import BlockQtSignals

from ..qt import QT_API
from ..qt.QtCore import *
from ..qt.QtWidgets import *
from ..qt.QtGui import *
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.backend_bases import NavigationToolbar2
import matplotlib.colors

from ..utils.widgets import set_icon_size, ComboBox

from .config import CONFIG

class Cursors:
    # this class is only used as a simple namespace
    HAND, POINTER, SELECT_REGION, MOVE = range(4)
cursors = Cursors()

cursord = {
    cursors.MOVE: Qt.CursorShape.SizeAllCursor,
    cursors.HAND: Qt.CursorShape.PointingHandCursor,
    cursors.POINTER: Qt.CursorShape.ArrowCursor,
    cursors.SELECT_REGION: Qt.CursorShape.CrossCursor,
}


class BwColor:
    def __init__(self, colors, axes, compress=0.5):
        self.colors = plask.ColorFromDict(colors, axes)
        self.compress = compress
        bc = 1.0 - compress
        self.background = tuple(bc * c for c in matplotlib.colors.to_rgb(axes.get_facecolor()))

    def __call__(self, material):
        try:
            color = self.colors(material)
            if isinstance(color, str):
                if color.startswith('#'): color = color[1:]
                r, g, b = tuple(ord(c)/255. for c in color.decode('hex'))
            else:
                r, g, b = color
        except:
            r, b, b = 0.5, 0.5, 0.5
        bw = self.compress * (0.2126*r + 0.7152*b + 0.0722*b)
        return tuple(bw + c for c in self.background)


class PlotWidgetBase(QWidget):

    class NavigationToolbar(NavigationToolbar2QT):

        toolitems = (
            ('Plot', 'Plot selected geometry object', 'draw-brush', 'plot', None, 'plot_plot'),
            ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', True, 'plot_refresh'),
            (None, None, None, None, None, None),
            ('Home', 'Zoom to whole geometry', 'go-home', 'home', None, 'plot_home'),
            ('Back', 'Back to previous view', 'go-previous', 'back', None, 'plot_back'),
            ('Forward', 'Forward to next view', 'go-next', 'forward', None, 'plot_forward'),
            (None, None, None, None, None, None),
            ('Export', 'Export the figure', 'document-save', 'save_figure', None, 'plot_export'),
            (None, None, None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'transform-move', 'pan', False, 'plot_pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False, 'plot_zoom'),
            (None, None, None, None, None, None),
            ('Aspect', 'Set equal aspect ratio for both axes', 'system-lock-screen', 'aspect', False, 'plot_aspect'),
            (None, None, None, None, None, None),
            ('Plane:', 'Select longitudinal-transverse plane', None, 'select_plane',
             (('tran-long', 'long-vert', 'tran-vert'), 2), 'plot_plane'),
        )

        def __init__(self, canvas, parent, controller=None, coordinates=True):
            self.widgets = {}

            # We need to copy-change the parent constructor, as the original does not suit our needs
            QToolBar.__init__(self, parent)
            self.setAllowedAreas(Qt.ToolBarArea(Qt.ToolBarArea.TopToolBarArea | Qt.ToolBarArea.BottomToolBarArea))
            self.coordinates = coordinates
            self._actions = {}
            self._create_toolbar()
            NavigationToolbar2.__init__(self, canvas)

            self.controller = weakref.proxy(controller)
            if 'select_plane' in self._actions:
                self.disable_planes(('long','tran','vert'))

        def _icon(self, name):
            if name is not None:
                return QIcon.fromTheme(name)

        def _init_toolbar(self):
            pass    # this may be called by an old Matplotlib

        def _create_toolbar(self):
            self.layout().setContentsMargins(0,0,0,0)
            for text, tooltip_text, icon, callback, checked, shortcut in self.toolitems:
                if text is None:
                    self.addSeparator()
                elif callback is None:
                    self.addWidget(QLabel(text))
                else:
                    if type(checked) in (tuple, list):
                        combo = ComboBox()
                        combo.addItems(checked[0])
                        combo.setCurrentIndex(checked[1])
                        combo.currentIndexChanged.connect(getattr(self, callback))
                        if tooltip_text is not None:
                            combo.setToolTip(tooltip_text)
                        self.widgets[callback] = combo
                        if text is None:
                            widget = combo
                        else:
                            widget = QWidget()
                            layout = QHBoxLayout()
                            layout.setContentsMargins(0, 2, 0, 0)
                            layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
                            layout.addWidget(QLabel(text))
                            layout.addWidget(combo)
                            widget.setLayout(layout)
                        action = self.addWidget(widget)
                    else:
                        ic = self._icon(icon)
                        if ic is not None:
                            action = self.addAction(ic, text, getattr(self, callback))
                        else:
                            action = self.addAction(text, getattr(self, callback))
                        if checked is not None:
                            action.setCheckable(True)
                            if checked: action.setChecked(True)
                        if tooltip_text is not None:
                            action.setToolTip(tooltip_text)
                    if shortcut is not None:
                        CONFIG.set_shortcut(action, shortcut)
                    self._actions[callback] = action
            # Add the x,y location widget at the right side of the toolbar
            # The stretch factor is 1 which means any resizing of the toolbar
            # will resize this label instead of the buttons.
            if self.coordinates:
                self.locLabel = QLabel("", self)
                self.locLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
                self.locLabel.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored))
                label_action = self.addWidget(self.locLabel)
                label_action.setVisible(True)

            self._axes = 'tran', 'vert'
            self._axes_names = 'long', 'tran', 'vert'

        def plot(self):
            if self.controller is not None:
                self.controller.plot()

        @property
        def _current_mode(self):
            try:
                return self.mode.name
            except AttributeError:
                return self._active if self._active is not None else 'NONE'

        def set_cursor(self, cursor):
            super(PlotWidgetBase.NavigationToolbar, self).set_cursor(cursor)
            self._lastCursor = cursor

        def mouse_move(self, event):
            if not event.inaxes or not self._current_mode or self._current_mode == 'NONE':
                if self._lastCursor != cursors.POINTER:
                    self.set_cursor(cursors.POINTER)
            else:
                if self._current_mode == 'ZOOM':
                    if self._lastCursor != cursors.SELECT_REGION:
                        self.set_cursor(cursors.SELECT_REGION)
                elif (self._current_mode == 'PAN' and
                      self._lastCursor != cursors.MOVE):
                    self.set_cursor(cursors.MOVE)

            if event.xdata is not None and event.ydata is not None:
                s = u'{2[0]} = {0:.4f} µm  {2[1]} = {1:.4f} µm'.format(float(event.xdata), float(event.ydata), self._axes)
            else:
                s = ''

            if len(self.mode):
                self.set_message('%s   %s' % (s, self.mode))
            else:
                self.set_message(s)

        def aspect(self):
            try:
                parent = self.canvas.parent()
            except AttributeError:
                try:
                    parent = self.parent()
                except TypeError:
                    parent = self.parent
            parent.aspect_locked = not parent.aspect_locked
            if parent.aspect_locked:
                parent.axes.set_aspect('equal')
            else:
                parent.axes.set_aspect('auto')
                self._update_view()
            parent.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
            self.canvas.draw()

        def auto_refresh(self):
            if self.controller is not None:
                self.controller.plot_auto_refresh = not self.controller.plot_auto_refresh

        def select_plane(self, index):
            plane = ('10', '02', '12')[index]
            self._axes = self._axes_names[int(plane[0])], self._axes_names[int(plane[1])]
            self.controller.checked_plane = plane
            self.set_message(self.mode)

        def disable_planes(self, axes):
            self._actions['select_plane'].setVisible(False)
            try:
                if ',' in axes: axes = axes.split(',')
            except TypeError:
                self._axes = 'tran', 'vert'
            else:
                self._axes = axes[-2:]

        def enable_planes(self, axes=None):
            self._actions['select_plane'].setVisible(True)
            if axes is not None and axes != self._axes_names:
                if ',' in axes:
                    axes = axes.split(',')
                indx = self.widgets['select_plane'].currentIndex()
                with BlockQtSignals(self.widgets['select_plane']):
                    self.widgets['select_plane'].clear()
                    self.widgets['select_plane'].addItems((axes[1]+'-'+axes[0], axes[0]+'-'+axes[2], axes[1]+'-'+axes[2]))
                self._axes_names = axes
                self._axes = axes[int(self.controller.checked_plane[0])], axes[int(self.controller.checked_plane[1])]
                with BlockQtSignals(self.widgets['select_plane']):
                    self.widgets['select_plane'].setCurrentIndex(indx)
                self.set_message(self.mode)

        def clear_history(self):
            try:
                self._nav_stack.clear()
            except AttributeError:
                self._views.clear()
                self._positions.clear()

        def set_history_buttons(self):
            try:
                stack = self._nav_stack
            except AttributeError:
                stack = self._views
            if len(stack) <= 1:
                self._actions['back'].setEnabled(False)
                self._actions['forward'].setEnabled(False)
            elif stack._pos == 0:
                self._actions['back'].setEnabled(False)
                self._actions['forward'].setEnabled(True)
            elif stack._pos == len(stack)-1:
                self._actions['back'].setEnabled(True)
                self._actions['forward'].setEnabled(False)
            else:
                self._actions['back'].setEnabled(True)
                self._actions['forward'].setEnabled(True)

    def __init__(self, controller=None, parent=None):
        super().__init__(parent)

        self.selectors = []
        self.guidelines = {}
        self.controller = weakref.proxy(controller)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        #self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.figure.set_facecolor(self.palette().color(QPalette.ColorRole.Window).name())
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.updateGeometry()
        self.toolbar = self.NavigationToolbar(self.canvas, self, controller)
        set_icon_size(self.toolbar)

        vbox = QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)
        vbox.update()
        vbox.setContentsMargins(0, 0, 2, 1)

        self.axes = self.figure.add_subplot(111, adjustable='datalim')
        self.axes.grid(True)
        # self.axes.tick_params(axis='both', length=6, width=1, direction='in', which='major', zorder=9,
        #                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        # self.axes.tick_params(axis='both', length=3, width=1, direction='in', which='minor', zorder=9)
        self.axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # Aspect lock state
        self.aspect_locked = False

        self.setLayout(vbox)

    def zoom_bbox(self, box, margin=0.1):
        try:
            stack = self.toolbar._nav_stack
        except AttributeError:
            stack = self.toolbar._views
        if stack.empty():
            self.toolbar.push_current()
        ax = plane_to_axes(self.plane, 2 if isinstance(box, plask.geometry.Box2D) else 3)
        m = (box.upper[ax[0]] - box.lower[ax[0]]) * margin
        self.axes.set_xlim(box.lower[ax[0]] - m, box.upper[ax[0]] + m)
        m = (box.upper[ax[1]] - box.lower[ax[1]]) * margin
        self.axes.set_ylim(box.lower[ax[1]] - m, box.upper[ax[1]] + m)
        if ax[0] > ax[1] and not self.axes.yaxis_inverted():
            self.axes.invert_yaxis()
        self.toolbar.push_current()
        self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.figure.set_tight_layout(0)

    def plot_updater(self, set_limits, plane='12'):
        xlim, ylim = self.axes.get_xlim(), self.axes.set_ylim()
        # self.figure.clear()
        self.axes.cla()
        self.selectors = []
        self.axes.minorticks_on()
        self.axes.grid(which='major', ls='-', lw=1, alpha=0.4, color='0.5')
        self.axes.grid(which='minor', ls='-', lw=1, alpha=0.1, color='0.5')
        self.axes.axhline(0., ls='-', color=CONFIG['plots/axes_color'], alpha=0.4, zorder=3)
        self.axes.axvline(0., ls='-', color=CONFIG['plots/axes_color'], alpha=0.4, zorder=3)
        margin = 0.1 if set_limits else None
        yield margin
        for ax in self.axes.xaxis, self.axes.yaxis:
            ax.set_major_locator(MaxNLocator(nbins=10, steps=(1, 10)))
            ax.set_minor_locator(MaxNLocator(nbins=100, steps=(1, 10)))
        if not set_limits:
            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
        self.axes.set_aspect('equal' if self.aspect_locked else 'auto')
        self.canvas.draw()
        self.plane = plane

    def dock_window(self, window):
        res = QDockWidget('Geometry', window)
        res.setContentsMargins(0, 0, 0, 0)
        res.setWidget(self)
        return res

    def clear(self):
        self.axes.clear()
        self.canvas.draw()
