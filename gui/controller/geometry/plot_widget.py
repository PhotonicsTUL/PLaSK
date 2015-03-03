# coding=utf-8
import plask

from ...qt import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.ticker import MultipleLocator, MaxNLocator


class Cursors(object):
    # this class is only used as a simple namespace
    HAND, POINTER, SELECT_REGION, MOVE = list(range(4))
cursors = Cursors()

cursord = {
    cursors.MOVE: QtCore.Qt.SizeAllCursor,
    cursors.HAND: QtCore.Qt.PointingHandCursor,
    cursors.POINTER: QtCore.Qt.ArrowCursor,
    cursors.SELECT_REGION: QtCore.Qt.CrossCursor,
}


class NavigationToolbar(NavigationToolbar2QT):

    def __init__(self, canvas, parent, controller=None, coordinates=True):
        super(NavigationToolbar, self).__init__(canvas, parent, coordinates)
        self.controller = controller

    def _icon(self, name):
        if name is not None:
            return QtGui.QIcon.fromTheme(name)

    def _init_toolbar(self):
        self.layout().setContentsMargins(0,0,0,0)
        for text, tooltip_text, icon, callback, checked in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                ic = self._icon(icon)
                if ic is not None:
                    a = self.addAction(ic, text, getattr(self, callback))
                else:
                    a = self.addAction(text, getattr(self, callback))
                if checked is not None:
                    a.setCheckable(True)
                    if checked: a.setChecked(True)
                self._actions[callback] = a
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
        self.buttons = {}
        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtGui.QLabel("", self)
            self.locLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
            self.locLabel.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored))
            label_action = self.addWidget(self.locLabel)
            label_action.setVisible(True)

        self._axes = 'tran', 'vert'
        self._axes_names = 'long', 'tran', 'vert'

    toolitems = (
        # ('Home', 'Reset original view', 'go-home', 'home'),
        ('Plot', 'Plot selected geometry object', 'applications-graphics', 'plot', None),
        ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', True),
        (None, None, None, None, None),
        ('Back', 'Back to  previous view', 'go-previous', 'back', None),
        ('Forward', 'Forward to next view', 'go-next', 'forward', None),
        (None, None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'transform-move', 'pan', False),
        ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False),
        (None, None, None, None, None),
        ('Aspect', 'Set equal aspect ratio for both axes', 'system-lock-screen', 'aspect', False),
        (None, None, None, None, None),
        ('long-tran', 'Select longitudinal-transverse plane', None, 'plane01', False),
        ('long-vert', 'Select longitudinal-vertical plane', None, 'plane02', False),
        ('tran-vert', 'Select transverse-vertical plane', None, 'plane12', True),
    )

    def mouse_move(self, event):
        if not event.inaxes or not self._active:
            if self._lastCursor != cursors.POINTER:
                self.set_cursor(cursors.POINTER)
                self._lastCursor = cursors.POINTER
        else:
            if self._active == 'ZOOM':
                if self._lastCursor != cursors.SELECT_REGION:
                    self.set_cursor(cursors.SELECT_REGION)
                    self._lastCursor = cursors.SELECT_REGION
            elif (self._active == 'PAN' and
                  self._lastCursor != cursors.MOVE):
                self.set_cursor(cursors.MOVE)
                self._lastCursor = cursors.MOVE

        s = u'{1[0]} = {0.xdata:.4f} µm  {1[1]} = {0.ydata:.4f} µm'.format(event, self._axes)

        if len(self.mode):
            self.set_message('%s   %s' % (s, self.mode))
        else:
            self.set_message(s)

    def aspect(self):
        try: parent = self.parent()
        except TypeError: parent = self.parent
        parent.aspect_locked = not parent.aspect_locked
        parent.axes.set_aspect('equal' if parent.aspect_locked else 'auto')
        parent.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.draw()

    def plot(self):
        if self.controller is not None:
            self.controller.plot()

    def auto_refresh(self):
        if self.controller is not None:
            self.controller.plot_auto_refresh = not self.controller.plot_auto_refresh

    def _select_plane(self, plane):
        self._actions['plane01'].setChecked(plane == '01')
        self._actions['plane02'].setChecked(plane == '02')
        self._actions['plane12'].setChecked(plane == '12')
        self._axes = self._axes_names[int(plane[0])], self._axes_names[int(plane[1])]
        self.controller.checked_plane = plane
        if self.controller.plotted_tree_element is not None and \
           getattr(self.controller.plotted_tree_element, 'dim') == 3:
            self.controller.plot_element(self.controller.plotted_tree_element, set_limits=True)
        self.set_message(self.mode)

    def disable_planes(self, axes):
        self._actions['plane01'].setVisible(False)
        self._actions['plane02'].setVisible(False)
        self._actions['plane12'].setVisible(False)
        self._axes = axes[-2:]

    def enable_planes(self, axes=None):
        self._actions['plane01'].setVisible(True)
        self._actions['plane02'].setVisible(True)
        self._actions['plane12'].setVisible(True)
        if axes is not None:
            if ',' in axes:
                axes = axes.split(',')
            if len(axes[0]) > 1 or len(axes[1]) > 1 or len(axes[2]) > 1:
                sep = '-'
            else:
                sep = ''
            self._actions['plane01'].setText(axes[0]+sep+axes[1])
            self._actions['plane02'].setText(axes[0]+sep+axes[2])
            self._actions['plane12'].setText(axes[1]+sep+axes[2])
            self._axes_names = axes
            self._axes = axes[int(self.controller.checked_plane[0])], axes[int(self.controller.checked_plane[1])]

    def plane01(self):
        self._select_plane('01')

    def plane02(self):
        self._select_plane('02')

    def plane12(self):
        self._select_plane('12')


class PlotWidget(QtGui.QGroupBox):

    def __init__(self, controller=None, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        self.controller = controller
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        #self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.updateGeometry()
        self.toolbar = NavigationToolbar(self.canvas, self, controller)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)

        self.axes = self.figure.add_subplot(111, adjustable='datalim')
        self.axes.grid()
        # self.axes.tick_params(axis='both', length=6, width=1, direction='in', which='major', zorder=9,
        #                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        # self.axes.tick_params(axis='both', length=3, width=1, direction='in', which='minor', zorder=9)
        self.axes.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        # Aspect lock state
        self.aspect_locked = False

        self.setLayout(vbox)

    def resizeEvent(self, event):
        super(PlotWidget, self).resizeEvent(event)
        self.figure.set_tight_layout(0)

    def update_plot(self, to_plot, set_limits=True, plane='12'):
        # self.figure.clear()
        self.axes.cla()
        self.axes.minorticks_on()
        if to_plot is not None:
            self.axes.grid(which='major', ls='-', lw=1, alpha=0.4, color='0.5')
            self.axes.grid(which='minor', ls='-', lw=1, alpha=0.1, color='0.5')
            self.axes.axhline(0., ls='-', color='k', alpha=0.4, zorder=3)
            self.axes.axvline(0., ls='-', color='k', alpha=0.4, zorder=3)
            margin = 0.1 if set_limits else None
            plask.plot_geometry(axes=self.axes, geometry=to_plot, fill=True, margin=margin, zorder=1,
                                plane=plane, lw=1.5)
            for ax in self.axes.xaxis, self.axes.yaxis:
                ax.set_major_locator(MaxNLocator(nbins=10, steps=(1, 10)))
                ax.set_minor_locator(MaxNLocator(nbins=100, steps=(1, 10)))
            self.axes.set_aspect('equal' if self.aspect_locked else 'auto')
            self.canvas.draw()

    def dock_window(self, window):
        res = QtGui.QDockWidget('Geometry', window)
        res.setContentsMargins(0, 0, 0, 0)
        res.setWidget(self)
        return res