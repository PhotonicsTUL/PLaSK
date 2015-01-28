import plask

from ...qt import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar


class PlotWidget(QtGui.QGroupBox):

    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        #self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.updateGeometry()
        self.plot_toolbar = NavigationToolbar(self.canvas, self)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.plot_toolbar)
        vbox.addWidget(self.canvas)

        self.axes = self.figure.add_subplot(111)
        self.axes.grid()
        self.axes.tick_params(axis='both', length=6, width=1, direction='in', which='major', zorder=9,
                              labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        self.axes.tick_params(axis='both', length=3, width=1, direction='in', which='minor', zorder=9)

        self.setLayout(vbox)

    def resizeEvent(self, event):
        super(PlotWidget, self).resizeEvent(event)
        self.figure.set_tight_layout(0)

    def update_plot(self, to_plot, set_limits=True):
        # self.figure.clear()
        self.axes.cla()
        self.axes.minorticks_on()
        if to_plot is not None:
            self.axes.grid(which='major', ls='-', lw=1, alpha=0.5, color='0.5')
            self.axes.grid(which='minor', ls='-', lw=1, alpha=0.2, color='0.5')
            self.axes.axhline(0., ls='-', color='k', alpha=0.5, zorder=3)
            self.axes.axvline(0., ls='-', color='k', alpha=0.5, zorder=3)
            margin = 0.02 if set_limits else None
            plask.plot_geometry(axes=self.axes, geometry=to_plot, fill=True, margin=margin, zorder=1,
                                plane='01')
            self.canvas.draw()


    def dock_window(self, window):
        res = QtGui.QDockWidget('Geometry', window)
        res.setContentsMargins(0, 0, 0, 0)
        res.setWidget(self)
        return res