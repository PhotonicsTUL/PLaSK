from ...qt import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from .plot import plot_geometry_object

class PlotWidget(QtGui.QGroupBox):

    def __init__(self, parent = None):
        super(PlotWidget, self).__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        #self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.canvas.updateGeometry()
        self.plot_toolbar = NavigationToolbar(self.canvas, self)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.plot_toolbar)
        vbox.addWidget(self.canvas)

        self.setLayout(vbox)

    def resizeEvent(self, event):
        super(PlotWidget, self).resizeEvent(event)
        self.figure.set_tight_layout(0)

    def update_plot(self, to_plot):
        self.figure.clear()
        if to_plot is not None:
            plot_geometry_object(figure=self.figure, geometry=to_plot, fill=True, set_limits=True)
            self.figure.set_tight_layout(0)
            self.canvas.draw()

    def dock_window(self, window):
        res = QtGui.QDockWidget('Geometry', window)
        res.setContentsMargins(0, 0, 0, 0)
        res.setWidget(self)
        return res