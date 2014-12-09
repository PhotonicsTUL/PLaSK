from ...qt import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from .plot import plot_geometry

class PlotDock(QtGui.QDockWidget):

    def __init__(self, window):
        super(PlotDock, self).__init__('Geometry', window)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.canvas.updateGeometry()
        self.plot_toolbar = NavigationToolbar(self.canvas, self)

    def resizeEvent(self, event):
        super(PlotDock, self).resizeEvent(event)
        self.figure.set_tight_layout(0)

    def update_plot(self, to_plot):
        self.figure.clear()
        if to_plot is not None:
            plot_geometry(self.figure, to_plot)
            self.canvas.draw()