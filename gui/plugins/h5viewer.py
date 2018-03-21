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

# plugin: HDF5 File Viewer
# description: Viewer of the fields saved to HDF5 files with 'save_field' function.


import sys
import h5py

from matplotlib.figure import Figure
import matplotlib.colorbar

import plask

import gui
from gui.qt import QT_API
from gui.qt.QtCore import Qt
from gui.qt.QtGui import *
from gui.qt.QtWidgets import *
from gui.utils.matplotlib import PlotWidgetBase, cursors
from gui.utils.qsignals import BlockQtSignals
from gui.utils.widgets import set_icon_size
from gui.xpldocument import XPLDocument

if QT_API == 'PyQt5':
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class FieldWidget(QWidget):

    class NavigationToolbar(PlotWidgetBase.NavigationToolbar):

        toolitems = (
            ('Save', 'Save image', 'document-save', 'save_figure', None),
            (None, None, None, None, None),
            ('Home', 'Zoom to whole geometry', 'go-home', 'home', None),
            ('Back', 'Back to previous view', 'go-previous', 'back', None),
            ('Forward', 'Forward to next view', 'go-next', 'forward', None),
            (None, None, None, None, None),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'transform-move', 'pan', False),
            ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False),
            (None, None, None, None, None),
            ('Aspect', 'Set equal aspect ratio for both axes', 'system-lock-screen', 'aspect', False),
            (None, None, None, None, None),
            ('Component:', 'Select vector component to plot', None, 'select_component',
             (('long', 'tran', 'vert', 'abs'), 2)),
        )

        def __init__(self, canvas, parent, controller=None, coordinates=True):
            super(FieldWidget.NavigationToolbar, self).__init__(canvas, parent, controller, coordinates)
            self._actions['select_component'].setVisible(False)
            self.comp = 2
            self.mag = True

        def select_component(self, index):
            self.comp = index
            self.parent.update_plot(self.parent.plotted_field, self.parent.plotted_geometry, None, False)

        def enable_component(self, visible, mag):
            self._actions['select_component'].setVisible(visible)
            self.mag = mag

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

            if event.xdata is not None and event.ydata is not None:
                if self.parent.is_profile:
                    s = u'{2[0]} = {0:.4f} µm  value = {1:.3g}'\
                        .format(float(event.xdata), float(event.ydata), self._axes)
                else:
                    s = u'{2[0]} = {0:.4f} µm  {2[1]} = {1:.4f} µm' \
                        .format(float(event.xdata), float(event.ydata), self._axes)
            else:
                s = ''

            if len(self.mode):
                self.set_message('%s   %s' % (s, self.mode))
            else:
                self.set_message(s)

        def set_axes_names(self, names, idx):
            widget = self.widgets['select_component']
            with BlockQtSignals(widget):
                widget.clear()
                widget.addItems(tuple(names))
                if self.mag:
                    widget.addItem('Magnitude')
                widget.setCurrentIndex(min(self.comp, widget.count()-1))
            self._axes = tuple(names[i] for i in idx)

        def aspect(self):
            try: parent = self.parent()
            except TypeError: parent = self.parent
            parent.aspect_locked = not parent.aspect_locked
            if parent.aspect_locked:
                parent.axes.set_aspect('equal')
            else:
                parent.axes.set_aspect('auto')
            self._update_view()
            parent.figure.tight_layout(pad=0.1)
            self.canvas.draw()

    def __init__(self, controller=None, parent=None):
        super(FieldWidget, self).__init__(parent)

        self.controller = controller
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        #self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QPalette.Background).name())
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
        self.cax = None

        self.aspect_locked = False
        self.is_profile = None

        self.setLayout(vbox)

    def update_plot(self, field, geometry, axes_names, reset_zoom=True):
        xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
        self.axes.clear()
        plane = None

        self.is_profile = False
        if isinstance(field.mesh, plask.mesh.Mesh1D):
            self._actions['select_component'].setVisible(len(field.array.shape) > 1)
            self.is_profile = True
            axi = -2,
        elif isinstance(field.mesh, plask.mesh.Rectangular2D):
            if len(field.mesh.axis0) == 1 or len(field.mesh.axis1) == 1:
                self.is_profile = True
            elif reset_zoom:
                xlim = field.mesh.axis0[0], field.mesh.axis0[-1]
                ylim = field.mesh.axis1[0], field.mesh.axis1[-1]
            self.toolbar.enable_component(len(field.array.shape) > 2, field.array.shape[-1] != 4)
            axi = 1, 2
        elif isinstance(field.mesh, plask.mesh.Rectangular3D):
            axs = field.mesh.axis0, field.mesh.axis1, field.mesh.axis2
            axl = tuple(len(a) for a in axs)
            axi = tuple(i for i in range(3) if axl[i] != 1)
            if axi == (0, 1): axi = (1, 0)
            if len(axi) == 1:
                self.is_profile = True
            elif len(axi) == 2:
                if reset_zoom:
                    xlim = axs[axi[0]][0], axs[axi[0]][-1]
                    ylim = axs[axi[1]][0], axs[axi[1]][-1]
                plane = '{}{}'.format(*axi)
            else:
                raise ValueError("Field mesh must have one dimension equal to 1")
            self.toolbar.enable_component(len(field.array.shape) > 3, field.array.shape[-1] != 4)
        else:
            raise TypeError("Unsupported mesh")

        comp = self.toolbar.comp
        if comp == 3:
            if self.toolbar.mag: comp = 'abs'
            else: comp = 2
        if self.is_profile:
            if self.cax is not None:
                self.figure.clf()
                self.axes = self.figure.add_subplot(111, adjustable='datalim')
                self.cax = None
            self.controller.plotted = plask.plot_profile(field, axes=self.axes, comp=comp)
        else:
            self.controller.plotted = plask.plot_field(field, axes=self.axes, plane=plane, comp=comp)
            self.axes.set_xlim(*xlim)
            self.axes.set_ylim(*ylim)
            if geometry is not None:
                try:
                    plask.plot_geometry(axes=self.axes, geometry=geometry, fill=False,
                                        plane=plane, lw=1.0, color='w', alpha=0.25)
                except:
                    pass
            if self.cax is None:
                self.cax, _ = matplotlib.colorbar.make_axes_gridspec(self.axes)
            self.figure.colorbar(mappable=self.controller.plotted, ax=self.axes, cax=self.cax)

        if axes_names is not None:
            self.toolbar.set_axes_names(axes_names, axi)
            self.axes.set_xlabel(u"${}$ [µm]".format(axes_names[axi[0]]))
            if not self.is_profile:
                self.axes.set_ylabel(u"${}$ [µm]".format(axes_names[axi[1]]))

        self.plotted_field = field
        self.plotted_geometry = geometry

        self.axes.set_aspect('equal' if self.aspect_locked else 'auto')
        self.canvas.draw()
        self.figure.tight_layout(pad=0.1)


class ResultsWindow(QMainWindow):

    @staticmethod
    def _get_fields(h5file):
        fields = []

        def visit(name, item):
            if isinstance(item, h5py.Group):
                if '_mesh' in item and '_data' in item:
                    fields.append('/'+name)
        visit('', h5file)
        h5file.visititems(visit)

        return fields

    def __init__(self, filename, parent=None):
        super(ResultsWindow, self).__init__(parent)
        self.setWindowTitle(filename)
        self.plotted_field = None

        self.h5file = h5py.File(filename, 'r')
        fields = self._get_fields(self.h5file)

        self.field_geometries = {}

        splitter1 = QSplitter(self)
        splitter2 = QSplitter(splitter1)
        splitter2.setOrientation(Qt.Vertical)
        splitter1.addWidget(splitter2)

        self.setCentralWidget(splitter1)

        self.field_list = QListWidget(splitter2)
        splitter2.addWidget(self.field_list)

        self.geometry_list = QListWidget(splitter2)
        splitter2.addWidget(self.geometry_list)
        self.geometry_list.hide()

        self.plot_widget = FieldWidget(self, parent=splitter1)
        splitter1.addWidget(self.plot_widget)

        self.document = parent.document
        self.update_geometries()

        if len(fields) > 0:
            self.field_list.setSelectionMode(QAbstractItemView.SingleSelection)
            self.field_list.addItems(fields)
            self.field_list.currentTextChanged.connect(self.field_changed)
            self.field_list.item(0).setSelected(True)
            self.geometry_list.currentTextChanged.connect(self.geometry_changed)
        else:
            self.field_list.setSelectionMode(QAbstractItemView.NoSelection)
            self.field_list.addItem("No fields in the selected file!")

        self.showMaximized()

    def update_geometries(self):
        if not isinstance(self.document, XPLDocument):
            return False
        self.manager = plask.Manager(draft=True)
        self.geometries2d = [g.name for g in self.document.geometry.model.get_roots(2)]
        self.geometries3d = [g.name for g in self.document.geometry.model.get_roots(3)]
        roots = self.document.geometry.model.get_roots()
        self.axes = {g.name: g.get_axes_conf() for g in roots}
        self._default_axes = roots[0].get_axes_conf() if roots else plask.config.axes
        try:
            self.manager.load(self.document.get_content(sections=('defines', 'geometry')))
        except Exception as e:
            from gui import _DEBUG
            if _DEBUG:
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
            return False

    def resizeEvent(self, event):
        super(ResultsWindow, self).resizeEvent(event)
        self.plot_widget.figure.tight_layout(pad=0.1)

    def field_changed(self, name):
        field = plask.load_field(self.h5file, name)
        geometries = ["(none)"]
        if isinstance(field.mesh, plask.mesh.Mesh2D):
            geometries.extend(self.geometries2d)
        elif isinstance(field.mesh, plask.mesh.Mesh3D):
            geometries.extend(self.geometries3d)
        if len(geometries) > 1:
            try:
                geom = str(self.field_geometries[name])
                index = geometries.index(geom)
            except (KeyError, ValueError):
                geom = "(none)"
                index = 0
            with BlockQtSignals(self.geometry_list):
                self.geometry_list.clear()
                self.geometry_list.addItems(geometries)
                self.geometry_list.show()
                self.geometry_list.item(index).setSelected(True)
        else:
            geom = "(none)"
            self.geometry_list.hide()

        if geom in self.manager.geo:
            geometry = self.manager.geo[geom]
        else:
            geometry = None
        self.plot_widget.update_plot(field, geometry, self.axes.get(geom, self._default_axes))

    def geometry_changed(self, geom):
        geom = str(geom)
        fld = self.field_list.currentItem().text()
        field = plask.load_field(self.h5file, fld)
        self.field_geometries[fld] = geom
        if geom in self.manager.geo:
            geometry = self.manager.geo[geom]
        else:
            geometry = None
        self.plot_widget.update_plot(field, geometry, self.axes.get(geom, self._default_axes), False)


class AnalyzeResultsAction(QAction):

    def __init__(self, parent):
        super(AnalyzeResultsAction, self).__init__(QIcon.fromTheme('edit-find'),
                                             'Anal&yze Results...', parent)
        self.setShortcut(Qt.CTRL + Qt.SHIFT + Qt.Key_A)
        self.triggered.connect(self.execute)

    def execute(self):
        filename = QFileDialog.getOpenFileName(None, "Open HDF5 File", gui.CURRENT_DIR, "HDF5 file (*.h5)")
        if type(filename) == tuple: filename = filename[0]
        if not filename: return

        window = ResultsWindow(filename, self.parent())
        window.show()


if gui.ACTIONS:
    gui.ACTIONS.append(None)
gui.ACTIONS.append(AnalyzeResultsAction)
