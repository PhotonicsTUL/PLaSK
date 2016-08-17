# -*- coding: utf-8 -*-
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
import numpy as np
import itertools

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib import ticker
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot

from ...qt import QtGui
from ...qt.QtCore import Qt

from ...xpldocument import FieldParser

from ...controller.geometry.object import GNObjectController
from ...utils.str import none_to_empty
from ...utils.widgets import MultiLineEdit


class GNLatticeController(GNObjectController):

    def _on_point_set(self, index, value):
        def setter(n, v): n.vectors = n.vectors[0:index] + (v,) + n.vectors[index+1:]
        self._set_node_by_setter_undoable(setter, value, self.node.vectors[index],
            'change {} lattice vector'.format('first' if index == 0 else 'second')
        )

    def _segments_changed(self):
        segments = (seg for seg in (val.strip() for val in self.segments.get_values()) if seg)
        self._set_node_property_undoable('segments', ' ^ '.join(segments))
        return True

    def construct_lattice_edit(self, node_property_name=None, display_property_name=None,
                               change_cb=None, sep='\n', edit_cb=None):
        res = MultiLineEdit(change_cb=change_cb)
        # res = TextEditWithCB(key_cb=key_cb)
        # res.setTabChangesFocus(True)
        # res.setFixedHeight(int(3.5 * QtGui.QFontMetrics(res.font()).height()))
        if edit_cb is not None:
            left = QtGui.QPushButton("&Edit...")
            left.clicked.connect(edit_cb)
        else:
            left = "Boundaries:"
        self._get_current_form().addRow(left, res)
        if change_cb is not None:
            res.focus_out_cb = change_cb
        elif node_property_name is not None:
            res.focus_out_cb = lambda: self._set_node_property_undoable(node_property_name, sep.join(res.get_values()),
                                                                        display_property_name)
        return res

    def construct_form(self):
        self.construct_group('Lattice vectors')
        self.vectors = (self.construct_point_controllers(row_name='First:',
                                                         change_cb=lambda vec: self._on_point_set(0, vec)),
                        self.construct_point_controllers(row_name='Second:',
                                                         change_cb=lambda vec: self._on_point_set(1, vec)))
        self.construct_group('Lattice Boundaries')
        self.segments = self.construct_lattice_edit(node_property_name='segments',
                                                    change_cb=self._segments_changed, edit_cb=self.edit_segments)
        self.segments.setToolTip(u'One or more polygons formed by two or more vertices separated by ``;`` characters.\n'
                                 u'Each vertex consists of two space-separated integers. Every polygon should be put\n'
                                 u'in a separate line: it either adds or removes nodes from the lattice.')
        super(GNLatticeController, self).construct_form()

    def fill_form(self):
        super(GNLatticeController, self).fill_form()
        for i in range(0, self.node.dim):
            self.vectors[0][i].setText(none_to_empty(self.node.vectors[0][i]))
            self.vectors[1][i].setText(none_to_empty(self.node.vectors[1][i]))
        self.segments.set_values(s.strip() for s in none_to_empty(self.node.segments).split('^'))

    def edit_segments(self):
        parser = FieldParser(self.document)
        vecs = []
        try:
            for vec in self.node.vectors:
                v = []
                for c in vec:
                    v.append(float(parser.eval(c)))
                vecs.append(np.array(v))
        except:
            QtGui.QMessageBox.critical(None, "Wring Lattice Vectors",
                                             "No proper lattice vectors are defined. "
                                             "Define them first before starting the boundary editor.")
            return

        items = []
        msg = True
        for seg in parser.eval(none_to_empty(self.node.segments)).split('^'):
            item = []
            try:
                for pt in seg.split(';'):
                    pt = pt.strip()
                    if not pt: continue
                    x, y = (int(p) for p in pt.split())
                    item.append((x,y))
            except:
                if msg:
                    answer = QtGui.QMessageBox.warning(None, "Unrecognized Boundary",
                                                       "At least one boundary segment cannot be parsed. "
                                                       "Do you want to launch the editor ignoring wrong "
                                                       "segments?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                    if answer == QtGui.QMessageBox.Yes: msg = False
                    else: return
            else:
                if item:
                    items.append(item)

        dialog = LatticeEditor(vecs, items)
        if dialog.exec_():
            segments = ' ^ '.join('; '.join('{} {}'.format(*xy) for xy in item) for item in dialog.items)
            print(segments)
            self._set_node_property_undoable('segments', segments)


class MultipleLocator(ticker.MultipleLocator):
    def __init__(self, base=1.0):
        super(MultipleLocator, self).__init__(base)
        self.create_dummy_axis()

    def __call__(self, v1, v2):
        self.set_bounds(v1, v2)
        locs = super(MultipleLocator, self).__call__()
        return np.array(locs), len(locs), None


class LatticeEditor(QtGui.QDialog):

    def __init__(self, vecs, items=None, parent=None):
        super(LatticeEditor, self).__init__(parent)
        self.setWindowTitle("Edit Lattice Boundaries")
        vbox = QtGui.QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        vbox.addWidget(self.canvas)
        #self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.figure.    subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas.updateGeometry()
        buttons = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)
        self.setLayout(vbox)

        if vecs[0][2] == 0. and vecs[1][2] == 0.:
            a, b = vecs[0][1], -vecs[0][0]
            c, d = vecs[1][1], -vecs[1][0]
        elif vecs[0][1] == 0. and vecs[1][1] == 0.:
            a, b = vecs[0][0], vecs[0][2]
            c, d = vecs[1][0], vecs[1][2]
        elif vecs[0][0] == 0. and vecs[1][0] == 0.:
            a, b = vecs[0][1], vecs[0][2]
            c, d = vecs[1][1], vecs[1][2]
        else:
            a = np.sqrt(np.sum(vecs[0]**2))
            b = 0.
            c = np.dot(*vecs) / a
            d = np.sqrt(np.sum(vecs[1]**2) - c**2)

        self.tr = Affine2D.from_values(a, b, c, d, 0., 0.)

        self.tri = self.tr.inverted()
        loc = MultipleLocator(1)
        grid_helper = GridHelperCurveLinear(self.tr, grid_locator1=loc, grid_locator2=loc)
        self.axes = Subplot(self.figure, 1, 1, 1, grid_helper=grid_helper, adjustable='datalim')
        self.figure.add_subplot(self.axes)
        self.axes.set_aspect(1.)
        self.axes.axis['t1'] = self.axes.new_floating_axis(0, 0.)
        self.axes.axis['t2'] = self.axes.new_floating_axis(1, 0.)
        self.axes.grid(True)

        if items:
            cc1, cc2 = zip(*itertools.chain(*items))
            lo1, lo2 = min(cc1)-1, min(cc2)-1
            hi1, hi2 = max(cc1)+1, max(cc2)+1
            del cc1, cc2
            xy = [self.tr.transform(c) for c in ((lo1, lo2), (lo1, hi2), (hi1, lo2), (hi1, hi2))]
            self.axes.set_ylim(min(y[1] for y in xy), max(y[1] for y in xy))
            self.axes.set_xlim(min(x[0] for x in xy), max(x[0] for x in xy))
        else:
            self.axes.set_ylim(-5.5, 5.5)
            self.axes.set_xlim(-5.5, 5.5)

        self.canvas.mpl_connect('draw_event', self.draw_callback)
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.mark = Line2D([0], [0], marker='o', mfc='#e5ae38', ms=6., animated=True)
        self.mark.set_visible(False)
        self.canvas.draw()
        self.items = [] if items is None else items
        self._set_lines()
        self.current = None
        self._undo_stack = [list(self.items)]
        self._undo_index = 0

    def _set_lines(self):
        self.axes.lines = []
        self.axes.add_line(self.mark)
        for item in self.items:
            item = [self.tr.transform(p) for p in item]
            xx = [p[0] for p in item] + [item[0][0]]
            yy = [p[1] for p in item] + [item[0][1]]
            line = Line2D(xx, yy, ls='-', color='#30a2da', lw=2., animated=True)
            self.axes.add_line(line)

    def _get_node(self, event):
        return tuple(round(c) for c in self.tri.transform([event.xdata, event.ydata]))

    def draw_callback(self, event=None):
        self.mark.set_visible(False)
        for line in self.axes.lines[0:]:
            self.axes.draw_artist(line)
        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self.axes.draw_artist(self.mark)
        self.canvas.blit(self.axes.bbox)

    def _draw_current(self, x, y):
        if self.current is None: return
        xy = [self.tr.transform(p) for p in self.current]
        line = self.axes.lines[-1]
        line.set_xdata([p[0] for p in xy] + [x])
        line.set_ydata([p[1] for p in xy] + [y])
        self.axes.draw_artist(line)

    def _save_items(self):
        self._undo_index += 1
        self._undo_stack = self._undo_stack[:self._undo_index]
        self._undo_stack.append(list(self.items))

    def _undo(self):
        if self._undo_index > 0:
            self._undo_index -= 1
            self.items = list(self._undo_stack[self._undo_index])
            self._set_lines()
            self.canvas.draw()

    def _redo(self):
        if self._undo_index < len(self._undo_stack)-1:
            self._undo_index += 1
            self.items = list(self._undo_stack[self._undo_index])
            self._set_lines()
            self.canvas.draw()

    def button_press_callback(self, event):
        if event.inaxes is None: return
        if event.button == 1:
            pt = self._get_node(event)
            if self.current is None:
                #self.background = self.canvas.copy_from_bbox(self.axes.bbox)
                self.current = [pt]
                x, y = self.tr.transform(pt)
                line = Line2D([x], [y], ls='-', color='#fc4f30', lw=2., marker='o', ms='4', animated=True)
                self.axes.add_line(line)
            if pt != self.current[-1]:
                if pt != self.current[0]:
                    self.current.append(pt)
                else:
                    self.items.append(self.current)
                    self._save_items()
                    self.canvas.restore_region(self.background)
                    line = self.axes.lines[-1]
                    line.set_color('#30a2da')
                    line.set_marker(None)
                    self._draw_current(*self.tr.transform(pt))
                    self.background = self.canvas.copy_from_bbox(self.axes.bbox)
                    self.current = None
                    self.canvas.blit(self.axes.bbox)
        elif event.button == 3:
            if self.current is not None:
                if len(self.current) == 1:
                    self.current = None
                    del self.axes.lines[-1]
                    self.canvas.restore_region(self.background)
                    self.canvas.blit(self.axes.bbox)
                else:
                    del self.current[-1]
                    self.motion_notify_callback(event)
            elif event.dblclick == True:
                pt = self._get_node(event)
                x, y = self.axes.transData.transform(self.tr.transform(pt))
                event = MouseEvent('Lattice MouseEvent', self.canvas, x, y)
                for i, line in reversed(list(enumerate(self.axes.lines[1:]))):
                    if line.contains(event)[0]:
                        del self.items[i]
                        self._save_items()
                        del self.axes.lines[i + 1]
                        self.canvas.draw()
                        break

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.current is not None:
                self.current = None
                del self.axes.lines[-1]
                self.canvas.restore_region(self.background)
                self.canvas.blit(self.axes.bbox)

        elif event.key() == Qt.Key_U: self._undo()
        elif event.key() == Qt.Key_R: self._redo()

    def motion_notify_callback(self, event):
        x, y = self.tr.transform(self._get_node(event))
        if np.isnan(x) or np.isnan(y):
            self.mark.set_visible(False)
            return

        self.canvas.restore_region(self.background)

        if event.inaxes is None:
            self.mark.set_visible(False)
        else:
            self.mark.set_xdata([x])
            self.mark.set_ydata([y])
            self.mark.set_visible(True)
            self.axes.draw_artist(self.mark)

        self._draw_current(x, y)

        self.canvas.blit(self.axes.bbox)

