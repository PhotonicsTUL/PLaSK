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
from ...utils.config import CONFIG


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

        bounds = []
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
                    bounds.append(item)

        dialog = LatticeEditor(vecs, bounds)
        if dialog.exec_():
            segments = ' ^ '.join('; '.join('{:.0f} {:.0f}'.format(x+.001, y+.001) for (x,y) in item)
                                  for item in dialog.bounds)
            print(segments)
            self._set_node_property_undoable('segments', segments)


class Cursors(object):
    # this class is only used as a simple namespace
    HAND, POINTER, SELECT_REGION, MOVE = range(4)
cursors = Cursors()


class NavigationToolbar(NavigationToolbar2QT):

    def __init__(self, canvas, parent, coordinates=True):
        self.widgets = {}
        super(NavigationToolbar, self).__init__(canvas, parent, coordinates)

    def _icon(self, name):
        if name is not None:
            return QtGui.QIcon.fromTheme(name)

    HELP = \
        "This is a graphical lattice boundaries editor. Click any lattice node\n" \
        "in order to create a polygon defining a lattice boundary. Click consecutive\n" \
        "polygon points and to finish just close the loop. You can remove existing\n" \
        "polygons by double-clicking right mouse button on any of its vertices.\n" \
        "The lattice will include all lattice points within each polygon (including\n" \
        "boundaries). By nesting polygons within each other you can create holes,\n" \
        "however the boundaries of each polygon will be included into the final lattice.\n\n" \
        "Undo/redo buttons on the toolbar allow you to revert wrong editing or deletion."

    toolbounds = (
        ('Undo', 'Undo previous line edit', 'edit-undo', 'undo', None),
        ('Redo', 'Redo line edit', 'edit-redo', 'redo', None),
        (None, None, None, None, None),
        ('Home', 'Zoom to whole geometry', 'go-home', 'home', None),
        ('Back', 'Back to previous view', 'go-previous', 'back', None),
        ('Forward', 'Forward to next view', 'go-next', 'forward', None),
        (None, None, None, None, None),
        ('Pan', 'Pan aHelp goes herexes with left mouse, zoom with right', 'transform-move', 'pan', False),
        ('Zoom', 'Zoom to rectangle', 'zoom-in', 'zoom', False),
        (None, None, None, None, None),
        ('Help', HELP, 'help-contents', 'help', None),
    )

    def _init_toolbar(self):
        self.layout().setContentsMargins(0,0,0,0)
        for text, tooltip_text, icon, callback, checked in self.toolbounds:
            if text is None:
                self.addSeparator()
            elif callback is None:
                self.addWidget(QtGui.QLabel(text))
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
                self._actions[callback] = action
        self.buttons = {}
        self._actions['undo'].setEnabled(False)
        self._actions['redo'].setEnabled(False)
        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtGui.QLabel("", self)
            self.locLabel.setAlignment(Qt.AlignRight | Qt.AlignTop)
            self.locLabel.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored))
            label_action = self.addWidget(self.locLabel)
            label_action.setVisible(True)

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
            x, y = self.parent.get_node(event)
            s = u'{0:.0f}, {1:.0f}'.format(x+.001, y+.001)
        else:
            s = ''

        if len(self.mode):
            self.set_message('%s   %s' % (s, self.mode))
        else:
            self.set_message(s)

    def clear_history(self):
        self._views.clear()
        self._positions.clear()

    def set_history_buttons(self):
        if len(self._views) <= 1:
            self._actions['back'].setEnabled(False)
            self._actions['forward'].setEnabled(False)
        elif self._views._pos == 0:
            self._actions['back'].setEnabled(False)
            self._actions['forward'].setEnabled(True)
        elif self._views._pos == len(self._views)-1:
            self._actions['back'].setEnabled(True)
            self._actions['forward'].setEnabled(False)
        else:
            self._actions['back'].setEnabled(True)
            self._actions['forward'].setEnabled(True)

    def edit_mode(self):
        return self._active is None

    def set_undo_buttons(self):
        self._actions['undo'].setEnabled(self.parent.undo_index != 0)
        self._actions['redo'].setEnabled(self.parent.undo_index < len(self.parent.undo_stack) - 1)

    def undo(self):
        self.parent.undo()

    def redo(self):
        self.parent.redo()

    def help(self):
        QtGui.QToolTip.showText(QtGui.QCursor.pos(), self.HELP)


class LatticeEditor(QtGui.QDialog):

    class MultipleLocator(ticker.MultipleLocator):
        def __init__(self, base=1.0):
            super(LatticeEditor.MultipleLocator, self).__init__(base)
            self.create_dummy_axis()
        def __call__(self, v1, v2):
            self.set_bounds(v1, v2)
            locs = super(LatticeEditor.MultipleLocator, self).__call__()
            return np.array(locs), len(locs), None

    def __init__(self, vecs, bounds=None, parent=None):
        super(LatticeEditor, self).__init__(parent)
        self.setWindowTitle("Edit Lattice Boundaries")
        vbox = QtGui.QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)
        #self.canvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.figure.set_facecolor(self.palette().color(QtGui.QPalette.Background).name())
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
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
        loc = self.MultipleLocator(1)
        grid_helper = GridHelperCurveLinear(self.tr, grid_locator1=loc, grid_locator2=loc)
        self.axes = Subplot(self.figure, 1, 1, 1, grid_helper=grid_helper, adjustable='datalim')
        self.figure.add_subplot(self.axes)
        self.axes.set_aspect(1.)
        self.axes.axis['t1'] = self.axes.new_floating_axis(0, 0.)
        self.axes.axis['t2'] = self.axes.new_floating_axis(1, 0.)
        self.axes.grid(True)

        if bounds:
            cc1, cc2 = zip(*itertools.chain(*bounds))
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
        self.mark = Line2D([0], [0], marker='o', mfc=CONFIG['geometry/lattice_mark_color'], ms=6.,
                           animated=True)
        self.mark.set_visible(False)
        self.points = Line2D([], [], marker='o', mfc=CONFIG['geometry/lattice_line_color'], ms=8., alpha=0.5,
                             zorder=0, animated=True)
        self.canvas.draw()
        self.bounds = [] if bounds is None else bounds
        self._set_lines()
        self.current = None
        self.undo_stack = [list(self.bounds)]
        self.undo_index = 0

    def _set_lines(self):
        self.axes.lines = []
        self.axes.add_line(self.mark)
        self.axes.add_line(self.points)
        for item in self.bounds:
            item = [self.tr.transform(p) for p in item]
            xx = [p[0] for p in item] + [item[0][0]]
            yy = [p[1] for p in item] + [item[0][1]]
            line = Line2D(xx, yy, ls='-', color=CONFIG['geometry/lattice_line_color'], lw=2., animated=True)
            self.axes.add_line(line)

    def get_node(self, event):
        return tuple(round(c) for c in self.tri.transform([event.xdata, event.ydata]))

    def draw_callback(self, event=None):
        self.mark.set_visible(False)
        self.update_points()
        for line in self.axes.lines[1:]:
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

    def _save_bounds(self):
        self.undo_index += 1
        self.undo_stack = self.undo_stack[:self.undo_index]
        self.undo_stack.append(list(self.bounds))
        self.toolbar.set_undo_buttons()

    def undo(self):
        if self.undo_index > 0:
            self.undo_index -= 1
            self.bounds = list(self.undo_stack[self.undo_index])
            self._set_lines()
            self.toolbar.set_undo_buttons()
            self.canvas.draw()

    def redo(self):
        if self.undo_index < len(self.undo_stack)-1:
            self.undo_index += 1
            self.bounds = list(self.undo_stack[self.undo_index])
            self._set_lines()
            self.toolbar.set_undo_buttons()
            self.canvas.draw()

    def button_press_callback(self, event):
        if not self.toolbar.edit_mode() or event.inaxes is None: return
        if event.button == 1:
            pt = self.get_node(event)
            if self.current is None:
                #self.background = self.canvas.copy_from_bbox(self.axes.bbox)
                self.current = [pt]
                x, y = self.tr.transform(pt)
                line = Line2D([x], [y], ls='-', color=CONFIG['geometry/lattice_active_color'], lw=2.,
                              marker='o', ms='4', animated=True)
                self.axes.add_line(line)
            if pt != self.current[-1]:
                if pt != self.current[0]:
                    self.current.append(pt)
                else:
                    self.bounds.append(self.current)
                    self._save_bounds()
                    self.canvas.restore_region(self.background)
                    line = self.axes.lines[-1]
                    line.set_color(CONFIG['geometry/lattice_line_color'])
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
                pt = self.get_node(event)
                x, y = self.axes.transData.transform(self.tr.transform(pt))
                event = MouseEvent('Lattice MouseEvent', self.canvas, x, y)
                for i, line in reversed(list(enumerate(self.axes.lines[2:]))):
                    if line.contains(event)[0]:
                        del self.bounds[i]
                        del self.axes.lines[i+2]
                        self._save_bounds()
                        self.update_points()
                        self.canvas.draw()
                        break

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.current is not None:
                self.current = None
                del self.axes.lines[-1]
                self.canvas.restore_region(self.background)
                self.canvas.blit(self.axes.bbox)
        else:
            super(LatticeEditor, self).keyPressEvent(event)

    def motion_notify_callback(self, event):
        x, y = self.tr.transform(self.get_node(event))
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

    def update_points(self):
        pass