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
import sys
import weakref
from lxml import etree
from bisect import bisect

from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *
from ...model.info import Info
from ...model.geometry.reader import axes_as_list
from ...model.materials import HandleMaterialsModule
from ...utils import get_manager
from ...utils.config import CONFIG
from ..source import SourceEditController

try:
    import plask
except ImportError:
    plask = None
else:
    from .plot_widget import PlotWidget as GeometryPlotWidget

    class PlotWidget(GeometryPlotWidget):

        class NavigationToolbar(GeometryPlotWidget.NavigationToolbar):

            toolitems = (
                ('Plot', 'Plot selected geometry object', 'draw-brush', 'plot', None, 'plot_plot'),
                ('Refresh', 'Refresh plot after each change of geometry', 'view-refresh', 'auto_refresh', True, 'plot_refresh'),
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

            def select_plane(self, index):
                super().select_plane(index)
                self.controller.plot()


class SourceWithPreviewWidget(QSplitter):

    def __init__(self, controller, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.controller = weakref.proxy(controller)

    @property
    def editor(self):
        return self.controller.source.editor


class GeometrySourceController(SourceEditController):

    if plask is not None:
        def __init__(self, document=None, model=None, line_numbers=True):
            super().__init__(document, model, line_numbers)
            self.manager = None
            self.checked_plane = '12'
            self.timer = QTimer()
            self.timer.setSingleShot(True)
            self.timer.setInterval(2000)
            self.timer.timeout.connect(self.update_preview)
            self.plotted_tree_element = None  # we do not use this, but it toolbar base class checks it

        def create_source_widget(self, parent):
            splitter = SourceWithPreviewWidget(self, parent)
            self.source = super().create_source_widget(splitter)
            self.source.toolbar.addSeparator()
            preview_action = self.source.add_action('&Preview', 'preview', None, self.show_geometry_preview)
            preview_action.setCheckable(True)
            preview_action.setChecked(True)
            self.geometry_view = PlotWidget(self, splitter)
            self.geometry_view.toolbar.enable_planes()
            splitter.setStretchFactor(0, 5)
            splitter.setStretchFactor(1, 2)
            self.plot_auto_refresh = True
            return splitter

        def on_edit_enter(self):
            super().on_edit_enter()
            self.source.editor.textChanged.connect(self.text_changed)
            self.last_index = None
            self._index = None
            self._elements = None
            self.plot()
            self.last_line = None
            self.cursor_moved()
            self.source.editor.cursorPositionChanged.connect(self.cursor_moved)

        def on_edit_exit(self):
            self.manager = None
            try: self.source.editor.cursorPositionChanged.disconnect(self.cursor_moved)
            except: pass
            try: self.source.editor.textChanged.disconnect(self.text_changed)
            except: pass
            return super().on_edit_exit()

    def show_geometry_preview(self, checked):
        self.geometry_view.setVisible(checked)

    def cursor_moved(self):
        if not self.geometry_view.isVisible(): return
        current_line = self.source.editor.textCursor().blockNumber()
        if current_line != self.last_line:
            self.last_line = current_line
            try:
                if self._elements is None:
                    self._elements = [e for e in etree.fromstring(
                        "<plask><geometry>\n" + self.source.editor.toPlainText() + "\n</geometry></plask>")[0]
                                      if e.tag is not etree.Comment]
                line_numbers = [e.sourceline-2 for e in self._elements[1:]]
                index = bisect(line_numbers, current_line)
                try:
                    axes = axes_as_list(self._elements[index].attrib.get('axes'))
                    if self._elements[index].tag.lower() == 'cartesian3d':
                        self.geometry_view.toolbar.enable_planes(axes)
                    else:
                        self.geometry_view.toolbar.disable_planes(axes)
                except IndexError:
                    pass
            except:
                pass
            else:
                if index != self.last_index:
                    self._index = index
                    self.timer.start()

    def text_changed(self):
        if not self.geometry_view.isVisible(): return
        self.last_index = self._index = self._elements = None
        self.timer.start()

    def zoom_to_root(self):
        if self.plotted_object is not None:
            box = self.plotted_object.bbox
            self.geometry_view.zoom_bbox(box)

    def plot_current_element(self, set_limits=False):
        if not self.geometry_view.isVisible(): return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                if self.manager is None:
                    manager = get_manager()
                    with HandleMaterialsModule(self.document):
                        manager.load(self.document.get_contents(sections=('defines', 'materials')))
                else:
                    manager = self.manager
                    manager.geo.clear()
                    manager.pth.clear()
                    manager._roots.clear()
                text = "<plask><geometry>\n" + self.source.editor.toPlainText() + "\n</geometry></plask>"
                manager.load("\n"*(self.model.line_in_file-1) + text)
                self.manager = manager
                if self._elements is None:
                    self._elements = [e for e in etree.fromstring(text)[0] if e.tag is not etree.Comment]
                if self._index is None:
                    line_numbers = [e.sourceline-2 for e in self._elements[1:]]
                    current_line = self.source.editor.textCursor().blockNumber()
                    index = bisect(line_numbers, current_line)
                else:
                    index = self._index
                self.plotted_object = manager._roots[index]
                self.geometry_view.update_plot(self.plotted_object, set_limits=set_limits, plane=self.checked_plane)
                self.last_index = index
            except plask.XMLError as e:
                self.model.info_message("Could not update geometry preview: {}".format(str(e)), Info.WARNING, line=e.line)
                from ... import _DEBUG
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
                    sys.stderr.flush()
                return False
            except Exception as e:
                self.model.info_message("Could not update geometry preview: {}".format(str(e)), Info.WARNING)
                from ... import _DEBUG
                if _DEBUG:
                    import traceback
                    traceback.print_exc()
                    sys.stderr.flush()
                return False
            else:
                self.model.info_message()
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.processEvents()

    def update_preview(self):
        if self.geometry_view.isVisible() and self.plot_auto_refresh:
            self.plot_current_element(self._index != self.last_index)

    def plot(self):
        self.timer.stop()
        self.plot_current_element(True)

    def select_info(self, info):
        try:
            line = info.line
        except AttributeError:
            pass
        else:
            self.document.window.goto_line(line)

    def reconfig(self):
        super().reconfig()
        if self.geometry_view is not None:
            colors = CONFIG['geometry/material_colors'].copy()
            self.geometry_view.get_color = plask.ColorFromDict(colors, self.geometry_view.axes)
            self.geometry_view.axes.set_facecolor(CONFIG['plots/face_color'])
            self.geometry_view.axes.grid(True, color=CONFIG['plots/grid_color'])
            self.plot_current_element()
