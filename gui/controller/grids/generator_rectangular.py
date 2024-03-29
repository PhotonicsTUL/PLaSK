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
import weakref

from . import GridController
from ..defines import DefinesCompletionDelegate, get_defines_completer
from ...model.geometry.geometry import GNGeometryBase
from ..table import table_with_manipulators
from ...model.grids.generator_rectangular import RectangularDivideGenerator
from ...model.grids.mesh_rectangular import AXIS_NAMES
from ...qt.QtCore import Qt
from ...qt.QtWidgets import *
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...utils.widgets import ComboBoxDelegate, EditComboBox


class RectangularSimpleGeneratorController(GridController):

    def __init__(self, document, model):
        super().__init__(
            document=document, model=model)

        self.form = QGroupBox()
        form_layout = QHBoxLayout()

        label = QLabel(" Split Boundaries:")
        form_layout.addWidget(label)
        self.make_split_combo(form_layout, weakref.proxy(self))

        form_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.form.setLayout(form_layout)

    def make_split_combo(self, form_layout, weakself):
        self.split = EditComboBox()
        self.split.editingFinished.connect(
            lambda: weakself._change_attr('split', empty_to_none(self.split.currentText()),
                                          'mesh split on object boundaries'))
        self.split.addItems(['', 'yes', 'no'])
        self.split.setEditable(True)
        self.split.setToolTip('&lt;boundaries <b>split</b>=""&gt;<br/>Split mesh lines at object boundaries '
                              '(only useful for plotting material parameters).')
        self.split.lineEdit().setPlaceholderText('no')
        form_layout.addWidget(self.split)

    def fill_form(self):
        super().fill_form()
        with BlockQtSignals(self.split):
            self.split.setEditText(none_to_empty(self.grid_model.split))

    def get_widget(self):
        return self.form

    def select_info(self, info):
        super().select_info(info)
        getattr(self, info.property).setFocus()


class RectangularRegularGeneratorController(RectangularSimpleGeneratorController):

    LABELS = (('',),
              ('  horizontal:', '  vertical:'),
              ('  longitudinal:', '  transverse:', '  vertical:'))

    def __init__(self, document, model):
        super().__init__(
            document=document, model=model)

        self.form = QGroupBox()
        form_layout = QHBoxLayout()
        spacing_label = QLabel(u"Spacing (µm):")
        form_layout.addWidget(spacing_label)

        self.defines = get_defines_completer(
            self.document.defines.model, self.form)

        weakself = weakref.proxy(self)

        dim = model.dim
        for i in range(dim):
            label = QLabel(self.LABELS[dim-1][i])
            form_layout.addWidget(label)
            attr = 'spacing{}'.format(i)
            edit = QLineEdit()
            edit.editingFinished.connect(
                lambda attr=attr: weakself._change_attr(attr, empty_to_none(getattr(weakself, attr).text())))
            edit.setCompleter(self.defines)
            edit.setToolTip('&lt;spacing <b>every{}</b>=""&gt;<br/>Approximate single element size.'
                            .format('' if dim == 1 else str(i)))
            form_layout.addWidget(edit)
            setattr(self, attr, edit)

        label = QLabel(" Split Boundaries:")
        form_layout.addWidget(label)
        self.make_split_combo(form_layout, weakself)

        self.form.setLayout(form_layout)

    def fill_form(self):
        super().fill_form()
        for i in range(self.grid_model.dim):
            attr = 'spacing{}'.format(i)
            getattr(self, attr).setText(
                none_to_empty(getattr(self.grid_model, attr)))


class RectangularRefinedGeneratorController(GridController):

    def _make_param_hbox(self, container_to_add, label, tooltip, defines_completer, model_path=None):
        hbox_div = QHBoxLayout()
        res = tuple(QLineEdit() for _ in range(0, self.model.dim))
        for i, r in enumerate(res):
            if self.model.dim != 1:
                axis_name = AXIS_NAMES[self.model.dim-1][i]
                hbox_div.addWidget(
                    QLabel('{}:'.format(axis_name if axis_name else str(i))))
            else:
                axis_name = 'horizontal'
            hbox_div.addWidget(r)
            r.setToolTip(
                tooltip.format('{}'.format(i), ' in {} direction'.format(axis_name) if axis_name else ''))
            r.setCompleter(defines_completer)
            if model_path is None:
                r.editingFinished.connect(self.fire_changed)
            else:
                weakself = weakref.proxy(self)
                r.editingFinished.connect(
                    lambda i=i, r=r: weakself._change_attr((model_path, i), empty_to_none(r.text())))
        container_to_add.addRow(label, hbox_div)
        return res

    def __init__(self, document, model):
        super().__init__(
            document=document, model=model)

        self.form = QGroupBox()

        self.defines = get_defines_completer(
            self.document.defines.model, self.form)

        vbox = QVBoxLayout()

        self.options = QHBoxLayout()

        weakself = weakref.proxy(self)

        self.aspect = QLineEdit()
        self.aspect.editingFinished.connect(
            lambda: weakself._change_attr('aspect', empty_to_none(weakself.aspect.text())))
        self.aspect.setCompleter(self.defines)
        self.aspect.setToolTip('&lt;options <b>aspect</b>=""&gt;<br/>'
                               'Maximum aspect ratio for the rectangular and cubic elements generated '
                               'by this generator.')
        self.options.addWidget(QLabel("aspect:"))
        self.options.addWidget(self.aspect)

        self.form_layout = QFormLayout()
        self.form_layout.addRow('Options:', self.options)

        vbox.addLayout(self.form_layout)

        self.refinements = QTableView()
        self.refinements.setModel(model.refinements)
        one = int(self.model.dim == 1)
        if not one:
            self.refinements.setItemDelegateForColumn(0, ComboBoxDelegate(AXIS_NAMES[self.model.dim-1],
                                                                          self.refinements, editable=False))

        def object_names():
            return document.geometry.model.get_names(filter=lambda x: not isinstance(x, GNGeometryBase))
        self.refinements.setItemDelegateForColumn(1-one, ComboBoxDelegate(object_names,
                                                                          self.refinements, editable=True))
        try:
            paths = document.geometry.model.get_paths()
        except AttributeError:
            pass
        else:
            self.refinements.setItemDelegateForColumn(2-one, ComboBoxDelegate(paths,
                                                                              self.refinements, editable=True))
        defines_delegate = DefinesCompletionDelegate(
            document.defines.model, self.refinements)
        self.refinements.setItemDelegateForColumn(3-one, defines_delegate)
        self.refinements.setItemDelegateForColumn(4-one, defines_delegate)
        self.refinements.setItemDelegateForColumn(5-one, defines_delegate)
        # self.refinements.horizontalHeader().setResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.refinements.setColumnWidth(1-one, 140)
        self.refinements.setColumnWidth(2-one, 120)
        self.refinements.setMinimumHeight(100)
        vbox.addWidget(table_with_manipulators(self.refinements,
                                               title='Refinements', add_undo_action=False))

        # vbox.addStretch()
        self.form.setLayout(vbox)

    def fill_form(self):
        super().fill_form()
        self.aspect.setText(none_to_empty(self.grid_model.aspect))

    def get_widget(self):
        return self.form

    def select_info(self, info):
        super().select_info(info)
        try:
            if info.property == 'refinements':  # select refinements index using info.reinf_row and info.reinf_col
                self.refinements.setCurrentIndex(
                    self.model.refinements.createIndex(info.reinf_row, info.reinf_col))
        except AttributeError:
            pass


class RectangularDivideGeneratorController(RectangularRefinedGeneratorController):
    """Ordered and rectangular 2D and 3D divide generator script."""

    def __init__(self, document, model):
        super().__init__(
            document=document, model=model)

        weakself = weakref.proxy(self)

        dim = model.dim
        self.gradual = []
        self.options.insertWidget(0, QLabel("gradual axis:"))
        if dim != 1:
            line = QFrame()
            line.setFrameShape(QFrame.Shape.VLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            line.setLineWidth(1)
            self.options.insertWidget(1, line)
        for i in range(dim-1, -1, -1):
            gradual = EditComboBox()
            gradual.editingFinished.connect(
                lambda i=i: weakself._change_attr(('gradual', i), empty_to_none(weakself.gradual[i].currentText())))
            axis_name = AXIS_NAMES[dim-1][i]
            gradual.addItems(['', 'yes', 'no'])
            gradual.setEditable(True)
            gradual.setMinimumWidth(150)
            gradual.setToolTip('&lt;options <b>gradual{}</b>=""&gt;<br/>'
                               'Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated'
                               ' mesh may differ more than by the factor of two) for {}axis. Gradual is enabled by default.'
                               .format('' if dim == 1 else str(i), axis_name + (' ' if axis_name else '')))
            gradual.lineEdit().setPlaceholderText('yes')
            if axis_name:
                self.options.insertWidget(1, QLabel(axis_name + ':'))
                self.options.insertWidget(2, gradual)
            else:
                self.options.insertWidget(1, gradual)
            self.gradual.insert(0, gradual)

        self.prediv = self._make_param_hbox(self.form_layout, 'Pre-refining divisions:',
                                            '&lt;<b>prediv by{}</b>=""&gt;<br/>'
                                            'The number of the initial divisions of each geometry object{}.',
                                            self.defines, 'prediv')

        self.postdiv = self._make_param_hbox(self.form_layout, 'Post-refining divisions:',
                                             '&lt;<b>postdiv by{}</b>=""&gt;<br/>'
                                             'The number of the final divisions of each geometry object{}.',
                                             self.defines, 'postdiv')

    def fill_form(self):
        super().fill_form()
        with self.mute_changes():
            for i, gradual in enumerate(self.gradual):
                with BlockQtSignals(gradual):
                    gradual.setEditText(none_to_empty(self.model.gradual[i]))
            for i in range(0, self.model.dim):
                self.prediv[i].setText(self.model.prediv[i])
                self.postdiv[i].setText(self.model.postdiv[i])

    # def save_data_in_model(self):
    #    super().save_data_in_model()
    #    self.model.prediv = [empty_to_none(self.prediv[i].text()) for i in range(0, self.model.dim)]
    #    self.model.postdiv = [empty_to_none(self.postdiv[i].text()) for i in range(0, self.model.dim)]

    # def on_edit_enter(self):
    #    super().on_edit_enter()
    #    with self.mute_changes():
    #        for i in range(0, self.model.dim):
    #            self.prediv[i].setText(self.model.prediv[i])
    #            self.postdiv[i].setText(self.model.postdiv[i])


class RectangularSmoothGeneratorController(RectangularRefinedGeneratorController):
    """Ordered and rectangular 2D and 3D divide generator script."""

    def __init__(self, document, model):
        super().__init__(
            document=document, model=model)

        self.small = self._make_param_hbox(self.form_layout, 'Smallest element:',
                                           '&lt;<b>step small{}</b>=""&gt;<br/>'
                                           'The size of the smallest element near each object edge{}.',
                                           self.defines, 'small')
        self.large = self._make_param_hbox(self.form_layout, 'Largest element:',
                                           '&lt;<b>step large{}</b>=""&gt;<br/>'
                                           'Maximum size of mesh elements{}.',
                                           self.defines, 'large')
        self.factor = self._make_param_hbox(self.form_layout, 'Increase factor:',
                                            '&lt;<b>step factor{}</b>=""&gt;<br/>'
                                            'Increase factor for element sizes further from object edges{}.',
                                            self.defines, 'factor')

    # def save_data_in_model(self):
    #    super().save_data_in_model()
    #    self.model.small = [empty_to_none(self.small[i].text()) for i in range(0, self.model.dim)]
    #    self.model.large = [empty_to_none(self.large[i].text()) for i in range(0, self.model.dim)]
    #    self.model.factor = [empty_to_none(self.factor[i].text()) for i in range(0, self.model.dim)]

    def fill_form(self):
        with self.mute_changes():
            super().fill_form()
            for i in range(0, self.model.dim):
                self.small[i].setText(self.model.small[i])
                self.large[i].setText(self.model.large[i])
                self.factor[i].setText(self.model.factor[i])
