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

from . import GridController
from ..defines import DefinesCompletionDelegate, get_defines_completer
from ...model.geometry.geometry import GNGeometryBase
from ..table import table_with_manipulators
from ...model.grids.generator_rectilinear import RectilinearDivideGenerator
from ...model.grids.mesh_rectilinear import AXIS_NAMES
from ...qt import QtGui
from ...utils.str import empty_to_none
from ...utils.widgets import ComboBoxDelegate, ComboBox


class RectilinearRefinedGeneratorController(GridController):

    warnings_help = {
        'missing': 'Warn if any refinement references to non-existing object. Defaults to true.',
        'multiple': 'Warn if any refinement references to multiple objects. Defaults to true.',
        'outside': 'Warn if refining line lies outside of the specified object. Defaults to true.'
    }

    def _make_param_hbox(self, container_to_add, label, tooltip, defines_completer):
        hbox_div = QtGui.QHBoxLayout()
        res = tuple(QtGui.QLineEdit() for _ in range(0, self.model.dim))
        for i,r in enumerate(res):
            if self.model.dim != 1:
                axis_name = AXIS_NAMES[self.model.dim-1][i]
                hbox_div.addWidget(QtGui.QLabel('{}:'.format(axis_name if axis_name else str(i))))
            else:
                axis_name = 'horizontal'
            hbox_div.addWidget(r)
            r.setToolTip(
                tooltip.format('{}'.format(i), ' in {} direction'.format(axis_name) if axis_name else ''))
            r.setCompleter(defines_completer)
            r.editingFinished.connect(self.fire_changed)
        container_to_add.addRow(label, hbox_div)
        return res

    def __init__(self, document, model):
        super(RectilinearRefinedGeneratorController, self).__init__(document=document, model=model)

        self.form = QtGui.QGroupBox()

        self.defines = get_defines_completer(self.document.defines.model, self.form)

        vbox = QtGui.QVBoxLayout()

        self.options = QtGui.QHBoxLayout()

        self.aspect = QtGui.QLineEdit()
        self.aspect.editingFinished.connect(lambda : self._change_attr('aspect', self.aspect.text()))   #self.fire_changed
        self.aspect.setCompleter(self.defines)
        self.aspect.setToolTip('&lt;options <b>aspect</b>=""&gt;<br/>'
                               'Maximum aspect ratio for the rectangular and cubic elements generated '
                               'by this generator.')
        self.options.addWidget(QtGui.QLabel("aspect:"))
        self.options.addWidget(self.aspect)

        self.form_layout = QtGui.QFormLayout()
        self.form_layout.addRow('Options:', self.options)

        warnings_layout = QtGui.QHBoxLayout()
        for w in RectilinearDivideGenerator.warnings:
            cb = ComboBox()
            cb.editingFinished.connect(self.fire_changed)
            cb.currentIndexChanged.connect(self.fire_changed)
            cb.addItems(['', 'yes', 'no'])
            cb.setEditable(True)
            cb.setToolTip('&lt;warnings <b>{}</b>=""&gt;\n'.format(w) +
                          RectilinearDivideGeneratorController.warnings_help.get(w, ''))
            setattr(self, 'warn_'+w, cb)
            label = QtGui.QLabel(w+':')
            label.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Preferred)
            warnings_layout.addWidget(label)
            cb.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
            warnings_layout.addWidget(cb)
        self.form_layout.addRow('Warnings:', warnings_layout)

        vbox.addLayout(self.form_layout)

        self.refinements = QtGui.QTableView()
        self.refinements.setModel(model.refinements)
        one = int(self.model.dim == 1)
        if one:
            self.refinements.setItemDelegateForColumn(0, ComboBoxDelegate(AXIS_NAMES[self.model.dim-1],
                                                                          self.refinements, editable=False))
        def object_names():
            return self.document.geometry.model.names(filter=lambda x: not isinstance(x, GNGeometryBase))
        self.refinements.setItemDelegateForColumn(1-one, ComboBoxDelegate(object_names,
                                                                          self.refinements, editable=True))
        self.refinements.setItemDelegateForColumn(2-one, ComboBoxDelegate(self.document.geometry.model.paths,
                                                                          self.refinements, editable=True))
        defines_delegate = DefinesCompletionDelegate(self.document.defines.model, self.refinements)
        self.refinements.setItemDelegateForColumn(3-one, defines_delegate)
        self.refinements.setItemDelegateForColumn(4-one, defines_delegate)
        self.refinements.setItemDelegateForColumn(5-one, defines_delegate)
        # self.refinements.horizontalHeader().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        self.refinements.setColumnWidth(1-one, 140)
        self.refinements.setColumnWidth(2-one, 120)
        self.refinements.setMinimumHeight(100)
        vbox.addWidget(table_with_manipulators(self.refinements, title='Refinements', add_undo_action=False))

        #vbox.addStretch()
        self.form.setLayout(vbox)

    def fill_form(self):
        self.aspect.setText(self.grid_model.aspect)

    def save_data_in_model(self):
        for attr_name in ['warn_'+w for w in RectilinearDivideGenerator.warnings]:
            setattr(self.model, attr_name, empty_to_none(getattr(self, attr_name).currentText()))
        #self.model.aspect = self.aspect.text()

    def on_edit_enter(self):
        super(RectilinearRefinedGeneratorController, self).on_edit_enter()
        with self.mute_changes():
            for attr_name in ['warn_'+w for w in RectilinearDivideGenerator.warnings]:
                a = getattr(self.model, attr_name)
                getattr(self, attr_name).setEditText('' if a is None else a)

    def get_widget(self):
        return self.form


class RectilinearDivideGeneratorController(RectilinearRefinedGeneratorController):
    """Ordered and rectangular 2D and 3D divide generator script."""

    def __init__(self, document, model):
        super(RectilinearDivideGeneratorController, self).__init__(document=document, model=model)

        self.gradual = ComboBox()
        self.gradual.editingFinished.connect(self.fire_changed)
        self.gradual.currentIndexChanged.connect(self.fire_changed)
        self.gradual.addItems(['', 'yes', 'no'])
        self.gradual.setEditable(True)
        self.gradual.setMinimumWidth(150)
        self.gradual.setToolTip('&lt;options <b>gradual</b>=""&gt;<br/>'
                                'Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated'
                                ' mesh may differ more than by the factor of two). Gradual is enabled by default.')
        self.options.insertWidget(0, QtGui.QLabel("gradual:"))
        self.options.insertWidget(1, self.gradual)

        self.prediv = self._make_param_hbox(self.form_layout, 'Pre-refining divisions:',
                                            '&lt;<b>prediv by{}</b>=""&gt;<br/>'
                                            'The number of the initial divisions of each geometry object{}.',
                                            self.defines)
        self.postdiv = self._make_param_hbox(self.form_layout, 'Post-refining divisions:',
                                             '&lt;<b>postdiv by{}</b>=""&gt;<br/>'
                                             'The number of the final divisions of each geometry object{}.',
                                             self.defines)

    def save_data_in_model(self):
        super(RectilinearDivideGeneratorController, self).save_data_in_model()
        self.model.gradual = empty_to_none(self.gradual.currentText())
        self.model.set_prediv([empty_to_none(self.prediv[i].text()) for i in range(0, self.model.dim)])
        self.model.set_postdiv([empty_to_none(self.postdiv[i].text()) for i in range(0, self.model.dim)])

    def on_edit_enter(self):
        with self.mute_changes():
            super(RectilinearDivideGeneratorController, self).on_edit_enter()
            self.gradual.setEditText('' if self.model.gradual is None else self.model.gradual)
            for i in range(0, self.model.dim):
                self.prediv[i].setText(self.model.get_prediv(i))
                self.postdiv[i].setText(self.model.get_postdiv(i))


class RectilinearSmoothGeneratorController(RectilinearRefinedGeneratorController):
    """Ordered and rectangular 2D and 3D divide generator script."""

    def __init__(self, document, model):
        super(RectilinearSmoothGeneratorController, self).__init__(document=document, model=model)

        self.small = self._make_param_hbox(self.form_layout, 'Smallest element:',
                                           '&lt;<b>step small{}</b>=""&gt;<br/>'
                                           'The size of the smallest element near each object edge{}.',
                                           self.defines)
        self.large = self._make_param_hbox(self.form_layout, 'Largest element:',
                                           '&lt;<b>step large{}</b>=""&gt;<br/>'
                                           'Maximum size of mesh elements{}.',
                                           self.defines)
        self.factor = self._make_param_hbox(self.form_layout, 'Increase factor:',
                                            '&lt;<b>step factor{}</b>=""&gt;<br/>'
                                            'Increase factor for element sizes further from object edges{}.',
                                            self.defines)

    def save_data_in_model(self):
        super(RectilinearSmoothGeneratorController, self).save_data_in_model()
        self.model.set_small([empty_to_none(self.small[i].text()) for i in range(0, self.model.dim)])
        self.model.set_large([empty_to_none(self.large[i].text()) for i in range(0, self.model.dim)])
        self.model.set_factor([empty_to_none(self.factor[i].text()) for i in range(0, self.model.dim)])

    def on_edit_enter(self):
        with self.mute_changes():
            super(RectilinearSmoothGeneratorController, self).on_edit_enter()
            for i in range(0, self.model.dim):
                self.small[i].setText(self.model.get_small(i))
                self.large[i].setText(self.model.get_large(i))
                self.factor[i].setText(self.model.get_factor(i))
