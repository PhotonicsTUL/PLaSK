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

from . import GridController
from ..defines import DefinesCompletionDelegate, get_defines_completer
from ...model.geometry.geometry import GNGeometryBase
from ..table import table_with_manipulators
from ...model.grids.generator_rectangular import RectangularDivideGenerator
from ...model.grids.mesh_rectangular import AXIS_NAMES
from ...qt.QtWidgets import *
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty
from ...utils.widgets import ComboBoxDelegate, ComboBox


class RectangularRegularGeneratorController(GridController):

    LABELS = (('',),
              ('  horizontal:', '  vertical:'),
              ('  longitudinal:', '  transverse:', '  vertical:'))

    def __init__(self, document, model):
        super(RectangularRegularGeneratorController, self).__init__(document=document, model=model)

        self.form = QGroupBox()
        form_layout = QHBoxLayout()
        spacing_label = QLabel(u"Spacing [µm]:")
        form_layout.addWidget(spacing_label)

        self.defines = get_defines_completer(self.document.defines.model, self.form)

        dim = model.dim
        for i in range(dim):
            label = QLabel(self.LABELS[dim-1][i])
            form_layout.addWidget(label)
            attr = 'spacing{}'.format(i)
            edit = QLineEdit()
            edit.editingFinished.connect(
                lambda attr=attr: self._change_attr(attr, empty_to_none(getattr(self,attr).text()))
            )
            edit.setCompleter(self.defines)
            edit.setToolTip('&lt;spacing <b>every{}</b>=""&gt;<br/>Approximate single element size.'
                            .format('' if dim == 1 else str(i)))
            form_layout.addWidget(edit)
            setattr(self,attr, edit)

        self.form.setLayout(form_layout)

    def fill_form(self):
        super(RectangularRegularGeneratorController, self).fill_form()
        for i in range(self.grid_model.dim):
            attr = 'spacing{}'.format(i)
            getattr(self,attr).setText(none_to_empty(getattr(self.grid_model,attr)))

    def get_widget(self):
        return self.form

    def select_info(self, info):
        super(RectangularRegularGeneratorController, self).select_info(info)
        getattr(self, info.property).setFocus()


class RectangularRefinedGeneratorController(GridController):

    warnings_help = {
        'missing': 'Warn if any refinement references to non-existing object. Defaults to true.',
        'multiple': 'Warn if any refinement references to multiple objects. Defaults to true.',
        'outside': 'Warn if refining line lies outside of the specified object. Defaults to true.'
    }

    def _make_param_hbox(self, container_to_add, label, tooltip, defines_completer, model_path = None):
        hbox_div = QHBoxLayout()
        res = tuple(QLineEdit() for _ in range(0, self.model.dim))
        for i, r in enumerate(res):
            if self.model.dim != 1:
                axis_name = AXIS_NAMES[self.model.dim-1][i]
                hbox_div.addWidget(QLabel('{}:'.format(axis_name if axis_name else str(i))))
            else:
                axis_name = 'horizontal'
            hbox_div.addWidget(r)
            r.setToolTip(
                tooltip.format('{}'.format(i), ' in {} direction'.format(axis_name) if axis_name else ''))
            r.setCompleter(defines_completer)
            if model_path is None:
                r.editingFinished.connect(self.fire_changed)
            else:
                r.editingFinished.connect(lambda i=i, r=r: self._change_attr((model_path, i), empty_to_none(r.text())))
        container_to_add.addRow(label, hbox_div)
        return res

    def __init__(self, document, model):
        super(RectangularRefinedGeneratorController, self).__init__(document=document, model=model)

        self.form = QGroupBox()

        self.defines = get_defines_completer(self.document.defines.model, self.form)

        vbox = QVBoxLayout()

        self.options = QHBoxLayout()

        self.aspect = QLineEdit()
        self.aspect.editingFinished.connect(lambda : self._change_attr('aspect', empty_to_none(self.aspect.text())))
        self.aspect.setCompleter(self.defines)
        self.aspect.setToolTip('&lt;options <b>aspect</b>=""&gt;<br/>'
                               'Maximum aspect ratio for the rectangular and cubic elements generated '
                               'by this generator.')
        self.options.addWidget(QLabel("aspect:"))
        self.options.addWidget(self.aspect)

        self.form_layout = QFormLayout()
        self.form_layout.addRow('Options:', self.options)

        warnings_layout = QHBoxLayout()
        for w in RectangularDivideGenerator.warnings:
            cb = ComboBox()
            cb.editingFinished.connect(lambda w=w, cb=cb: self._change_attr('warn_'+w, empty_to_none(cb.currentText()), w+' warning') )
            #cb.editingFinished.connect(self.fire_changed)
            #cb.currentIndexChanged.connect(self.fire_changed)
            cb.addItems(['', 'yes', 'no'])
            cb.setEditable(True)
            cb.setToolTip('&lt;warnings <b>{}</b>=""&gt;\n'.format(w) +
                          RectangularDivideGeneratorController.warnings_help.get(w, ''))
            setattr(self, 'warn_'+w, cb)
            label = QLabel(w+':')
            label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            warnings_layout.addWidget(label)
            cb.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
            warnings_layout.addWidget(cb)
        self.form_layout.addRow('Warnings:', warnings_layout)

        vbox.addLayout(self.form_layout)

        self.refinements = QTableView()
        self.refinements.setModel(model.refinements)
        one = int(self.model.dim == 1)
        if not one:
            self.refinements.setItemDelegateForColumn(0, ComboBoxDelegate(AXIS_NAMES[self.model.dim-1],
                                                                          self.refinements, editable=False))
        def object_names():
            return self.document.geometry.model.get_names(filter=lambda x: not isinstance(x, GNGeometryBase))
        self.refinements.setItemDelegateForColumn(1-one, ComboBoxDelegate(object_names,
                                                                          self.refinements, editable=True))
        try:
            paths = self.document.geometry.model.get_paths()
        except AttributeError:
            pass
        else:
            self.refinements.setItemDelegateForColumn(2-one, ComboBoxDelegate(paths,
                                                                              self.refinements, editable=True))
        defines_delegate = DefinesCompletionDelegate(self.document.defines.model, self.refinements)
        self.refinements.setItemDelegateForColumn(3-one, defines_delegate)
        self.refinements.setItemDelegateForColumn(4-one, defines_delegate)
        self.refinements.setItemDelegateForColumn(5-one, defines_delegate)
        # self.refinements.horizontalHeader().setResizeMode(QHeaderView.ResizeToContents)
        self.refinements.setColumnWidth(1-one, 140)
        self.refinements.setColumnWidth(2-one, 120)
        self.refinements.setMinimumHeight(100)
        vbox.addWidget(table_with_manipulators(self.refinements, title='Refinements', add_undo_action=False))

        #vbox.addStretch()
        self.form.setLayout(vbox)

    def fill_form(self):
        super(RectangularRefinedGeneratorController, self).fill_form()
        self.aspect.setText(none_to_empty(self.grid_model.aspect))
        with self.mute_changes():
            for attr_name in ['warn_'+w for w in RectangularDivideGenerator.warnings]:
                cb = getattr(self, attr_name)
                a = getattr(self.model, attr_name)
                with BlockQtSignals(cb): cb.setEditText(none_to_empty(a))

    def get_widget(self):
        return self.form

    def select_info(self, info):
        super(RectangularRefinedGeneratorController, self).select_info(info)
        try:
            if info.property == 'refinements':  # select refinements index using info.reinf_row and info.reinf_col
                self.refinements.setCurrentIndex(self.model.refinements.createIndex(info.reinf_row, info.reinf_col))
        except AttributeError: pass


class RectangularDivideGeneratorController(RectangularRefinedGeneratorController):
    """Ordered and rectangular 2D and 3D divide generator script."""

    def __init__(self, document, model):
        super(RectangularDivideGeneratorController, self).__init__(document=document, model=model)

        self.gradual = ComboBox()
        self.gradual.editingFinished.connect(lambda : self._change_attr('gradual', empty_to_none(self.gradual.currentText())))
        #self.gradual.editingFinished.connect(self.fire_changed)
        #self.gradual.currentIndexChanged.connect(self.fire_changed)
        self.gradual.addItems(['', 'yes', 'no'])
        self.gradual.setEditable(True)
        self.gradual.setMinimumWidth(150)
        self.gradual.setToolTip('&lt;options <b>gradual</b>=""&gt;<br/>'
                                'Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated'
                                ' mesh may differ more than by the factor of two). Gradual is enabled by default.')
        self.options.insertWidget(0, QLabel("gradual:"))
        self.options.insertWidget(1, self.gradual)

        self.prediv = self._make_param_hbox(self.form_layout, 'Pre-refining divisions:',
                                            '&lt;<b>prediv by{}</b>=""&gt;<br/>'
                                            'The number of the initial divisions of each geometry object{}.',
                                            self.defines, 'prediv')
        self.postdiv = self._make_param_hbox(self.form_layout, 'Post-refining divisions:',
                                             '&lt;<b>postdiv by{}</b>=""&gt;<br/>'
                                             'The number of the final divisions of each geometry object{}.',
                                             self.defines, 'postdiv')

    def fill_form(self):
        super(RectangularDivideGeneratorController, self).fill_form()
        with self.mute_changes():
            with BlockQtSignals(self.gradual):
                self.gradual.setEditText(none_to_empty(self.model.gradual))
            for i in range(0, self.model.dim):
                self.prediv[i].setText(self.model.prediv[i])
                self.postdiv[i].setText(self.model.postdiv[i])

    #def save_data_in_model(self):
    #    super(RectangularDivideGeneratorController, self).save_data_in_model()
    #    self.model.prediv = [empty_to_none(self.prediv[i].text()) for i in range(0, self.model.dim)]
    #    self.model.postdiv = [empty_to_none(self.postdiv[i].text()) for i in range(0, self.model.dim)]

    #def on_edit_enter(self):
    #    super(RectangularDivideGeneratorController, self).on_edit_enter()
    #    with self.mute_changes():
    #        for i in range(0, self.model.dim):
    #            self.prediv[i].setText(self.model.prediv[i])
    #            self.postdiv[i].setText(self.model.postdiv[i])


class RectangularSmoothGeneratorController(RectangularRefinedGeneratorController):
    """Ordered and rectangular 2D and 3D divide generator script."""

    def __init__(self, document, model):
        super(RectangularSmoothGeneratorController, self).__init__(document=document, model=model)

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

    #def save_data_in_model(self):
    #    super(RectangularSmoothGeneratorController, self).save_data_in_model()
    #    self.model.small = [empty_to_none(self.small[i].text()) for i in range(0, self.model.dim)]
    #    self.model.large = [empty_to_none(self.large[i].text()) for i in range(0, self.model.dim)]
    #    self.model.factor = [empty_to_none(self.factor[i].text()) for i in range(0, self.model.dim)]

    def fill_form(self):
        with self.mute_changes():
            super(RectangularSmoothGeneratorController, self).fill_form()
            for i in range(0, self.model.dim):
                self.small[i].setText(self.model.small[i])
                self.large[i].setText(self.model.large[i])
                self.factor[i].setText(self.model.factor[i])
