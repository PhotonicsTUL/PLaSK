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

from .. import Controller
from ..table import table_with_manipulators
from ...model.grids.generator_rectilinear import RectilinearDivideGenerator
from ...qt import QtGui
from ...utils.str import empty_to_none

class RectilinearDivideGeneratorConroller(Controller):
    """ordered and rectangular 2D and 3D divide generator script"""

    axes_names = [
        [""], ["horizontal", "vertical"], ["longitudinal", "transverse", "vertical"]
    ]

    warning_help = {
        'missing': 'Warn if any refinement references to non-existing object. Defaults to true.',
        'multiple': 'Warn if any refinement references to multiple objects. Defaults to true.',
        'outside': 'Warn if refining line lies outside of the specified object. Defaults to true.'
    }

    def _make_div_hbox(self, container_to_add, label, tooltip):
        hbox_div = QtGui.QHBoxLayout()
        res = tuple(QtGui.QLineEdit() for _ in range(0, self.model.dim))
        for i in range(0, self.model.dim):
            axis_name = RectilinearDivideGeneratorConroller.axes_names[self.model.dim-1][i]
            hbox_div.addWidget(QtGui.QLabel('by {}{}:'.format(i, ' (' + axis_name + ')' if axis_name else '')))
            hbox_div.addWidget(res[i])
            res[i].setToolTip(tooltip.format(' in {} direction'.format(axis_name) if axis_name else ''))
        container_to_add.addRow(label, hbox_div)
        return res

    def __init__(self, document, model):
        super(RectilinearDivideGeneratorConroller, self).__init__(document=document, model=model)

        self.form = QtGui.QGroupBox()

        vbox = QtGui.QVBoxLayout()
        form_layout = QtGui.QFormLayout()

        self.gradual = QtGui.QComboBox()    #not checkbox to allow put defines {}
        self.gradual.addItems(['', 'yes', 'no'])
        self.gradual.setEditable(True)
        self.gradual.setToolTip("Turn on/off smooth mesh step (i.e. if disabled, the adjacent elements of the generated"
                                " mesh may differ more than by the factor of two). Gradual is enabled by default.")
        form_layout.addRow("gradual", self.gradual)

        self.prediv = self._make_div_hbox(form_layout, "prediv",
                                          "Set number of the initial divisions of each geometry object{}.")
        self.postdiv = self._make_div_hbox(form_layout, "postdiv",
                                           "Set number of the final divisions of each geometry object{}.")

        warnings_layout = QtGui.QHBoxLayout()
        for w in RectilinearDivideGenerator.warnings:
            cb  = QtGui.QComboBox()
            cb.addItems(['', 'true', 'false'])
            cb.setEditable(True)
            cb.setToolTip(RectilinearDivideGeneratorConroller.warning_help.get(w, ''))
            setattr(self, 'warning_'+w, cb)
            label = QtGui.QLabel(w+':')
            label.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Preferred)
            warnings_layout.addWidget(label)
            cb.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
            warnings_layout.addWidget(cb)
        form_layout.addRow('warnings', warnings_layout)

        vbox.addLayout(form_layout)

        self.refinements = QtGui.QTableView()
        self.refinements.setModel(model.refinements)
        vbox.addWidget(table_with_manipulators(self.refinements, title="Refinements:"))

        #vbox.addStretch()
        self.form.setLayout(vbox)

    def save_data_in_model(self):
        for attr_name in ['gradual'] + ['warning_'+w for w in RectilinearDivideGenerator.warnings]:
            setattr(self.model, attr_name, empty_to_none(getattr(self, attr_name).currentText()))
        self.model.set_prediv([empty_to_none(self.prediv[i].text()) for i in range(0, self.model.dim)])
        self.model.set_postdiv([empty_to_none(self.postdiv[i].text()) for i in range(0, self.model.dim)])

    def on_edit_enter(self):
        for attr_name in ['gradual'] + ['warning_'+w for w in RectilinearDivideGenerator.warnings]:
            a = getattr(self.model, attr_name)
            getattr(self, attr_name).setEditText('' if a is None else a)
        for i in range(0, self.model.dim):
            self.prediv[i].setText(self.model.get_prediv(i))
            self.postdiv[i].setText(self.model.get_postdiv(i))

    def get_widget(self):
        return self.form