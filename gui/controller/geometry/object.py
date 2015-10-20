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

from .node import GNodeController
from ...model.geometry.reader import axes_to_str
from ...utils.qsignals import BlockQtSignals
from ...utils.str import empty_to_none, none_to_empty


AXES = {
    2: ('', 'y,z', 'x,y', 'r,z', 't,v', 'tran,vert'),
    3: ('', 'x,y,z', 'z,x,y', 'p,r,z', 'l,t,v', 'long,tran,vert')
}

class GNObjectController(GNodeController):

    #def __init__(self, document, model, node):
    #    super(GNObjectController, self).__init__(document, model, node)
    #    self.in_parent_controller = self.node.get_controller_for_inparent()

    def construct_form(self, roles=True):
        from .geometry import GNGeometryController
        self.construct_group('Basic Settings')
        self.name = self.construct_line_edit('Name:', node_property_name='name')
        self.name.setToolTip('&lt;{} <b>name</b>="" ...&gt;<br/>'
                                'Object name for further reference.'
                                ' In the script section, the object is available by GEO table,'
                                ' which is indexed by names of geometry objects.'.format(self.node.tag_name(False)))
        if roles:
            self.role = self.construct_line_edit('Roles:', node_property_name='role', display_property_name='roles')
            self.role.setToolTip('&lt;{} <b>role</b>="" ...&gt;<br/>'
                                    'Object role. Important for some solvers.'.format(self.node.tag_name(False)))
        else:
            self.role = None
        self.axes = self.construct_combo_box('Axes:', AXES.get(self.node.dim, ('',)), node_property_name='axes')
        self.axes.setToolTip('&lt;{} <b>axes</b>="" ...&gt;<br/>'
                            'Specification of the axes.'
                            ' Most popular values are <it>xy</it>, <it>yz</it>, <it>rz</it>'
                            ' (letters are names of the horizontal and vertical axis, respectively).'
                             .format(self.node.tag_name(False)))
        self.in_parent_controller = self.node.get_controller_for_inparent(self.document, self.model)
        if self.in_parent_controller is not None:
            self.vbox.insertWidget(0, self.in_parent_controller.get_widget())

    def save_data_in_model(self):
        if self.in_parent_controller is not None: self.in_parent_controller.save_data_in_model()

    def fill_form(self):
        super(GNObjectController, self).fill_form()
        self.name.setText(none_to_empty(self.node.name))
        if self.role is not None:
            self.role.setText(none_to_empty(self.node.role))
        with BlockQtSignals(self.axes) as ignored:
            self.axes.setEditText(axes_to_str(self.node.axes))
        if self.in_parent_controller is not None: self.in_parent_controller.fill_form()
