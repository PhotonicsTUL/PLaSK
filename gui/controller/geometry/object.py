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


from .node import GNodeController
from ...model.geometry.reader import axes_to_str
from ...utils.qsignals import BlockQtSignals
from ...utils.str import none_to_empty


AXES = {
    2: ('', 'y,z', 'x,y', 'z,x', 'r,z', 't,v', 'tran,vert'),
    3: ('', 'x,y,z', 'z,x,y', 'y,z,x', 'p,r,z', 'l,t,v', 'long,tran,vert')
}


class GNObjectController(GNodeController):

    have_mesh_settings = True

    # def __init__(self, document, model, node):
    #     super().__init__(document, model, node)
    #     self.in_parent_controller = self.node.get_controller_for_inparent()

    def construct_form(self, roles=True):
        self.construct_group('Basic Settings', 0)
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

        if self.have_mesh_settings:
            from .leaf import GNLeafController
            self.construct_group('Meshing Settings')
            if isinstance(self, GNLeafController):
                self.step_num = self.construct_line_edit('Maximum steps number:', node_property_name='step_num',
                                                         display_property_name='maximum steps number')
                self.step_num.setToolTip(u'&lt;{} <b>steps-num</b>="" steps-dist="" ...&gt;<br/>'
                                         u'Maximum number of the mesh steps in every direction the object is '
                                         u'divided into, if it is non-uniform. (integer)'
                                         .format(self.node.tag_name(False)))
                self.step_dist = self.construct_line_edit('Minimum step size:', node_property_name='step_dist',
                                                          display_property_name='minimum step size', unit=u'µm')
                self.step_dist.setToolTip(u'&lt;{} steps-num="" <b>steps-dist</b>="" ...&gt;<br/>'
                                          u'Minimum step size if the object is non-uniform.'
                                          .format(self.node.tag_name(False)))
            else:
                self.step_num = self.construct_line_edit('Maximum steps number in objects:',
                                                         node_property_name='step_num',
                                                         display_property_name='objects maximum steps number')
                self.step_num.setToolTip(u'&lt;{} <b>steps-num</b>="" steps-dist="" ...&gt;<br/>'
                                         u'Maximum number of the mesh steps in every direction, each object below is '
                                         u'divided into, if it is non-uniform. (integer)'
                                         .format(self.node.tag_name(False)))
                self.step_dist = self.construct_line_edit('Minimum step size in objects:',
                                                          node_property_name='step_dist',
                                                          display_property_name='objects minimum step size',
                                                          unit=u'µm')
                self.step_dist.setToolTip(u'&lt;{} steps-num="" <b>steps-dist</b>="" ...&gt;<br/>'
                                          u'Minimum step size for each object below, if it is non-uniform.'
                                          .format(self.node.tag_name(False)))

    def save_data_in_model(self):
        if self.in_parent_controller is not None: self.in_parent_controller.save_data_in_model()

    def fill_form(self):
        super().fill_form()
        self.name.setText(none_to_empty(self.node.name))
        if self.role is not None:
            self.role.setText(none_to_empty(self.node.role))
        with BlockQtSignals(self.axes) as ignored:
            self.axes.setEditText(axes_to_str(self.node.axes))
        if self.in_parent_controller is not None: self.in_parent_controller.fill_form()
        if self.have_mesh_settings:
            self.step_num.setPlaceholderText(str(self.node.get_step_num(False) or 10))
            self.step_num.setText(none_to_empty(self.node.step_num))
            self.step_dist.setPlaceholderText(str(self.node.get_step_dist(False) or 0.005))
            self.step_dist.setText(none_to_empty(self.node.step_dist))
