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

import os

import gui
from gui.qt import QtGui
from gui.xpldocument import XPLDocument

try:
    import plask
except ImportError:
    export_dan_operation = None
else:

    def _pad(val, l=11):
        s = str(val)
        return s + ''.join([' ']*max(l-len(s), 0))

    def _parse_material(mat):
        mat = str(mat).split(':')
        if len(mat) == 1:
            return mat[0], 'ST', 0.
        dpc = mat[1].split('=')
        if len(dpc) == 2:
            dp, dc = dpc
        else:
            return mat[0]+"_"+mat[1], 'ST', 0.
        if len(dp) > 1 and dp[-2] == ' ':
            dp = dp[:-2] # there is no common way to get dopant concentration from carriers concentration
        return mat[0], dp, float(dc) * 1e6

    def write_dan(name, solvers, geo, allm=True):
        """Write dan files for given prefix

           'allm' indicates whether consider all materials as ones with constant values
        """

        ofile = open(name+'_temp.dan', 'w')
        def out(text):
            ofile.write(text + ' ')
        def outl(text):
            ofile.write(text + '\n')

        outl(name)
        outl("All                     laser_material")
        if type(geo) == plask.geometry.Cylindrical2D:
            outl("1           1           axis_symmetry_and_laser_length")
        elif type(geo) == plask.geometry.Cartesian2D:
            outl("0           %g          axis_symmetry_and_laser_length" % geo.extrusion.length)
        else:
            raise TypeError("3D geometry not supported")

        outl("0                       horizontal_setting")

        leafs = geo.get_leafs()
        boxes = geo.get_leafs_bboxes()

        outl("%s 1e-6        number_or_regions_and_dimension_scale" % _pad(len(leafs)))

        #if len(manager.profiles) != 0:
            #print_log(LOG_WARNING, "Cannot determine where constant profiles defined in the XPL file are used.")
            #print_log(LOG_WARNING, "You must add constant heats to the '%s_temp.dan' manually." % name)

        # Determine solvers
        thermal = None
        electrical = None
        for solver in solvers:
            if type(solver).__module__.split('.')[0] == "thermal":
                thermal = solver
            elif type(solver).__module__ == 'electrical.fem': # or  type(solver).__module__ == 'electrical.other_relevant_lib':
                electrical = solver

        for i,(obj,box) in enumerate(zip(leafs, boxes)):
            # first line
            out("\n" + _pad(i+1))
            out(_pad(box.lower[0])); out(_pad(box.lower[1])); out(_pad(box.upper[0])); out(_pad(box.upper[1]));
            point = plask.vec(0.5*(box.lower[0]+box.upper[0]), 0.5*(box.lower[1]+box.upper[1]))
            mat = geo.get_material(point)
            mn, dp, dc = _parse_material(mat)
            outl(mn)
            noheat = geo.item.has_role('noheat', point)
            # second line
            if geo.item.has_role('active', point):
                mt = 'j'
                if electrical is not None:
                    cy = plask.average(electrical.pnjcond)
                else:
                    cy = 0.2
                cx = 1e-6
            elif allm or mat.cond(300.) == mat.cond(400.):
                mt = 'm'
                try:
                    cx, cy = mat.cond(300.)
                except NotImplementedError:
                    cx, cy = plask.material.air.cond(300.)
                    noheat = True
            else:
                raise NotImplementedError("Allowing not constant maters is not implemented")
            outl( "%s %s %s           electrical_conductivity" % (_pad(cx,23), _pad(cy,23), mt) )
            # third line
            outl( "%s %s doping_concentration" % (_pad(dc,47), _pad(dp)) )
            # fourth line
            if allm or mat.thermk(300.) == mat.thermk(400.):
                mt = 'm'
                try:
                    kx, ky = mat.thermk(300.)
                except NotImplementedError:
                    kx, ky = plask.material.air.thermk(300.)
                    noheat = True
            else:
                raise NotImplementedError("Allowing not constant materials is not implemented")
            outl( "%s %s %s           thermal_conductivity" % (_pad(kx,23), _pad(ky,23), mt) )
            # fifth line
            if noheat:
                outl("0                                               0.0         no_heat_sources")
            elif geo.item.has_role('active', point):
                outl("-200                                            0.0         junction_heat")
            else:
                outl("-100                                            0.0         joules_heat")

        outl("")

        def parse_boundary(solver, name):
            lines = []
            if solver is not None:
                for i,(bound,value) in enumerate(eval("solver.%s_boundary" % name)):
                    points = [ solver.mesh[i] for i in bound(solver.mesh, solver.geometry) ]
                    if points:
                        xx = [ p[0] for p in points ]
                        yy = [ p[1] for p in points ]
                        samex = xx.count(xx[0]) == len(xx)
                        samey = yy.count(yy[0]) == len(yy)
                        if not (samex or samey):
                            #TODO
                            plask.print_log(plask.LOG_WARNING,
                                            "Boundary no %d for %s is not straight line. "
                                            "Skipping it. Add it manually." % (i,name))
                        else:
                            lines.append("%s %s %s %s %s %s_boundary" % (_pad(min(xx)), _pad(min(yy)), _pad(max(xx)), _pad(max(yy)), _pad(value), name))
            outl("%s boundary_conditions_%s" % (_pad(len(lines),59),name))
            for line in lines:
                outl(line)

        parse_boundary(electrical, 'voltage')
        parse_boundary(thermal, 'temperature')
        parse_boundary(thermal, 'convection')
        parse_boundary(thermal, 'heatflux')
        parse_boundary(thermal, 'radiation')

        outl("0                                                           mesh_lines")        #TODO (mesh generator refinements)

        outl("THE_END")

    def export_dan(parent):
        if not isinstance(parent.document, XPLDocument):
            msgbox = QtGui.QMessageBox()
            msgbox.setWindowTitle("Export Error")
            msgbox.setText("You can only export from xpl file.")
            msgbox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgbox.setIcon(QtGui.QMessageBox.Error)
            msgbox.exec_()
            return

        # geometries = parent.document.geometry.model.roots

        manager = plask.Manager()
        try:
            manager.load(parent.document.get_content(sections=('geometry')))
            controller = parent.document.geometry.controllers[0]
            current_model = controller.current_root()
            name = controller.current_root().name
            geom = manager.geo[str(name)]
            filename = u'{}_{}'.format(os.path.splitext(parent.document.filename)[0], name)
            write_dan(filename, manager.solvers.values(), geom)
        except Exception as err:
            msgbox = QtGui.QMessageBox()
            msgbox.setWindowTitle("RPSMES Export Error")
            msgbox.setText("There was an error while writing the RPSMES file.")
            msgbox.setDetailedText(str(err))
            msgbox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgbox.setIcon(QtGui.QMessageBox.Critical)
            if gui._DEBUG:
                import traceback
                traceback.print_exc()
        else:
            msgbox = QtGui.QMessageBox()
            msgbox.setWindowTitle("RPSMES Export")
            msgbox.setText("Geometry '{}' exported to '{}_temp.dan'. "
                           "You should examine the file and correct it!".format(name, filename))
            msgbox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgbox.setIcon(QtGui.QMessageBox.Information)
        msgbox.exec_()

    def export_dan_operation(parent):
        action = QtGui.QAction(QtGui.QIcon.fromTheme('document-save'),
                               '&Export RPSMES .dan file...', parent)
        action.triggered.connect(lambda: export_dan(parent))
        return action
