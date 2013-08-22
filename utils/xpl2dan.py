#!/usr/bin/env plask

import sys
import os

from plask import *

def _pad(val, l=11):
    s = str(val)
    return s + ''.join([' ']*max(l-len(s), 0))

def _parse_material(mat):
    mat = str(mat).split(':')
    if len(mat) == 1:
        return mat[0], 'ST', 0.
    dp, dc = mat[i].split('=')
    if dp[-2] == ' ': dp = dp[:-2] # there is no common way to get dopant concentration from carriers concentration
    return mat[0], dp, float(dc)


def write_dan(name, manager, geo, allm=True):
    '''Write dan files for given prefix

       'allm' indicates whether consider all materials as ones with constant values
    '''

    print_log(LOG_INFO, "Writing %s_temp.dan" % name)
    ofile = open(name+'_temp.dan', 'w')
    def out(text):
        ofile.write(text + ' ')
    def outl(text):
        ofile.write(text + '\n')

    outl(name)
    outl("All                     laser_material")
    if type(geo) == geometry.Cylindrical2D:
        outl("1           1           axis_symmetry_and_laser_length")
    elif type(geo) == geometry.Cartesian2D:
        outl("0           %g          axis_symmetry_and_laser_length" % geo.extrusion.length)
    else:
        raise TypeError("3D geometry not supported")

    outl("0                       horizontal_setting")

    leafs = geo.get_leafs()
    boxes = geo.get_leafs_bboxes()

    outl("%s 1e-6        number_or_regions_and_dimension_scale" % _pad(len(leafs)))

    if len(manager.profiles) != 0:
        print_log(LOG_WARNING, "Cannot determine where constant profiles defined in the XPL file are used.")
        print_log(LOG_WARNING, "You must add constant heats to the '%s_temp.dan' manually." % name)

    # Determine solvers
    thermal = None
    electrical = None
    for solver in manager.solvers.values():
        if type(solver).__module__.split('.')[0] == "thermal":
            thermal = solver
        elif type(solver).__module__ == 'electrical.fem': # or  type(solver).__module__ == 'electrical.other_relevant_lib':
            electrical = solver

    for i,(obj,box) in enumerate(zip(leafs, boxes)):
        # first line
        out("\n" + _pad(i+1))
        out(_pad(box.lower[0])); out(_pad(box.lower[1])); out(_pad(box.upper[0])); out(_pad(box.upper[1]));
        point = vec(0.5*(box.lower[0]+box.upper[0]), 0.5*(box.lower[1]+box.upper[1]))
        mat = geo.get_material(point)
        mn, dp, dc = _parse_material(mat)
        outl(mn)
        noheat = geo.item.has_role('noheat', point)
        # second line
        if geo.item.has_role('active', point):
            mt = 'j'
            if electrical is not None:
                cx, cy = electrical.pnjcond
            else:
                cx, cy = 1e-6, 0.2
        elif allm or mat.cond(300.) == mat.cond(400.):
            mt = 'm'
            try:
                cx, cy = mat.cond(300.)
            except NotImplementedError:
                cx, cy = material.air.cond(300.)
                noheat = True
        else:
            raise NotImplementedError("Allowing not constant maters is not implemented")
        outl( "%s %s %s           electrical_conductivity" % (_pad(cx,23), _pad(cy,23), mt) )
        # third line
        outl( "%s %s doping_concentration" % (_pad(dc,47), _pad(dp)) )
        # fourth line
        if allm or mat.thermk(300.) == material.thermk(400.):
            mt = 'm'
            try:
                kx, ky = mat.thermk(300.)
            except NotImplementedError:
                kx, ky = material.air.thermk(300.)
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
                points = [ solver.mesh[i] for i in bound(solver.mesh) ]
                if points:
                    xx = [ p[0] for p in points ]
                    yy = [ p[1] for p in points ]
                    samex = xx.count(xx[0]) == len(xx)
                    samey = yy.count(yy[0]) == len(yy)
                    if not (samex or samey):
                        print_log(LOG_WARNING, "Boundary no %d for %s is not straight line. Skipping it. Add it manually." % (i,name))
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


if __name__ == "__main__":

    try:
        iname = sys.argv[1]
    except IndexError:
        print_log(LOG_CRITICAL_ERROR, "Usage: %s input_file.xpl\n" % sys.argv[0])
        sys.exit(2)
    else:
        dest_dir = os.path.dirname(iname)
        name = os.path.join(dest_dir, iname[:-4])

        manager = plask.Manager()
        manager.load(iname)
        geos = [g for g in manager.geometrics.values() if isinstance(g, geometry.Geometry)]
        if len(geos) != 1:
            raise ValueError("More than one geometry defined in %s" % iname)
        write_dan(name, manager, geos[0])

        print_log(LOG_INFO, "Done!")

