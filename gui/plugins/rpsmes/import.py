#!/usr/bin/env python3
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


from __future__ import print_function

import sys
import os

try:
    import gui
    from gui.qt.QtCore import Qt
    from gui.qt.QtWidgets import *
    from gui.qt.QtGui import *
    from gui.utils.config import CONFIG
except ImportError:
    qt = False
else:
    qt = True

try:
    next = next
except NameError:
    _raiseStopIteration = object()
    def next(iterator, default=_raiseStopIteration):
        if not hasattr(iterator, 'next'):
            raise TypeError("not an iterator")
        try:
            return iterator.next()
        except StopIteration:
            if default is _raiseStopIteration:
                raise
            else:
                return default


class RpsmesFile:
    """
    Generator, which skips empty lines, strips the '\n' character, and splits line by tabs
    """
    def __init__(self, fname):
        self.name = fname
        self.ifile = open(fname)
        self.line = 0

    def __iter__(self):
        for line in self.ifile:
            self.line += 1
            if line[-1] == "\n": line = line[:-1]
            if line.strip(): yield line.split()

    def raise_exception(self, exc, msg):
        raise exc("{} in '{}' line {}: {}".format(exc.__name__, os.path.basename(self.name), self.line, msg))


class UniqueId:
    """Generator of unique names"""

    def __init__(self, prefix, suffix='', fmt="02d", initial=1):
        self.prefix = prefix
        self.suffix = suffix
        self.fmt = fmt
        self.counter = initial - 1

    def __call__(self):
        self.counter += 1
        return self.prefix + ('{:'+self.fmt+'}').format(self.counter) + self.suffix


unique_object_name = UniqueId("object")


class Material:
    """Materials read from *.dan file"""

    def __init__(self, label, kind="metal"):
        self.base = None
        self.kind = kind
        self.condtype = None
        self.sigma = None
        self.kappa = None
        self.label = label

    def __eq__(self, other):
        return \
            self.base == other.base and \
            self.kind == other.kind and \
            self.condtype == other.condtype and \
            self.sigma == other.sigma and \
            self.kappa == other.kappa and \
            self.label == other.label

    def write(self, output, name):
        if self.base:
            output.write('  <material name="{}" base="{}">\n'.format(name, self.base))
        else:
            if self.condtype:
                output.write('  <material name="{}" base="{}">\n    <condtype>{}</condtype>\n'
                             .format(name, self.kind, self.condtype))
            else:
                output.write('  <material name="{}" base="{}">\n'.format(name, self.kind))
        if self.kappa is not None:
            if self.kappa[0] == self.kappa[1]:
                output.write('    <thermk>{}</thermk>\n'.format(self.kappa[0]))
            else:
                output.write('    <thermk>{}, {}</thermk>\n'.format(*self.kappa))
        if self.sigma is not None:
            if self.sigma[0] == self.sigma[1]:
                output.write('    <cond>{}</cond>\n'.format(self.sigma[0]))
            else:
                output.write('    <cond>{}, {}</cond>\n'.format(*self.sigma))
        output.write('  </material>\n')


class Region:
    """Regions read from *.dan file"""

    def __init__(self, axes):
        self.a0 = axes[0]
        self.a1 = axes[1]
        self.repeat = 0
        self.name = None
        self.roles = set()

    def write(self, output):
        w = '{:.4f}'.format(self.x1 - self.x0)
        h = '{:.4f}'.format(self.y1 - self.y0)
        more = ""
        if self.roles: more += ' role="{}"'.format(','.join(self.roles))
        if self.name: more += ' name="{}"'.format(self.name)
        locals().update(self.__dict__)
        if self.repeat:
            sx, sy = self.shift
            output.write('      <item {a0}="{x0:.4f}" {a1}="{y0:.4f}">'
                         '<arrange d{a0}="{sx}" d{a1}="{sy}" count="{repeat}">\n'
                         '        <block d{a0}="{w}" d{a1}="{h}" material="{material}"{more}/>\n'
                         '      </arrange></item>'.format(**locals()))
        else:
            output.write('      <item {a0}="{x0:.4f}" {a1}="{y0:.4f}">'
                         '<block d{a0}="{w}" d{a1}="{h}" material="{material}"{more}/></item>\n'
                         .format(**locals()))


ownmats = {
    'LED_nGaN': "GaN:Si",
    'LED_active': "In(0.06)GaN",
    'LED_pAlGaN': "Al(0.17)GaN:Mg",
    'LED_pGaN': "GaN:Mg",
}


def parse_material_name(mat, comp, dopant):
    if mat in ownmats:
        return ownmats[mat]
    if mat[0] == mat[0].lower():
        elements = ['']
    else:
        elements = []
    for l in mat:
        if l != l.lower(): elements.append('')
        elements[-1] = elements[-1] + l
    if len(elements) == 1:
        result = elements[0]
    elif len(elements) == 2:
        result = ''.join(elements)
    else:
        if comp == 0.:
            result = ''.join(elements[1:])
        elif comp == 1.:
            result = elements[0] + ''.join(elements[2:])
        else:
            result = elements[0] + '({})'.format(comp) + ''.join(elements[1:])
    if dopant != 'ST': result += ':{}'.format(dopant)
    return result


def write_xpl(name, sym, length, axes, materials, regions, heats, boundaries, pnjcond, actlevel):
    """Write output xpl file"""

    ofile = open(name, 'w')

    def out(text=''):
        ofile.write(text)
        ofile.write('\n')

    out('<plask>\n')

    geometry = ['cartesian2d', 'cylindrical'][sym]
    suffix = ['2D', 'Cyl'][sym]

    if sym == 0:
        geomore = ' length="{}"'.format(length)
    else:
        geomore = ''

    # materials
    materials = list(materials.items())
    if materials:
        out('<materials>')
        for mn, mat in materials:
            mat.write(ofile, mn)
        out('</materials>\n')

    # geometry
    out('<geometry>\n  <{} name="main" axes="{}"{}>'.format(geometry, axes, geomore))
    out('    <container>')
    for r in regions:
        r.write(ofile)
    out('    </container>')
    out('  </{}>\n</geometry>\n'.format(geometry))

    # default mesh generator
    out('<grids>')
    out('  <generator type="rectangular2d" method="divide" name="default">\n'
        '    <postdiv by0="4" by1="2"/>\n  </generator>')
    out('</grids>\n')

    def save_boundaries(name):
        if boundaries[name]:
            out('    <{}>'.format(name))
            for data in boundaries[name]:
                out('      <condition {valname}="{value}" {optional}><place line="{dir}"'
                    ' start="{start}" stop="{stop}" at="{at}"/></condition>'.format(**data))
            out('    </{}>'.format(name))

    # default solvers
    therm = boundaries['temperature'] or boundaries['convection'] or boundaries['radiation'] or heats
    electr = boundaries['voltage']

    if therm or electr:
        out('<solvers>')
        if therm:
            out('  <thermal solver="Static{}" name="THERMAL">'.format(suffix))
            out('    <geometry ref="main"/>\n    <mesh ref="default"/>')
            save_boundaries('temperature')
            save_boundaries('convection')
            save_boundaries('radiation')
            out('  </thermal>')
        if electr:
            out('  <electrical solver="Shockley{}" name="ELECTRICAL">'.format(suffix))
            out('    <geometry ref="main"/>\n    <mesh ref="default"/>')
            if pnjcond is not None:
                out('    <junction pnjcond="{:g}" beta="18" js="1"/>'.format(pnjcond[1]))
            save_boundaries('voltage')
            out('  </electrical>')
        out('</solvers>\n')

    # connections
    if (therm and electr):
        out('<connects>')
        out('  <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>')
        if not heats:
            out('  <connect in="THERMAL.inHeat" out="ELECTRICAL.outHeat"/>')
        else:
            out('  <!-- heats are attached in the script -->')
        out('</connects>\n')

    # # script
    out(
        '<script><![CDATA[\n# Here you may put your calculations. Below there is a sample script (tune it to your needs):\n')

    if bool(therm) ^ bool(electr):
        out('import sys, os')

    if heats:
        out('heat_profile = StepProfile(GEO.main)')
        for heat in heats.items():
            out('heat_profile[GEO.{}] = {}'.format(heat))

        if electr:
            out('THERMAL.inHeat = ELECTRICAL.outHeat + heat_profile.outHeat\n')
        else:
            out('THERMAL.inHeat = heat_profile.outHeat\n')

    if therm and electr:
        out('task = algorithm.ThermoElectric(THERMAL, ELECTRICAL)')
        out('task.run()')
    else:
        if therm:
            out('THERMAL.compute()')
        elif electr:
            out('ELECTRICAL.compute()')

    if electr:
        out('\nprint_log(LOG_INFO, "Total current: {:.3g} mA".format(abs(ELECTRICAL.get_total_current())))')

    if bool(therm) ^ bool(electr):
        if actlevel is not False:
            if actlevel is True:
                out('\nactbox = GEO.main.get_object_bboxes(GEO.active)[0]')
                out('actlevel = 0.5 * (actbox.lower[1] + actbox.upper[1])')
            else:
                out('actlevel = {:g}'.format(actlevel))
            out('actgrid = mesh.Rectangular2D(ELECTRICAL.mesh.axis0, mesh.Rectilinear([actlevel]))')
        if therm:
            out('\ntemperature = THERMAL.outTemperature(THERMAL.mesh)')
            out('heats = THERMAL.inHeat(THERMAL.mesh)')
        if electr:
            out('voltage = ELECTRICAL.outVoltage(ELECTRICAL.mesh)')
            out('current = ELECTRICAL.outCurrentDensity(ELECTRICAL.mesh)')
            if actlevel is not False:
                out('acurrent = ELECTRICAL.outCurrentDensity(actgrid, "SPLINE")')
        out('\nh5file = h5py.File(os.path.splitext(sys.argv[0])[0]+".h5", "w")')
        if therm:
            out('save_field(temperature, h5file, "Temperature")')
            out('save_field(heats, h5file, "Heat")')
        if electr:
            out('save_field(voltage, h5file, "Voltage")')
            out('save_field(current, h5file, "CurrentDensity")')

        out('h5file.close()')

    out('\nplot_geometry(GEO.main, margin=0.01)')
    out('defmesh = MSG.default(GEO.main.item)')
    out('plot_mesh(defmesh, color="0.75")')
    if electr:
        out('plot_boundary(ELECTRICAL.voltage_boundary, defmesh, ELECTRICAL.geometry, color="b", marker="D")')
    if therm:
        out('plot_boundary(THERMAL.temperature_boundary, defmesh, THERMAL.geometry, color="r")')
        out('plot_boundary(THERMAL.convection_boundary, defmesh, THERMAL.geometry, color="g")')
        out('plot_boundary(THERMAL.radiation_boundary, defmesh, THERMAL.geometry, color="y")')
    out('gcf().canvas.set_window_title("Default mesh")')

    if therm and electr:
        out('\nfigure()')
        out('task.plot_temperature()')
        out('\nfigure()')
        out('task.plot_voltage()')
        out('\nfigure()')
        out('task.plot_junction_current()')

    elif therm or electr:
        if therm:
            out('\nfigure()')
            out('plot_field(temperature, 16)')
            out('colorbar()')
            out('plot_geometry(GEO.main, color="w")')
            out('gcf().canvas.set_window_title("Temperature")')
            out('\nfigure()')
            out('plot_field(heats, 16)')
            out('colorbar()')
            out('plot_geometry(GEO.main, color="w")')
            out('gcf().canvas.set_window_title("Heat sources density")')
        if electr:
            out('\nfigure()')
            out('plot_field(voltage, 16)')
            out('colorbar()')
            out('plot_geometry(GEO.main, color="w")')
            out('gcf().canvas.set_window_title("Electric potential")')
            if actlevel is not False:
                out('\nfigure()')
                out('plot(actgrid.axis0, abs(acurrent.array[:,0,1]))')
                out('xlabel(u"{} [\\xb5m]")'.format(axes[0]))
                out('ylabel("current density (kA/cm$^2$)")')
                out('simplemesh = mesh.Rectangular2D.SimpleGenerator()(GEO.main.item)')
                out('for x in simplemesh.axis0:')
                out('    axvline(x, ls=":", color="k")')
                out('xlim(0., simplemesh.axis0[-2])')
                out('gcf().canvas.set_window_title("Current density in the active region")')

    out('\nshow()')

    out('\n]]></script>\n')

    out('</plask>')


def read_dan(fname):
    """Open and read dan file

       On exit this function returns dictionary of custom materials and list of regions
    """

    rpsmes = RpsmesFile(fname)
    input = iter(rpsmes)

    # Header
    name = next(input)[0]  # structure name (will be used for output file)
    matdb = next(input)[0]  # materials database spec (All by default)
    line = next(input)  # symmetry (0: Cartesian2D, 1: Cylindrical) type and length
    sym = int(line[0])
    length = float(line[1])
    setting = int(next(input)[0])  # setting (10,11 - temporal calculations, 100,100 - 3D)
    line = next(input)  # number of defined regions and scale
    nregions = int(line[0])
    scale = float(line[1]) * 1e6  # in xpl all dimensions are in microns

    pnjcond = None

    if setting >= 10:
        rpsmes.raise_exception(NotImplementedError,
                              "3D structure nor temporal data not implemented yet ({})".format(setting))

    # Set up symmetry
    axes = ['xy', 'rz'][sym]

    regions = []
    materials = {}
    heats = {}

    # Read each region
    for nr in range(nregions):
        r = Region(axes)

        # number, position, material
        line = next(input)
        n = int(line[0])
        if n == 0:
            r.repeat = int(line[1])
            r.shift = [0., 0.]
            r.shift[{'pionowo': 1, 'poziomo': 0}[line[3].lower()]] = scale * float(line[2])
            line = next(input)
        r.x0, r.y0, r.x1, r.y1 = [scale * float(x) for x in line[1:5]]
        mat = line[5]
        if mat == "WYPELNIENIE":
            line = next(input)
            line = next(input)
            line = next(input)
            line = next(input)
            continue

        # conductivity
        line = next(input)
        sigma = [float(line[0]), float(line[1])]
        sigma_t = line[2].lower()

        if sigma_t == 'j':
            pnjcond = sigma

        # doping
        line = next(input)
        doping, dopant = 1e-6 * float(line[0]), line[1]

        # heat conductivity
        line = next(input)
        kappa = [float(line[0]), float(line[1])]
        kappa_t = line[2].lower()

        if mat in ('GaN', 'AlN', 'InN'):
            h0 = scale * kappa[0]
            h1 = scale * kappa[1]
            h = r.y1 - r.y0
            if abs(h-h1) > 1e-3 or abs(h-h0) > 1e-3:
                kappa_t = 'x'
                if abs(h-h0) <= 1e-3: h0 = 'h'
                if abs(h-h1) <= 1e-3: h1 = 'h'
                kappa[0] = 'self.base.thermk(T, {})[0]'.format(h0)
                kappa[1] = 'self.base.thermk(T, {})[1]'.format(h1)

        # create custom material if necessary
        force_manual = '_' in mat
        if sigma_t not in ('n', 'p', 'j') or kappa_t not in ('n', 'p') or force_manual:
            material = Material(mat)
            if sigma_t not in ('n', 'p') or force_manual:
                if sigma_t == 'u':
                    material.sigma = [1e-16, 1e-16]
                else:
                    material.sigma = sigma
            else:
                material.base = parse_material_name(mat, sigma[0], dopant)
            if kappa_t not in ('n', 'p') or force_manual:
                material.kappa = kappa
            else:
                material.base = parse_material_name(mat, kappa[0], dopant)
            found = False
            for mk, mv in materials.items():
                if material == mv:
                    found = True
                    mat = mk
                    break
            if not found:
                if material.base is not None and ':' in material.base and '=' not in material.base:
                    suffix = ':' + material.base.split(':')[1]
                else:
                    suffix = ''
                unique_material_name = UniqueId(mat+'_', suffix)
                if (sigma_t in ('n', 'p') or kappa_t in ('n', 'p')) and not force_manual:
                    mat = unique_material_name()  # the given name is the one from the database
                while mat in materials and materials[mat] != material:
                    mat = unique_material_name()
                materials[mat] = material
        else:
            mat = parse_material_name(mat, kappa[0], dopant)

        r.material = mat
        if ':' in mat:
            r.material += "={:g}".format(doping)  # add doping information

        # heat sources
        line = next(input)
        ht = int(line[0])
        if sigma_t == 'j':
            r.roles.add('active')
        if ht == 0:
            r.roles.add('noheat')
        elif ht == -1:
            r.name = unique_object_name()
            heats[r.name] = float(line[1])
        elif ht == -200:
            r.roles.add('active')
        elif ht != -100:
            rpsmes.raise_exception(ValueError, "wrong heat source type")

        # save to the list (TODO: make it more clever, using heuristic algorithms to construct stacks and shelves)
        regions.append(r)

    # boundary conditions
    def parse_bc(vname='value', opt=''):
        bounds = []
        line = next(input)
        nbc = int(line[0])
        for nc in range(nbc):
            line = next(input)
            x0, y0, x1, y1 = [scale * float(x) for x in line[0:4]]
            try:
                val = float(line[4])
            except IndexError:
                val = 0.
            except ValueError:
                val = 0.
            if (x0 == x1):
                bounds.append(dict(dir='vertical', at=x0, start=y0, stop=y1, optional=opt, value=val, valname=vname))
            elif (y0 == y1):
                bounds.append(dict(dir='horizontal', at=y0, start=x0, stop=x1, optional=opt, value=val, valname=vname))
            else:
                rpsmes.raise_exception(ValueError, "boundary condition line is neither horizontal nor vertical")
        return bounds

    boundaries = {
        'voltage': parse_bc(),
        'temperature': parse_bc(),
        'convection': parse_bc('coeff', 'ambient="300"'),
        'radiation': parse_bc('emissivity', 'ambient="300"')}
    try:
        boundaries['mesh'] = parse_bc()
    except:
        pass

    actives = [r for r in regions if 'active' in r.roles]
    if len(actives) == 1:
        actives[0].name = "active"
        actlevel = True
    elif len(actives) == 0:
        actlevel = False
    else:
        actlevel = sum([0.5 * (r.y0 + r.y1) for r in actives]) / len(actives)

    return name, sym, length, axes, materials, regions, heats, boundaries, pnjcond, actlevel


if qt:

    def import_dan(parent):
        """Convert _temp.dan file to .xpl, save it to disk and open in PLaSK"""

        remove_self = parent.document.filename is None and not parent.isWindowModified()

        iname = QFileDialog.getOpenFileName(parent, "Import RPSMES file", gui.CURRENT_DIR,
                                                  "RPSMES file (*.dan)")
        if type(iname) == tuple:
            iname = iname[0]
        if not iname:
            return
        dest_dir = os.path.dirname(iname)

        try:
            read = read_dan(iname)
            obase = os.path.join(dest_dir,
                                 os.path.basename(iname)[:-9] if iname[-9] == '_' else os.path.basename(iname)[:-4])
        except Exception as err:
            if gui._DEBUG:
                import traceback
                traceback.print_exc()
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Import Error")
            msgbox.setText("There was an error while reading the RPSMES file.\n\n"
                           "Probably the chosen file was not in a RPSMES or the parser does not understand its syntax.")
            msgbox.setDetailedText(str(err))
            msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
            msgbox.setIcon(QMessageBox.Icon.Critical)
            qt_exec(msgbox)
        else:
            oname = obase + '.xpl'
            n = 1
            while os.path.exists(oname):
                oname = obase + '-{}.xpl'.format(n)
                n += 1
            try:
                write_xpl(oname, *read[1:])
            except Exception as err:
                if gui._DEBUG:
                    import traceback
                    traceback.print_exc()
                msgbox = QMessageBox()
                msgbox.setWindowTitle("Import Error")
                msgbox.setText("There was an error while saving the converted XPL file.")
                msgbox.setDetailedText(str(err))
                msgbox.setStandardButtons(QMessageBox.StandardButton.Ok)
                msgbox.setIcon(QMessageBox.Icon.Critical)
                qt_exec(msgbox)
            else:
                parent.load_file(oname)

    def import_dan_operation(parent):
        action = QAction(QIcon.fromTheme('document-open'),
                               '&Import RPSMES .dan file...', parent)
        action.triggered.connect(lambda: import_dan(parent))
        CONFIG.set_shortcut(action, 'import_dan', 'Import RPSMES .dan')
        return action


if __name__ == '__main__':
    for iname in sys.argv[1:]:
        try:
            dest_dir = os.path.dirname(iname)
            read = read_dan(iname)
            obase = os.path.join(dest_dir,
                                 os.path.basename(iname)[:-9] if iname[-9] == '_' else os.path.basename(iname)[:-4])
            oname = obase + '.xpl'
            n = 1
            while os.path.exists(oname):
                oname = obase + '-{}.xpl'.format(n)
                n += 1
            write_xpl(oname, *read[1:])
        except Exception:
            print("In file:", iname, file=sys.stderr)
            import traceback
            traceback.print_exc()
        else:
            print("Wrote '{}'".format(oname))
