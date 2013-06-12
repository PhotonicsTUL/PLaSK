#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
    This tool converts RPSMES .dan file to PLaSK .xpl file.

    Usage:

      dan2xpl input_file_temp.dan

    If the conversion is successful, the script writes XPL file with the structure and sample commands.
'''
import sys
import os, os.path

# Generator of unique names
class UniqueId(object):

    def __init__(self, prefix, fmt="%02d", initial=1):
        self.prefix = prefix
        self.fmt = fmt
        self.counter = initial - 1

    def __call__(self):
        self.counter += 1
        return self.prefix + self.fmt % self.counter


unique_object_name = UniqueId("object")

unique_material_name = UniqueId("material")

# Classes for storing data read from *.dan file
class Region(object):
    '''Region class'''

    def __init__(self, axes):
        self.a0 = axes[0]
        self.a1 = axes[1]
        self.repeat = 0
        self.name = None
        self.role = None

    def write(self, output):
        w = '%.4f' % (self.x1 - self.x0)
        h = '%.4f' % (self.y1 - self.y0)
        more = ""
        if self.repeat and not self.name: self.name = unique_object_name()
        if self.role: more += ' role="%s"' % self.role
        if self.name: more += ' name="%s"' % self.name
        locals().update(self.__dict__)
        output.write('      <item %(a0)s="%(x0).4f" %(a1)s="%(y0).4f"><block d%(a0)s="%(w)s" d%(a1)s="%(h)s" material="%(material)s"%(more)s/></item>\n' % locals())
        if self.repeat:
            x = self.x0
            y = self.y0
            for i in range(self.repeat):
                x += self.shift[0]
                y += self.shift[1]
                output.write('      <item %(a0)s="%(x).4f" %(a1)s="%(y).4f"><again ref="%(name)s"/></item>\n' % locals())


class Material(object):
    '''Material class'''

    def __init__(self, kind="metal"):
        self.base = None
        self.kind = kind
        self.condtype = None
        self.sigma = None
        self.kappa = None

    def __eq__(self, other):
        return \
            self.base == other.base and \
            self.kind == other.kind and \
            self.condtype == other.condtype and \
            self.sigma == other.sigma and \
            self.kappa == other.kappa

    def write(self, output, name):
        if self.base:
            output.write('  <material name="%s" base="%s">\n' % (name, self.base))
        else:
            if self.condtype:
                output.write('  <material name="%s" kind="%s" condtype="%s">\n' % (name, self.kind, self.condtype))
            else:
                output.write('  <material name="%s" kind="%s">\n' % (name, self.kind))
        if self.kappa is not None:
            output.write('    <thermk>%s, %s</thermk>\n' % tuple(self.kappa))
        if self.sigma is not None:
            output.write('    <cond>%s, %s</cond>\n' % tuple(self.sigma))
        output.write('  </material>\n')



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
        elements[-1] = elements[-1]+l
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
            result = elements[0] + '(%s)' % comp + ''.join(elements[1:])
    if dopant != 'ST': result += ":%s" % dopant
    return result



def read_dan(fname):
    '''Open and read dan file

       On exit this function returns dictionary of custom materials and list of regions
    '''

    print("Reading %s:" % fname)

    ifile = open(fname)


    # Set-up generator, which skips empty lines, strips the '\n' character, and splits line by tabs
    def Input(ifile):
        for line in ifile:
            if line[-1] == "\n": line = line[:-1]
            print("> " + line)
            if line.strip(): yield line.split()
    input = Input(ifile)

    # Header
    name = input.next()[0]                      # structure name (will be used for output file)
    matdb = input.next()[0]                     # materials database spec (All by default)
    line = input.next()                         # symmetry (0: Cartesian2D, 1: Cylindrical2D) type and length
    sym = int(line[0])
    length = float(line[1])
    setting = int(input.next()[0])              # setting (10,11 - temporal calculations, 100,100 - 3D)
    line = input.next()                         # number of defined regions and scale
    nregions = int(line[0])
    scale = float(line[1]) * 1e6                # in xpl all dimensions are in microns

    pnjcond = None

    if setting >= 10:
        raise NotImplementedError("3D structure nor temporal data not implemented yet (%s)" % setting) # TODO

    # Set up symmetry
    axes = ['xy', 'rz'][sym]

    regions = []
    materials = {}
    heats = {}

    # Read each region
    for nr in range(nregions):
        r = Region(axes)

        # number, position, material
        line = input.next()
        n = int(line[0])
        if n == 0:
            r.repeat = int(line[1]) - 1
            r.shift = [0., 0.]
            r.shift[{'pionowo': 1, 'poziomo': 0}[line[3].lower()]] = scale * float(line[2])
            line = input.next()
        r.x0, r.y0, r.x1, r.y1 = [ scale * float(x) for x in  line[1:5] ]
        mat = line[5]
        if mat == "WYPELNIENIE":
            line = input.next()
            line = input.next()
            line = input.next()
            line = input.next()
            continue

        # conductivity
        line = input.next()
        sigma = [float(line[0]), float(line[1])]
        sigma_t = line[2].lower()

        if sigma_t == 'j':
            pnjcond = sigma

        # doping
        line = input.next()
        doping, dopant = 1e-6 * float(line[0]), line[1]

        # heat conductivity
        line = input.next()
        kappa = [float(line[0]), float(line[1])]
        kappa_t = line[2].lower()

        # create custom material if necessary
        if sigma_t not in ('n','p','j') or kappa_t not in ('n','p'):
            material = Material()
            if sigma_t not in ('n','p'):
                material.sigma = sigma
            else:
                material.base = parse_material_name(mat, sigma[0], dopant)
            if kappa_t not in ('n','p'):
                material.kappa = kappa
            else:
                material.base = parse_material_name(mat, kappa[0], dopant)
            found = False
            for mk,mv in materials.items():
                if material == mv:
                    found = True
                    mat = mk
                    break
            if not found:
                if sigma_t in ('n','p') or kappa_t in ('n','p'):
                    mat = unique_material_name() # the given name is the one from database
                while mat in materials and materials[mat] != material:
                    mat = unique_material_name()
                materials[mat] = material
        else:
            mat = parse_material_name(mat, kappa[0], dopant)

        r.material = mat
        if ':' in mat: r.material += "=%g" % doping     # add doping information

        # heat sources
        line = input.next()
        ht = int(line[0])
        if ht == -200:
            r.role = 'active'
        elif ht == 0:
            r.role = 'noheat'
        elif ht == -1:
            r.name = unique_object_name()
            heats[r.name] = float(line[1])
        elif ht != -100:
            raise ValueError("wrong heat source type")

        # save to the list (TODO: make it more clever, using heuristic algorithms to construct stacks and shelves)
        regions.append(r)

    # boundary conditions
    def parse_bc():
        bounds = []
        line = input.next()
        nbc = int(line[0])
        for nc in range(nbc):
            line = input.next()
            x0, y0, x1, y1 = [ scale * float(x) for x in line[0:4] ]
            try: val = float(line[4])
            except IndexError: val = 0.
            except ValueError: val = 0.
            if (x0 == x1):
                bounds.append(dict(dir='vertical', at=x0, start=y0, stop=y1, value=val))
            elif (y0 == y1):
                bounds.append(dict(dir='horizontal', at=y0, start=x0, stop=x1, value=val))
            else:
                raise ValueError("boundary condition line is neither horizontal nor vertical")
        return bounds

    boundaries = {}
    boundaries['voltage'] = parse_bc()
    boundaries['temperature'] = parse_bc()
    boundaries['convection'] = parse_bc()
    boundaries['radiation'] = parse_bc()
    try: boundaries['mesh'] = parse_bc()
    except: pass

    actives = [ r for r in regions if r.role == 'active' ]
    if len(actives) == 1:
        actives[0].name = "active"
        actlevel = True
    elif len(actives) == 0:
        actlevel = False
    else:
        actlevel = sum([ 0.5 * (r.y0 + r.y1) for r in actives ]) / len(actives)

    print("")

    return name, sym, length, axes, materials, regions, heats, boundaries, pnjcond, actlevel


def write_xpl(name, sym, length, axes, materials, regions, heats, boundaries, pnjcond, actlevel):
    '''Write output xpl file'''

    print("Writing %s.xpl" % name)

    ofile = open(name+'.xpl', 'w')

    def out(text):
        ofile.write(text)
        ofile.write('\n')

    out('<plask>\n')

    geometry = ['cartesian2d', 'cylindrical'][sym]
    suffix = ['2D', 'Cyl'][sym]

    if sym == 0:
        geomore = ' length="%s"' % length
    else:
        geomore = ''

    # materials
    materials = list(materials.items())
    materials.sort(key=lambda t: t[0])
    if materials:
        out('<materials>')
        for mn,mat in materials:
            mat.write(ofile, mn)
        out('</materials>\n')

    # geometry
    out('<geometry>\n  <%s name="main" axes="%s"%s>' % (geometry, axes, geomore))
    out('    <container>')
    for r in regions:
        r.write(ofile)
    out('    </container>')
    out('  </%s>\n</geometry>\n' % geometry)

    # default mesh generator
    out('<grids>')
    out('  <generator type="rectilinear2d" method="divide" name="default">\n    <postdiv by0="4" by1="2"/>\n  </generator>')
    out('  <generator type="rectilinear2d" method="divide" name="plots">\n    <postdiv by="10"/>\n  </generator>')
    out('</grids>\n')

    def save_boundaries(name):
        if boundaries[name]:
            out('    <%s>' % name)
            for data in boundaries[name]:
                out(('      <condition value="%(value)s"><place line="%(dir)s"' +
                             ' start="%(start)s" stop="%(stop)s" at="%(at)s"/></condition>') % data)
            out('    </%s>' % name)

    # default solvers
    therm =  boundaries['temperature'] or boundaries['convection'] or boundaries['radiation'] or heats
    electr = boundaries['voltage']

    if therm or electr:
        out('<solvers>')
        if therm:
            out('  <thermal solver="Static%s" name="THERMAL">' % suffix)
            out('    <geometry ref="main"/>\n    <mesh ref="default"/>')
            save_boundaries('temperature')
            save_boundaries('convection')
            save_boundaries('radiation')
            out('  </thermal>')
        if electr:
            out('  <electrical solver="Beta%s" name="ELECTRICAL">' % suffix)
            out('    <geometry ref="main"/>\n    <mesh ref="default"/>')
            if pnjcond is not None:
                out('    <junction pnjcond="%g" heat="wavelength"/>' % pnjcond[1])
            save_boundaries('voltage')
            out('  </electrical>')
        out('</solvers>\n')

    # connections
    if (therm and electr):
        out('<connects>')
        out('  <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>')
        if not heats:
            out('  <connect in="THERMAL.inHeatDensity" out="ELECTRICAL.outHeatDensity"/>')
        else:
            out('  <!-- heats are attached in the script -->')
        out('</connects>\n')

    ## script
    out('<script><![CDATA[\n# Here you may put your calculations. Below there is a sample script (tune it to your needs):\n')


    if heats:
        out('heat_profile = StepProfile(GEO.main)')
        for heat in heats.items():
            out('heat_profile[GEO.%s] = %s' % heat)
        out('fixed_heat = ProviderForHeatDensity%s(heat_profile)' % suffix)

        if electr:
            out('THERMAL.inHeatDensity = ELECTRICAL.outHeatDensity + fixed_heat\n')
        else:
            out('THERMAL.inHeatDensity = fixed_heat\n')

    if electr:
        out('# Adjust the values below!')
        out('ELECTRICAL.inWavelength = 1300.')
        out('ELECTRICAL.js = 1.1')
        out('ELECTRICAL.beta = 19.\n')

    if therm and electr:
        out('terr = 300.')
        out('verr = 10.')
        out('ELECTRICAL.compute(1)')
        out('THERMAL.compute(1)')
        out('while terr > THERMAL.corrlim or verr > ELECTRICAL.corrlim:')
        out('    verr = ELECTRICAL.compute(6)')
        out('    terr = THERMAL.compute(1)')
    else:
        if therm:
            out('THERMAL.compute(0)')
        elif electr:
            out('ELECTRICAL.compute(0)')

    if electr:
        out('\nprint_log(LOG_INFO, "Total current: %.3gmA" % abs(ELECTRICAL.get_total_current()))')

    if therm or electr:
        out('\nplotgrid = MSG.plots(GEO.main.item)')

        if actlevel is not False:
            if actlevel is True:
                out('actbox = GEO.main.get_object_bboxes(GEO.active)[0]')
                out('actlevel = 0.5 * (actbox.lower[1] + actbox.upper[1])')
            else:
                out('actlevel = %g' % actlevel)
            out('actgrid = mesh.Rectilinear2D(plotgrid.axis0, [actlevel])')
            out('plotgrid.axis1.insert(actlevel)')
        out('')

        if therm:
            out('temperature = THERMAL.outTemperature(plotgrid)')
            out('heats = THERMAL.inHeatDensity(plotgrid)')
        if electr:
            out('voltage = ELECTRICAL.outPotential(plotgrid)')
            out('current = ELECTRICAL.outCurrentDensity(plotgrid)')
            if actlevel is not False:
                out('acurrent = ELECTRICAL.outCurrentDensity(actgrid, "SPLINE")')

        h5mode = 'w'
        out('\nif has_hdf5:')
        out('    import sys, os')
        out('    h5file = h5py.File(os.path.splitext(sys.argv[0])[0]+".h5", "w")')
        if therm:
            out('    save_field(temperature, h5file, "Temperature")')
            out('    save_field(heats, h5file, "HeatDensity")')
        if electr:
            out('    save_field(voltage, h5file, "Voltage")')
            out('    save_field(current, h5file, "CurrentDenstity")')

        out('    h5file.close()')

        out('\nif has_pylab:')
        out('    plot_geometry(GEO.main, set_limits=True)')
        out('    defmesh = MSG.default(GEO.main.item)')
        out('    plot_mesh(defmesh, color="0.75")')
        if electr:
            out('    plot_boundary(ELECTRICAL.voltage_boundary, defmesh, color="b", marker="D")')
        if therm:
            out('    plot_boundary(THERMAL.temperature_boundary, defmesh, color="r")')
            out('    plot_boundary(THERMAL.convection_boundary, defmesh, color="g")')
            out('    plot_boundary(THERMAL.radiation_boundary, defmesh, color="y")')
        out('    gcf().canvas.set_window_title("Default mesh")')
        if therm:
            out('\n    figure()')
            out('    plot_field(temperature, 16)')
            out('    colorbar()')
            out('    plot_geometry(GEO.main, color="w")')
            out('    gcf().canvas.set_window_title("Temperature")')
            out('\n    figure()')
            out('    plot_field(heats, 16)')
            out('    colorbar()')
            out('    plot_geometry(GEO.main, color="w")')
            out('    gcf().canvas.set_window_title("Heat sources density")')
        if electr:
            out('\n    figure()')
            out('    plot_field(voltage, 16)')
            out('    colorbar()')
            out('    plot_geometry(GEO.main, color="w")')
            out('    gcf().canvas.set_window_title("Electric potential")')
            if actlevel is not False:
                out('\n    figure()')
                out('    plot(actgrid.axis0, abs(acurrent.array[:,0,1]))')
                out('    xlabel(u"%s [\\xb5m]")' % axes[0])
                out('    ylabel("current density [kA/cm$^2$]")')
                out('    simplemesh = mesh.Rectilinear2D.SimpleGenerator()(GEO.main.item)')
                out('    for x in simplemesh.axis0:')
                out('        axvline(x, ls=":", color="k")')
                out('    xlim(0., simplemesh.axis0[-2])')
                out('    gcf().canvas.set_window_title("Current density in the active region")')
        out('\n    show()')

    out('\n]]></script>\n')

    out('</plask>')


if __name__ == "__main__":

    code = 0

    try:
        iname = sys.argv[1]
    except IndexError:
        sys.stderr.write("Usage: %s input_file_temp.dan\n" % sys.argv[0])
        code = 2
    else:
        dest_dir = os.path.dirname(iname)

        try:
            read = read_dan(iname)
            name = os.path.join(dest_dir, iname[:-9])
            write_xpl(name, *read[1:])
        except Exception as err:
            import traceback as tb
            tb.print_exc()
            #sys.stderr.write("\n%s: %s\n" % (err.__class__.__name__, err))
            code = 1
        else:
            print("\nDone!")

    sys.exit(code)
