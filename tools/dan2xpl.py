#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
    This tool converts RPSMES .dan file to PLaSK .xpl file.

    Usage:

      dan2xpl input_file_temp.dan

    If the conversion is successful, the script writes
'''
import sys
#import traceback

from numpy import *


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
        output.write('      <child %(a0)s="%(x0).4f" %(a1)s="%(y0).4f"><block %(a0)s="%(w)s" %(a1)s="%(h)s" material="%(material)s"%(more)s/></child>\n' % locals())
        if self.repeat:
            x = self.x0
            y = self.y0
            for i in range(self.repeat):
                x += self.shift[0]
                y += self.shift[1]
                output.write('      <child %(a0)s="%(x).4f" %(a1)s="%(y).4f"><ref name="%(name)s"/></child>\n' % locals())


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



def parse_material_name(mat, comp, dopant):
    if mat[0] == mat[0].lower(): elements = ['']
    else: elements = []
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

    print "Reading %s:" % fname

    try:
        ifile = open(iname)
    except IOError:
        sys.stderr.write("Cannot read file \n" % iname)
        sys.exit(2)


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
    line = input.next()                         # symmetry (0: Cartesian2D, 1: cylindrical) type and length (not used)
    sym = int(line[0])
    setting = int(input.next()[0])              # setting (10,11 - temporal calculations, 100,100 - 3D)
    line = input.next()                         # number of defined regions and scale
    nregions = int(line[0])
    scale = float(line[1]) * 1e6                # in xpl all dimensions are in microns

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
        sigma = array([float(line[0]), float(line[1])])
        sigma_t = line[2].lower()

        # doping
        line = input.next()
        doping, dopant = float(line[0]), line[1]

        # heat conductivity
        line = input.next()
        kappa = array([float(line[0]), float(line[1])])
        kappa_t = line[2].lower()

        # create custom material if necessary
        if sigma_t not in ('n','p','j') or kappa_t not in ('n','p'):
            material = Material()
            if sigma_t not in ('n','p','j'):
                material.sigma = sigma
            else:
                material.base = parse_material_name(mat, sigma[0], dopant)
            if kappa_t not in ('n','p'):
                material.kappa = kappa
            else:
                material.base = parse_material_name(mat, kappa[0], dopant)
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
            r.role = 'insulator'
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
            bounds = []
            if (x0 == x1):
                bounds.append(dict(dir='vertical', at=x0, start=y0, stop=y1, value=val))
            elif (y0 == y1):
                bounds.append(dict(dir='horizontal', at=y0, start=x0, stop=x1, value=val))
            else:
                raise ValueError("boundary condition line is neither horizontal nor vertical")
        return bounds

    boundaries = {}
    boundaries['potential'] = parse_bc()
    boundaries['temperature'] = parse_bc()
    boundaries['convection'] = parse_bc()
    boundaries['radiation'] = parse_bc()
    try: boundaries['mesh'] = parse_bc()
    except: pass

    return name, sym, axes, materials, regions, heats, boundaries


def write_xpl(name, sym, axes, materials, regions, heats, boundaries):
    '''Write output xpl file'''

    print "Writing %s.xpl" % name

    ofile = open(name+'.xpl', 'w')
    ofile.write('<plask>\n\n')

    geometry = ['cartesian2d', 'cylindrical'][sym]
    suffix = ['2D', 'Cyl'][sym]

    # materials
    if materials:
        ofile.write('<materials>\n')
        for mat in materials:
            materials[mat].write(ofile, mat)
        ofile.write('</materials>\n\n')

    # geometry
    ofile.write('<geometry>\n  <%s name="main" axes="%s">\n' % (geometry, axes))
    ofile.write('    <container>\n')
    for r in regions:
        r.write(ofile)
    ofile.write('    </container>\n')
    ofile.write('  </%s>\n</geometry>\n\n' % geometry)

    # default mesh generator
    ofile.write('<grids>\n  <generator type="rectilinear2d" method="divide" name="default">\n')
    ofile.write('    <postdiv by="2"/>\n  </generator>\n</grids>\n\n')

    def save_boundaries(name):
        if boundaries[name]:
            ofile.write('    <%s>\n' % name)
            for data in boundaries[name]:
                ofile.write(('      <condition value="%(value)s"><place line="%(dir)s"' +
                            ' start="%(start)s" stop="%(stop)s" at="%(at)s"/></condition>\n') % data)
            ofile.write('    </%s>\n' % name)

    # default solvers
    therm =  boundaries['temperature'] or boundaries['convection'] or boundaries['radiation'] or heats
    electr = boundaries['potential']

    if therm or electr:
        ofile.write('<solvers>\n')
        if therm:
            ofile.write('  <thermal solver="Fem%s" name="THERMAL">\n' % suffix)
            ofile.write('    <geometry ref="main"/>\n    <mesh ref="default"/>\n')
            save_boundaries('temperature')
            save_boundaries('convection')
            save_boundaries('radiation')
            ofile.write('  </thermal>\n')
        if electr:
            #ofile.write('  <electrical solver="Fem%s" name="electrical">' % suffix)
            ofile.write('  <electrical lib="fem2d" solver="%sFEM" name="ELECTRICAL">\n' % ['Cartesian', 'Cylindrical'][sym]) # TODO change to above
            ofile.write('    <geometry ref="main"/>\n    <mesh ref="default"/>\n')
            ofile.write('    <wavelength value="1300"/>\n') # TODO
            save_boundaries('potential')
            ofile.write('  </electrical>\n')
        ofile.write('</solvers>\n\n')

    # connections
    if (therm and electr) or heats:
        ofile.write('<connects>\n')
        if (therm and electr):
            ofile.write('  <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>\n')
        if heats:
            ofile.write('  <profile in="THERMAL.inHeatDensity">\n')
            for heat in heats.items():
                ofile.write('    <step object="%s" value="%s"/>\n' % heat)
            ofile.write('  </profile>\n')
            if electr:
                # TODO use sum provider when it gets fully supported in PLaSK
                print "WARNING: Electrical model present, but only manually specified heats are provided to thermal model!"
        elif electr:
            ofile.write('  <connect in="THERMAL.inHeatDensity" out="ELECTRICAL.outHeatDensity"/>\n')
        ofile.write('</connects>\n\n')

    ## script
    #ofile.write('<script>\n# Here you may put your calculations\n\n')
    #ofile.write('</script>\n\n')

    ofile.write('</plask>\n')


if __name__ == "__main__":

    try:
        iname = sys.argv[1]
    except IndexError:
        sys.stderr.write("Usage: %s input_file_temp.dan\n" % sys.argv[0])
        sys.exit(3)

    write_xpl(*read_dan(iname))

    print "Done!"