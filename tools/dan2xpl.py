#!/usr/bin/python
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
_unique_object_name_counter = 0
def unique_object_name():
    global _unique_object_name_counter
    _unique_object_name_counter += 1
    return "obj%d" % _unique_object_name_counter

_unique_material_name_counter = 0
def unique_material_name():
    global _unique_material_name_counter
    _unique_material_name_counter += 1
    return "mat%d" % _unique_material_name_counter

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
        locals().update(self.__dict__)
        w = '%.6g' % (self.x1 - self.x0)
        h = '%.6g' % (self.y1 - self.y0)
        more = ""
        if self.repeat and not self.name: self.name = unique_object_name()
        if self.role: more += ' role="%s"' % self.role
        if self.name: more += ' name="%s"' % self.name
        output.write('      <child %(a0)s="%(x0)s" %(a1)s="%(y0)s"><block %(a0)s="%(w)s" %(a1)s="%(h)s" material="%(material)s"%(more)s/></child>\n' % locals())
        if self.repeat:
            x = self.x0
            y = self.y0
            for i in range(self.repeat):
                x += self.shift[0]
                y += self.shift[1]
                output.write('      <child %(a0)s="%(x)s" %(a1)s="%(y)s"><ref name="%(name)s"/></child>\n' % locals())


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

    try:
        ifile = open(iname)
    except IOError:
        sys.stderr.write("Cannot read file \n" % iname)
        sys.exit(2)


    # Set-up generator, which skips empty lines, strips the '\n' character, and splits line by tabs
    def Input(ifile):
        for line in ifile:
            print line[:-1]
            if line.strip(): yield line[:-1].split()
    input = Input(ifile)

    # Header
    name = input.next()[0]                              # structure name (will be used for output file)
    matdb = input.next()[0]                             # materials database spec (All by default)
    sym = int(input.next()[0])                          # symmetry (0: Cartesian2D, 1: cylindrical) type and height (not used)
    input.next()                                        # horizontal something (not used)
    line = input.next()                                 # number of defined regions and scale
    nregions = int(line[0])
    scale = float(line[1]) * 1e6                        # in xpl all dimensions are in microns

    # Set up symmetry
    geometry = ['cartesian2d', 'cylindrical'][sym]
    axes = ['xy', 'rz'][sym]

    regions = []
    materials = {}
    heats = {}

    # Read each region
    for i in range(nregions):
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
        if sigma_t not in ('n','p') or kappa_t not in ('n','p'):
            material = Material()
            if sigma_t not in ('n','p'):
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
            mat = parse_material_name(mat, sigma[0], dopant)

        r.material = mat
        if ':' in mat: r.material += "=%g" % doping     # add doping information

        # heat sources
        line = input.next()
        ht = int(line[0])
        if ht == -200:
            r.role == 'active'
        elif ht == 0:
            r.role == 'insulator'
        elif ht == -1:
            r.name = unique_material_name()
            heats[r.name] = float(line[1])
        elif ht != -100:
            raise ValueError("wrong heat source type")

        # save to the list (TODO: make it more clever, using heuristic algorithms to construct stacks and shelves)
        regions.append(r)

    # boundary conditions

    return name, geometry, axes, materials, regions, heats


def write_xpl(name, geometry, axes, materials, regions, heats):
    '''Write output xpl file'''

    ofile = open(name+'.xpl', 'w')
    ofile.write('<plask>\n\n')

    if materials:
        ofile.write('<materials>\n')
        for mat in materials:
            materials[mat].write(ofile, mat)
        ofile.write('</materials>\n\n')

    ofile.write('<geometry>\n  <%s name="main" axes="%s">\n' % (geometry, axes))
    ofile.write('    <container>\n')
    for r in regions:
        r.write(ofile)
    ofile.write('    <container>\n')
    ofile.write('  </%s>/n</geometry>\n\n' % geometry)

    ofile.write('</plask>\n')


if __name__ == "__main__":

    try:
        iname = sys.argv[1]
    except IndexError:
        sys.stderr.write("Usage: %s input_file_temp.dan\n" % sys.argv[0])
        sys.exit(3)

    write_xpl(*read_dan(iname))