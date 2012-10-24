#!/usr/bin/python
'''
    This tool converts RPSMES .dan file to PLaSK .xpl file.

    Usage:

      dan2xpl input_file_temp.dan

    If the conversion is successful, the script writes
'''
import sys
import traceback


try:
    ifile = open(sys.argv[1])
except IndexError:
    sys.stderr.write("Usage: "+sys.argv[0]+" input_file_temp.dan\n")
    sys.exit(3)
except IOError:
    sys.stderr.write("Cannot read file "+sys.argv[1]+"\n")
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
line = input.next()                                 # symmetry (0: Cartesian2D, 1: cylindrical) type and height (not used)
symmetry = ['cartesian2d', 'cylindrical'][int(line[0])]
input.next()                                        # horizontal something (not used)
line = input.next()                                 # number of defined regions and scale
nregions = int(line[0])
try: scale = float(line[1])
except ValueError: scale = float(line[2])

regions = []
materials = {}

# Read each region
for i in range(nregions):
    r = {}

    # number, position, material
    line = input.next()
    n = int(line[0])
    if n == 0:
        repeats = int(line[1])
        shift = float(line[2])
        si = {'pionowo': 1, 'poziomo': 0}[line[3].lower()]
        line = input.next()
    else:
        repeats = 1
        shift = 0.
        si = 0
    x1, y1, x2, y2 = map(float, line[1:5])
    r['lower'], r['upper'] = [x1, y1], [x2, y2]
    mat = line[5]
    r['mat'] = mat

    # TODO: handle situations where there is material composition instead of sigma and kappa

    # conductivity
    line = input.next()
    r['sigma_x'], r['sigma_y'] = float(line[0]), float(line[1])
    r['sigma_t'] = line[2]

    # doping
    line = input.next()
    r['doping'], r['dopant'] = float(line[0]), line[1]

    # heat conductivity
    line = input.next()
    r['kappa_x'], r['kappa_y'] = float(line[0]), float(line[1])
    r['kappa_t'] = line[2]

    # heat sources
    line = input.next()
    ht = int(line[0])
    r['heat'] = 0
    r['role'] = None
    if ht == -200: r['role'] == 'active'
    elif ht == 0: r['role'] == 'insulator'
    elif ht == -1: r['heat'] = float(line[1])
    elif ht != -100: raise ValueError("wrong heat source type")

    # save to the list (TODO: make it more clever, using heuristic algorithms to construct stacks and shelves)
    resions.append(r)
    for j in range(repeats-1):
        r = r.copy()
        r['lower'][si] += shift
        r['upper'][si] += shift
        resions.append(r)

# Read boundary conditions