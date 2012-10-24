#!/bin/python
'''
    This tool converts RPSMES .dan file to PLaSK .xpl file.

    Usage:

      dan2xpl input_file_temp.dan

    If the conversion is successful, the script writes
'''
import sys



try:
    ifile = open(sys.argv[1])
except IndexError:
    sys.stderr.write("Usage: "+sys.argv[0]+" input_file_temp.dan\n")
    sys.exit(1)
except IOError:
    sys.stderr.write("Cannot read file "+sys.argv[1]+"\n")
    sys.exit(2)


# Set-up generator, which skips empty lines, strips the '\n' character, and splits line by tabs
def Input(ifile):
    for line in ifile:
        if line.strip(): yield line[:-1].split('\t')
input = Input(ifile)

try: # Reading the file

    # Header
    name = input.next()[0]                              # structure name (will be used for output file)
    matdb = input.next([0]                              # materials database spec (All by default)
    line = input.next()                                 # symmetry (0: Cartesian2D, 1: cylindrical) type and height (not used)
    symmetry = ['cartesian2d', 'cylindrical'][int(line[0])]
    input.next()                                        # horizontal something (not used)
    line = input.next()                                 # number of defined regions and scale
    nregions, scale = int(line[0]), float(line[1])

    # Read each region
    for i in range(nregions):
        line = input.next()
        n = int(line[0]); if n != i: throw ValueError("wrong region number")
        x1, y1, x2, y2 = map(float, line[1:5])
        material = line[6]

except Exception as err:
    sys.stderr.write("Error reading file "+sys.argv[1]+":\n  " + type(err)+": "+err.message+"\n")
    sys.exit(3)
