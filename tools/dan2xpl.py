#!/bin/python
'''
    This tool converts RPSMES .dan file to PLaSK .xpl file.

    Usage:

      dan2xpl input_file_temp.dan

    If the conversion is successful, the script writes
'''
import sys



try:
    input = open(sys.argv[1])
except IndexError:
    sys.stderr.write("Usage: "+sys.argv[0]+" input_file_temp.dan\n")
    sys.exit(1)
except IOError:
    sys.stderr.write("Cannot read file "+sys.argv[1]+"\n")
    sys.exit(2)

try: # Reading the file

    # Header
    name = input.next().strip()                         # structure name (will be used for output file)
    matdb = input.next().strip()                        # materials database spec (All by default)
    line = input.next().split('\t')                     # symmetry (0: Cartesian2D, 1: cylindrical) type and height (not used)
    symmetry = ['cartesian2d', 'cylindrical'][int(line[0])]
    input.next()                                        # horizontal something (not used)
    line = input.next().split('\t')                     # number of defined regions and scale
    nregions, scale = int(line[0]), float(line[1])

    # Read each region

except:
    sys.stderr.write("Error reading file "+sys.argv[1]+"\n")
    sys.exit(3)
