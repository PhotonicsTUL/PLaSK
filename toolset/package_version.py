#!/usr/bin/python
import re, sys
vre = re.compile(r'#define PLASK_VERSION "(.+)\.[\da-f]+"')
m = vre.search(open('include/plask/version.h').read())
if m is not None: sys.stdout.write(m.group(1))
