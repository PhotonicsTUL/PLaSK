#!/usr/bin/python
import re, sys
vre = re.compile('#define PLASK_VERSION "(.*)"')
m = vre.search(open('include/plask/version.h').read())
if m is not None: sys.stdout.write(m.group(1))
