#!/usr/bin/python
import re, sys
vre = re.compile('#define PLASK_VERSION "(\d{4}\.\d{2}\.\d{2})\.[\da-f]+')
m = vre.search(open('include/plask/version.h').read())
if m is not None: sys.stdout.write(m.group(1))
