#!/usr/bin/python
# code comes from: http://scons.org/wiki/EmbedManifestIntoTarget
import sys

# Defaults
name    = "Microsoft.VC90"
version = "9.0.21022.8"
key     = "1fc8b3b9a1e18e3b"

try:
    import msvcrt

    name    = msvcrt.LIBRARIES_ASSEMBLY_NAME_PREFIX
    version = msvcrt.CRT_ASSEMBLY_VERSION
    key     = msvcrt.VC_ASSEMBLY_PUBLICKEYTOKEN

except:
    pass

template = '''\
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"></requestedExecutionLevel>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="%s.CRT" version="%s" processorArchitecture="*" publicKeyToken="%s"></assemblyIdentity>
    </dependentAssembly>
  </dependency>
</assembly>''' %(name, version, key)

# Write it out to a file.
fout = open("msvcrt.manifest", "w")
fout.write(template)
fout.close()

fout = open("msvcr_exe.rc", "w")
fout.write("""#include "winuser.h"
1 RT_MANIFEST  msvcrt.manifest
""")
fout.close()

fout = open("msvcr_dll.rc", "w")
fout.write("""#include "winuser.h"
2 RT_MANIFEST  msvcrt.manifest
""")
fout.close()
