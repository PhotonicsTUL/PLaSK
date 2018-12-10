#!/usr/bin/python
print("Content-type: text/xml\n")

import os
import re
import time

directory = "/var/www/redmine/files/nightly"

print("""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>PLaSK</title>
    <link>http://phys.p.lodz.pl/appcast/plask.xml</link>
    <description>Most recent changes with links to PLaSK updates.</description>
    <language>en</language>""")


def print_enclosure(no, name, os_arch, dist, sep='-'):
    try:
        pre = re.compile(name.format('.*'))
        sources = [f for f in os.listdir(directory) if pre.match(f)]
        sources.sort(reverse=True)
        source = sources[0]
        name, ver = source[:-4].split(sep)[:2]
        filepath = os.path.join(directory, source)
        pubdate = time.ctime(os.path.getctime(filepath))
        length = os.path.getsize(filepath)
        print("""    <item>
        <title>{name} {ver}</title>
        <pubDate>{pubdate} +0000</pubDate>""".format(**locals()))
        if dist is not None:
            dist_info = '\n            sparkle:dist="{}"'.format(dist)
        else:
            dist_info = ''
        print("""        <enclosure
            url="https://phys.p.lodz.pl/redmine/attachments/download/{no}/{source}"
            length="{length}"
            sparkle:version="{ver}"
            sparkle:os="{os_arch}"{dist_info}
            type="application/octet-stream"
        />
    </item>""".format(**locals()))
    except Exception as err:
        print("    <!-- {}: {} -->".format(name, err))


print_enclosure(1, 'PLaSK-{}-win64-py36.exe', 'windows-x64', None)
print_enclosure(1, 'PLaSK-{}-win64-py36.exe', 'windows-x64', 'python-3.6')
print_enclosure(7, 'PLaSK-{}-win64-py37.exe', 'windows-x64', 'python-3.7')
print_enclosure(2, 'plask_{}_bionic_amd64.deb', 'linux-x64', 'ubuntu-bionic', sep='_')
print_enclosure(2, 'plask_{}_bionic_amd64.deb', 'linux-x64', 'neon-bionic', sep='_')
print_enclosure(3, 'plask_{}_xenial_amd64.deb', 'linux-x64', 'ubuntu-xenial', sep='_')
print_enclosure(3, 'plask_{}_xenial_amd64.deb', 'linux-x64', 'neon-xenial', sep='_')
print_enclosure(4, 'plask_{}_cosmic_amd64.deb', 'linux-x64', 'ubuntu-cosmic', sep='_')
#print_enclosure(7, 'plask-{}-rhel7.x86_64.rpm', 'linux-x64', 'redhat-7')
#print_enclosure(7, 'plask-{}-rhel7.x86_64.rpm', 'linux-x64', 'centos-7')
print_enclosure(8, 'plask_{}_debian_amd64.deb', 'linux-x64', 'debian-9', sep='_')

#print("""      <sparkle:version>{ver}</sparkle:version>
#      <link>https://phys.p.lodz.pl/redmine/projects/plask/files/</link>""".format(ver=ver))

print("""  </channel>
</rss>""")
