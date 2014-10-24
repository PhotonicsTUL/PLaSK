#!/bin/sh
#dirname abspath

grep -r fromTheme `dirname $0`/../gui/* | perl -ne "print \$2.\"\n\" if /fromTheme\(('|\")(.*)\\1\)/" | sort | uniq
