#!/bin/sh
#dirname abspath

grep -r fromTheme `dirname $0`/../gui/* | perl -ne "print \$1.\"\n\" if /fromTheme\('(.*)'\)/" | sort | uniq