#!/bin/sh
script=`realpath $0`
plask -i qtconsole --ipython-dir `dirname $script`/ipython
