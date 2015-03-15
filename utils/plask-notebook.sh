#!/bin/sh
script=`realpath $0`
plask -i notebook --ipython-dir `dirname $script`/ipython
