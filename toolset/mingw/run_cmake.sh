#!/bin/sh

if [ "$MINGW_ROOT" = "" ]; then
    MINGW_ROOT=/usr/local/mingw
fi
    
arch=i686

case "$arch" in
    i686) mingw=mingw32;;
    amd64) mingw=mingw-w64;;
esac

rm CMakeCache.txt 2>/dev/null

cmake .. -G Ninja \
	 -DCMAKE_TOOLCHAIN_FILE=$MINGW_ROOT/usr/$arch-pc-$mingw/share/cmake/mxe-conf.cmake \
	 -DNUMPY_INCLUDE_DIRS=$MINGW_ROOT/$arch/python27/Lib/site-packages/numpy/core/include \
	 -DBUILD_TESTING=OFF \
	 -DCMAKE_INSTALL_PREFIX=$MINGW_ROOT/dist \
	 -DRUN_TESTS=OFF \
	 -DUSE_OMP=OFF \
	 -DGSL_CONFIG_EXECUTABLE=$MINGW_ROOT/usr/$arch-pc-$mingw/bin/gsl-config \
	 -DDLLS_DIR=$MINGW_ROOT/$arch/dlls \
         -DUSE_FFTW=OFF \
	 -DCMAKE_BUILD_TYPE=Release	
#	 -DCMAKE_BUILD_TYPE=Debug	
