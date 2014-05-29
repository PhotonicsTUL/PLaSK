# module oryginally from: https://github.com/Grive/grive/blob/master/cmake/Modules/FindBFD.cmake

find_library( BFD_LIBRARY	NAMES bfd	PATH /usr/lib /usr/lib64 )

if(BFD_LIBRARY)
    set( BFD_FOUND TRUE )
endif(BFD_LIBRARY)

if (BFD_FOUND)
    message(STATUS "Found libbfd: ${BFD_LIBRARY}")
else (BFD_FOUND)
    message(STATUS "Could NOT find libbfd")
endif(BFD_FOUND)

mark_as_advanced(BFD_LIBRARY BFD_FOUND)
