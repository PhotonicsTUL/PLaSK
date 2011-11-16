cmake_minimum_required (VERSION 2.8)
project (plask)

add_definitions(-Wall)		# all warnings
add_definitions(-std=c++0x)	# C++0x (C++11) standard

include_directories(.)		# external modules can include plask using <plask/....h>
include_directories(extlibs)	# place for local external libraries

#compress and strip executable (target)
MACRO(compress targetname)
    if (PACK_EXE)
        get_property(apppath TARGET ${targetname} PROPERTY LOCATION)
        if (CMAKE_STRIP)
            add_custom_command(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_STRIP}
                                ARGS ${apppath})
        endif(CMAKE_STRIP)
        if (SELF_PACKER_FOR_EXECUTABLE)
            add_custom_command(TARGET ${targetname} POST_BUILD COMMAND ${SELF_PACKER_FOR_EXECUTABLE}
                                ARGS ${SELF_PACKER_FOR_EXECUTABLE_FLAGS} "-q" ${apppath})    #TODO remove "-q" when SelfPackers will be fixed
        endif(SELF_PACKER_FOR_EXECUTABLE)
    endif(PACK_EXE)
ENDMACRO(compress targetname)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "build/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "build/lib")
file(MAKE_DIRECTORY "build/bin" "build/lib")

option(PACK_EXE "self-pack and strip executables" OFF)  #compress exe
if (PACK_EXE)
    find_package(SelfPackers)
    find_program(CMAKE_STRIP NAMES strip)
endif(PACK_EXE)

# ----------===== plask library =====--------------------------------------------
#creating configuration file for plask library
#CONFIGURE_FILE(plask/config.h.cmake plask/config.h)
file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp plask/*.cxx plask/*.h)
add_library(plask STATIC ${plask_src})
#SET_TARGET_PROPERTIES(plask PROPERTIES COMPILE_FLAGS -fPIC)
