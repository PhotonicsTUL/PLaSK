# After calling 'cmake' use:
#  'make' to build all,
#  'make test' to run tests.

cmake_minimum_required (VERSION 2.8)
project (plask)

add_definitions(-Wall)		# all warnings
add_definitions(-std=c++0x)	# C++0x (C++11) standard

include_directories(.)		# external modules can include plask using <plask/....h>
include_directories(extlibs)	# place for local external libraries

# compress and strip executable (target)
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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")
file(MAKE_DIRECTORY "bin" "lib")

option(PACK_EXE "self-pack and strip executables" OFF)  #compress exe
if (PACK_EXE)
    find_package(SelfPackers)
    find_program(CMAKE_STRIP NAMES strip)
endif(PACK_EXE)

# ----------===== plask library =====--------------------------------------------
# creating configuration file for plask library
#CONFIGURE_FILE(plask/config.h.cmake plask/config.h)

file(GLOB_RECURSE plask_headers FOLLOW_SYMLINKS plask/*.h)
file(GLOB_RECURSE plask_src FOLLOW_SYMLINKS plask/*.cpp plask/*.cxx plask/*.h)
add_library(plask STATIC ${plask_src})

#SET_TARGET_PROPERTIES(plask PROPERTIES COMPILE_FLAGS -fPIC)

# ----------===== unit tests =====--------------------------------------------
option(MAKE_TESTS "build and run unit tests" ON)
if(MAKE_TESTS)
	ENABLE_TESTING()
	FIND_PACKAGE(Boost COMPONENTS unit_test_framework)
	if(Boost_FOUND)
		INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
		file(GLOB_RECURSE test_plask_src FOLLOW_SYMLINKS tests/plask/*.cpp tests/plask/*.cxx tests/plask/*.h)
		ADD_EXECUTABLE(test_plask ${test_plask_src})
		TARGET_LINK_LIBRARIES(test_plask ${Boost_LIBRARIES})	#plask
		ADD_TEST(plask ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_plask)	#can be execute by ctest
		#add_custom_command(TARGET test_plask POST_BUILD COMMAND test_plask)	#execute when build
		#add_custom_target(tests COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS test_plask)
	else(Boost_FOUND)
		message("Boost Test Library not found. Build without unit tests.")
	endif(Boost_FOUND)
endif(MAKE_TESTS)

# ----------===== documentation (api) =====--------------------------------------------
option(BUILD_DOCUMENTATION "build doxygen documentation" OFF)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        ADD_CUSTOM_TARGET(doc ALL ${DOXYGEN} ${CMAKE_SOURCE_DIR}/Doxyfile DEPENDS ${CMAKE_SOURCE_DIR}/Doxyfile ${plask_headers})
    else(DOXYGEN_FOUND)
        message("Doxygen not found. Build without documentation.")
    endif(DOXYGEN_FOUND)
endif(BUILD_DOCUMENTATION)

# ----------===== install =====--------------------------------------------
install(TARGETS plask LIBRARY DESTINATION lib ARCHIVE DESTINATION lib COMPONENT libplask)
install(FILES ${plask_headers} DESTINATION include/plask COMPONENT libplask-dev)
