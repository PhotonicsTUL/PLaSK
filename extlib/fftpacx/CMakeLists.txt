project(fftpacx C CXX)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(NOT MSVC)
    add_definitions("-Wno-shift-op-parentheses")
endif(NOT MSVC)

file(GLOB_RECURSE fftpacx_headers FOLLOW_SYMLINKS *.h)

add_library(fftpacx SHARED
            c1f2kb.c c1f2kf.c c1f3kb.c c1f3kf.c c1f4kb.c c1f4kf.c c1f5kb.c
            c1f5kf.c c1fgkb.c c1fgkf.c c1fm1b.c c1fm1f.c cfft1b.c cfft1f.c
            cfft1i.c cfft2b.c cfft2f.c cfft2i.c cfftmb.c cfftmf.c cfftmi.c
            cmf2kb.c cmf2kf.c cmf3kb.c cmf3kf.c cmf4kb.c cmf4kf.c cmf5kb.c
            cmf5kf.c cmfgkb.c cmfgkf.c cmfm1b.c cmfm1f.c cosq1b.c cosq1f.c
            cosq1i.c cosqb1.c cosqf1.c cosqmb.c cosqmf.c cosqmi.c cost1b.c
            cost1f.c cost1i.c costb1.c costf1.c costmb.c costmf.c costmi.c
            factor.c mcfti1.c mcsqb1.c mcsqf1.c mcstb1.c mcstf1.c mradb2.c
            mradb3.c mradb4.c mradb5.c mradbg.c mradf2.c mradf3.c mradf4.c
            mradf5.c mradfg.c mrftb1.c mrftf1.c mrfti1.c msntb1.c msntf1.c
            r1f2kb.c r1f2kf.c r1f3kb.c r1f3kf.c r1f4kb.c r1f4kf.c r1f5kb.c
            r1f5kf.c r1fgkb.c r1fgkf.c r2w.c rfft1b.c rfft1f.c rfft1i.c rfft2b.c
            rfft2f.c rfft2i.c rfftb1.c rfftf1.c rffti1.c rfftmb.c rfftmf.c rfftmi.c
            sinq1b.c sinq1f.c sinq1i.c sinqmb.c sinqmf.c sinqmi.c sint1b.c sint1f.c
            sint1i.c sintb1.c sintf1.c sintmb.c sintmf.c sintmi.c tables.c w2r.c
            xercon.c xerfft.cpp
            ${fftpacx_headers}
)

if(WIN32)
    install(TARGETS fftpacx RUNTIME DESTINATION bin COMPONENT solvers
                            ARCHIVE DESTINATION lib COMPONENT dev)
else(WIN32)  
    install(TARGETS fftpacx LIBRARY DESTINATION lib COMPONENT solvers)
endif(WIN32)

install(FILES fftpacx.h DESTINATION include COMPONENT dev)
