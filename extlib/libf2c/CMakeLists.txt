cmake_minimum_required(VERSION 3.14)
project(libf2c C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(CONF ${CMAKE_CONFIGURATION_TYPES})   # uses by MSVC
    STRING(TOUPPER "${CONF}" CONF)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONF} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib")
endforeach()
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)  # all functions will be exported from windows DLL

add_definitions(
    -DUSE_CLOCK
    -DMSDOS
    -DNO_ONEXIT
    -DNO_My_ctype
    -DNO_ISATTY
    -DNO_SSIZE_T
    -DNO_FPINIT
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(vcf2c SHARED
    abort_.c
    backspac.c
    c_abs.c
    c_cos.c
    c_div.c
    c_exp.c
    c_log.c
    c_sin.c
    c_sqrt.c
    cabs.c
    close.c
    d_abs.c
    d_acos.c
    d_asin.c
    d_atan.c
    d_atn2.c
    d_cnjg.c
    d_cos.c
    d_cosh.c
    d_dim.c
    d_exp.c
    d_imag.c
    d_int.c
    d_lg10.c
    d_log.c
    d_mod.c
    d_nint.c
    d_prod.c
    d_sign.c
    d_sin.c
    d_sinh.c
    d_sqrt.c
    d_tan.c
    d_tanh.c
    derf_.c
    derfc_.c
    dfe.c
    dolio.c
    dtime_.c
    due.c
    ef1asc_.c
    ef1cmc_.c
    endfile.c
    erf_.c
    erfc_.c
    err.c
    etime_.c
    exit_.c
    f77_aloc.c
    f77vers.c
    fmt.c
    fmtlib.c
    ftell_.c
    getenv_.c
    h_abs.c
    h_dim.c
    h_dnnt.c
    h_indx.c
    h_len.c
    h_mod.c
    h_nint.c
    h_sign.c
    hl_ge.c
    hl_gt.c
    hl_le.c
    hl_lt.c
    i77vers.c
    i_abs.c
    i_dim.c
    i_dnnt.c
    i_indx.c
    i_len.c
    i_mod.c
    i_nint.c
    i_sign.c
    iio.c
    ilnw.c
    inquire.c
    l_ge.c
    l_gt.c
    l_le.c
    l_lt.c
    lbitbits.c
    lbitshft.c
    lread.c
    lwrite.c
    open.c
    pow_ci.c
    pow_dd.c
    pow_di.c
    pow_hh.c
    pow_ii.c
    pow_ri.c
    pow_zi.c
    pow_zz.c
    r_abs.c
    r_acos.c
    r_asin.c
    r_atan.c
    r_atn2.c
    r_cnjg.c
    r_cos.c
    r_cosh.c
    r_dim.c
    r_exp.c
    r_imag.c
    r_int.c
    r_lg10.c
    r_log.c
    r_mod.c
    r_nint.c
    r_sign.c
    r_sin.c
    r_sinh.c
    r_sqrt.c
    r_tan.c
    r_tanh.c
    rdfmt.c
    rewind.c
    rsfe.c
    rsli.c
    rsne.c
    s_cat.c
    s_cmp.c
    s_copy.c
    s_paus.c
    s_rnge.c
    s_stop.c
    sfe.c
    sig_die.c
    signal_.c
    sue.c
    system_.c
    typesize.c
    uio.c
    uninit.c
    util.c
    wref.c
    wrtfmt.c
    wsfe.c
    wsle.c
    wsne.c
    xwsne.c
    z_abs.c
    z_cos.c
    z_div.c
    z_exp.c
    z_log.c
    z_sin.c
    z_sqrt.c
    arith.h
)

if(INSTALL_DEV)
    install(TARGETS vcf2c RUNTIME DESTINATION bin COMPONENT solvers
            ARCHIVE DESTINATION lib COMPONENT dev)
else()
    install(TARGETS vcf2c RUNTIME DESTINATION bin COMPONENT solvers)
endif()

add_executable(arithchk arithchk.c)

add_custom_command(
    OUTPUT arith.h
    COMMAND arithchk > arith.h
    DEPENDS arithchk
)
