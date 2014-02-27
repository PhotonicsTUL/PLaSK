#include <sstream>

#ifdef __cplusplus
extern "C" {
#endif

#include "f2c.h"

/**
 *  Purpose
 *  =======
 *
 *  XERFFT  is an error handler for library FFTPACK version 5.1 routines.
 *  It is called by an FFTPACK 5.1 routine if an input parameter has an
 *  invalid value.  A message is printed and execution stops.
 *
 *  Installers may consider modifying the STOP statement in order to
 *  call system-specific exception-handling facilities.
 *
 *  Arguments
 *  =========
 *
 *  SRNAME  (input) CHARACTER*6
 *          The name of the routine which called XERFFT.
 *
 *  INFO    (input) INTEGER
 *          When a single  invalid parameter in the parameter list of
 *          the calling routine has been detected, INFO is the position
 *          of that parameter.  In the case when an illegal combination
 *          of LOT, JUMP, N, and INC has been detected, the calling
 *          subprogram calls XERFFT with INFO = -1.
 */
int xerfft_(char *srname, integer *info, ftnlen)
{
    std::stringstream msg;

    if (*info >= 1)
        msg << "On entry to " << srname << " parameter number " << (*info) << " has an illegal value";
    else if (*info == -1)
        msg << "On entry to " << srname << " parameters LOT, JUMP, N and INC are inconsistent";
    else if (*info == -2)
        msg << "On entry to " << srname << " parameter L is greater than LDIM";
    else if (*info == -3)
        msg << "On entry to " << srname << " parameter M is greater than MDIM";
    else if (*info == -5)
        msg << "Within " << srname << " input error returned by lower level routine";
    else if (*info == -6)
        msg << "On entry to " << srname << " parameter LDIM is less than 2*(L/2+1)";

    throw msg.str();

    return 0;
}

#ifdef __cplusplus
}
#endif
