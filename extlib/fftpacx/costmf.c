/* costmf.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Table of constant values */

static integer c__6 = 6;
static integer c__8 = 8;
static integer c__10 = 10;
static integer c_n1 = -1;
static integer c_n5 = -5;

/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*     *                                                               * */
/*     *                  copyright (c) 2011 by UCAR                   * */
/*     *                                                               * */
/*     *       University Corporation for Atmospheric Research         * */
/*     *                                                               * */
/*     *                      all rights reserved                      * */
/*     *                                                               * */
/*     *                     FFTPACK  version 5.1                      * */
/*     *                                                               * */
/*     *                 A Fortran Package of Fast Fourier             * */
/*     *                                                               * */
/*     *                Subroutines and Example Programs               * */
/*     *                                                               * */
/*     *                             by                                * */
/*     *                                                               * */
/*     *               Paul Swarztrauber and Dick Valent               * */
/*     *                                                               * */
/*     *                             of                                * */
/*     *                                                               * */
/*     *         the National Center for Atmospheric Research          * */
/*     *                                                               * */
/*     *                Boulder, Colorado  (80307)  U.S.A.             * */
/*     *                                                               * */
/*     *                   which is sponsored by                       * */
/*     *                                                               * */
/*     *              the National Science Foundation                  * */
/*     *                                                               * */
/*     * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* Subroutine */ int costmf_(integer *lot, integer *jump, integer *n, integer 
	*inc, doublereal *x, integer *lenx, doublereal *wsave, integer *
	lensav, doublereal *work, integer *lenwrk, integer *ier)
{
    /* System generated locals */
    integer x_dim1, x_offset;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer iw1, ier1;
    extern /* Subroutine */ int mcstf1_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *,
	     integer *);
    extern logical xercon_(integer *, integer *, integer *, integer *);
    extern /* Subroutine */ int xerfft_(char *, integer *, ftnlen);


    /* Parameter adjustments */
    x_dim1 = *inc;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --wsave;
    --work;

    /* Function Body */
    *ier = 0;

    if (*lenx < (*lot - 1) * *jump + *inc * (*n - 1) + 1) {
	*ier = 1;
	xerfft_("COSTMF", &c__6, (ftnlen)6);
	goto L100;
    } else if (*lensav < (*n << 1) + (integer) (log((doublereal) (*n)) / log(
	    2.)) + 4) {
	*ier = 2;
	xerfft_("COSTMF", &c__8, (ftnlen)6);
	goto L100;
    } else if (*lenwrk < *lot * (*n + 1)) {
	*ier = 3;
	xerfft_("COSTMF", &c__10, (ftnlen)6);
	goto L100;
    } else if (! xercon_(inc, jump, n, lot)) {
	*ier = 4;
	xerfft_("COSTMF", &c_n1, (ftnlen)6);
	goto L100;
    }

    iw1 = *lot + *lot + 1;
    mcstf1_(lot, jump, n, inc, &x[x_offset], &wsave[1], &work[1], &work[iw1], 
	    &ier1);
    if (ier1 != 0) {
	*ier = 20;
	xerfft_("COSTMF", &c_n5, (ftnlen)6);
    }

L100:
    return 0;
} /* costmf_ */

