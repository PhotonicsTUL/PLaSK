/* cosq1i.f -- translated by f2c (version 20100827).
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

static integer c__3 = 3;
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

/* Subroutine */ int cosq1i_(integer *n, doublereal *wsave, integer *lensav, 
	integer *ier)
{
    /* System generated locals */
    integer i__1;

    /* Builtin functions */
    double log(doublereal), atan(doublereal), cos(doublereal);

    /* Local variables */
    integer k;
    doublereal fk, dt, pih;
    integer ier1, lnsv;
    extern /* Subroutine */ int rfft1i_(integer *, doublereal *, integer *, 
	    integer *), xerfft_(char *, integer *, ftnlen);


    /* Parameter adjustments */
    --wsave;

    /* Function Body */
    *ier = 0;
    if (*lensav < (*n << 1) + (integer) (log((doublereal) (*n)) / log(2.)) + 
	    4) {
	*ier = 2;
	xerfft_("COSQ1I", &c__3, (ftnlen)6);
	goto L300;
    }

    pih = atan(1.) * 2.;
    dt = pih / (doublereal) (*n);
    fk = 0.;
    i__1 = *n;
    for (k = 1; k <= i__1; ++k) {
	fk += 1.;
	wsave[k] = cos(fk * dt);
/* L101: */
    }
    lnsv = *n + (integer) (log((doublereal) (*n)) / log(2.)) + 4;
    rfft1i_(n, &wsave[*n + 1], &lnsv, &ier1);
    if (ier1 != 0) {
	*ier = 20;
	xerfft_("COSQ1I", &c_n5, (ftnlen)6);
    }
L300:
    return 0;
} /* cosq1i_ */

