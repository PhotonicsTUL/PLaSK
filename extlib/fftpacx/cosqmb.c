/* cosqmb.f -- translated by f2c (version 20100827).
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

/* Subroutine */ int cosqmb_(integer *lot, integer *jump, integer *n, integer 
	*inc, doublereal *x, integer *lenx, doublereal *wsave, integer *
	lensav, doublereal *work, integer *lenwrk, integer *ier)
{
    /* System generated locals */
    integer x_dim1, x_offset, i__1, i__2;

    /* Builtin functions */
    double log(doublereal), sqrt(doublereal);

    /* Local variables */
    integer m;
    doublereal x1;
    integer lj, ier1;
    extern /* Subroutine */ int mcsqb1_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, integer *);
    doublereal ssqrt2;
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
	xerfft_("COSQMB", &c__6, (ftnlen)6);
	goto L300;
    } else if (*lensav < (*n << 1) + (integer) (log((doublereal) (*n)) / log(
	    2.)) + 4) {
	*ier = 2;
	xerfft_("COSQMB", &c__8, (ftnlen)6);
	goto L300;
    } else if (*lenwrk < *lot * *n) {
	*ier = 3;
	xerfft_("COSQMB", &c__10, (ftnlen)6);
	goto L300;
    } else if (! xercon_(inc, jump, n, lot)) {
	*ier = 4;
	xerfft_("COSQMB", &c_n1, (ftnlen)6);
	goto L300;
    }

    lj = (*lot - 1) * *jump + 1;
    if ((i__1 = *n - 2) < 0) {
	goto L101;
    } else if (i__1 == 0) {
	goto L102;
    } else {
	goto L103;
    }
L101:
    i__1 = lj;
    i__2 = *jump;
    for (m = 1; i__2 < 0 ? m >= i__1 : m <= i__1; m += i__2) {
	x[m + x_dim1] = x[m + x_dim1];
/* L201: */
    }
    return 0;
L102:
    ssqrt2 = 1. / sqrt(2.);
    i__2 = lj;
    i__1 = *jump;
    for (m = 1; i__1 < 0 ? m >= i__2 : m <= i__2; m += i__1) {
	x1 = x[m + x_dim1] + x[m + (x_dim1 << 1)];
	x[m + (x_dim1 << 1)] = ssqrt2 * (x[m + x_dim1] - x[m + (x_dim1 << 1)])
		;
	x[m + x_dim1] = x1;
/* L202: */
    }
    return 0;
L103:
    mcsqb1_(lot, jump, n, inc, &x[x_offset], &wsave[1], &work[1], &ier1);
    if (ier1 != 0) {
	*ier = 20;
	xerfft_("COSQMB", &c_n5, (ftnlen)6);
    }

L300:
    return 0;
} /* cosqmb_ */

