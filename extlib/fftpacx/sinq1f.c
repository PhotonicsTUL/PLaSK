/* sinq1f.f -- translated by f2c (version 20100827).
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

/* Subroutine */ int sinq1f_(integer *n, integer *inc, doublereal *x, integer 
	*lenx, doublereal *wsave, integer *lensav, doublereal *work, integer *
	lenwrk, integer *ier)
{
    /* System generated locals */
    integer x_dim1, x_offset, i__1;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer k, kc, ns2, ier1;
    doublereal xhold;
    extern /* Subroutine */ int cosq1f_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *, integer *, 
	    integer *), xerfft_(char *, integer *, ftnlen);


    /* Parameter adjustments */
    x_dim1 = *inc;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --wsave;
    --work;

    /* Function Body */
    *ier = 0;

    if (*lenx < *inc * (*n - 1) + 1) {
	*ier = 1;
	xerfft_("SINQ1F", &c__6, (ftnlen)6);
	goto L300;
    } else if (*lensav < (*n << 1) + (integer) (log((doublereal) (*n)) / log(
	    2.)) + 4) {
	*ier = 2;
	xerfft_("SINQ1F", &c__8, (ftnlen)6);
	goto L300;
    } else if (*lenwrk < *n) {
	*ier = 3;
	xerfft_("SINQ1F", &c__10, (ftnlen)6);
	goto L300;
    }

    if (*n == 1) {
	return 0;
    }
    ns2 = *n / 2;
    i__1 = ns2;
    for (k = 1; k <= i__1; ++k) {
	kc = *n - k;
	xhold = x[k * x_dim1 + 1];
	x[k * x_dim1 + 1] = x[(kc + 1) * x_dim1 + 1];
	x[(kc + 1) * x_dim1 + 1] = xhold;
/* L101: */
    }
    cosq1f_(n, inc, &x[x_offset], lenx, &wsave[1], lensav, &work[1], lenwrk, &
	    ier1);
    if (ier1 != 0) {
	*ier = 20;
	xerfft_("SINQ1F", &c_n5, (ftnlen)6);
	goto L300;
    }
    i__1 = *n;
    for (k = 2; k <= i__1; k += 2) {
	x[k * x_dim1 + 1] = -x[k * x_dim1 + 1];
/* L102: */
    }
L300:
    return 0;
} /* sinq1f_ */

