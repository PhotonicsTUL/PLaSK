/* costf1.f -- translated by f2c (version 20100827).
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

/* Subroutine */ int costf1_(integer *n, integer *inc, doublereal *x, 
	doublereal *wsave, doublereal *work, integer *ier)
{
    /* System generated locals */
    integer x_dim1, x_offset, i__1;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer i__, k;
    doublereal t1, t2;
    integer kc;
    doublereal xi;
    integer nm1, np1;
    doublereal x1h;
    integer ns2;
    doublereal tx2, x1p3;
    integer ier1;
    doublereal snm1;
    integer modn, lenx;
    doublereal dsum;
    integer lnwk, lnsv;
    extern /* Subroutine */ int rfft1f_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *, integer *, 
	    integer *), xerfft_(char *, integer *, ftnlen);

    /* Parameter adjustments */
    x_dim1 = *inc;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    --wsave;

    /* Function Body */
    *ier = 0;
    nm1 = *n - 1;
    np1 = *n + 1;
    ns2 = *n / 2;
    if ((i__1 = *n - 2) < 0) {
	goto L200;
    } else if (i__1 == 0) {
	goto L101;
    } else {
	goto L102;
    }
L101:
    x1h = x[x_dim1 + 1] + x[(x_dim1 << 1) + 1];
    x[(x_dim1 << 1) + 1] = (x[x_dim1 + 1] - x[(x_dim1 << 1) + 1]) * .5;
    x[x_dim1 + 1] = x1h * .5;
    goto L200;
L102:
    if (*n > 3) {
	goto L103;
    }
    x1p3 = x[x_dim1 + 1] + x[x_dim1 * 3 + 1];
    tx2 = x[(x_dim1 << 1) + 1] + x[(x_dim1 << 1) + 1];
    x[(x_dim1 << 1) + 1] = (x[x_dim1 + 1] - x[x_dim1 * 3 + 1]) * .5;
    x[x_dim1 + 1] = (x1p3 + tx2) * .25;
    x[x_dim1 * 3 + 1] = (x1p3 - tx2) * .25;
    goto L200;
L103:
    dsum = x[x_dim1 + 1] - x[*n * x_dim1 + 1];
    x[x_dim1 + 1] += x[*n * x_dim1 + 1];
    i__1 = ns2;
    for (k = 2; k <= i__1; ++k) {
	kc = np1 - k;
	t1 = x[k * x_dim1 + 1] + x[kc * x_dim1 + 1];
	t2 = x[k * x_dim1 + 1] - x[kc * x_dim1 + 1];
	dsum += wsave[kc] * t2;
	t2 = wsave[k] * t2;
	x[k * x_dim1 + 1] = t1 - t2;
	x[kc * x_dim1 + 1] = t1 + t2;
/* L104: */
    }
    modn = *n % 2;
    if (modn == 0) {
	goto L124;
    }
    x[(ns2 + 1) * x_dim1 + 1] += x[(ns2 + 1) * x_dim1 + 1];
L124:
    lenx = *inc * (nm1 - 1) + 1;
    lnsv = nm1 + (integer) (log((doublereal) nm1) / log(2.)) + 4;
    lnwk = nm1;

    rfft1f_(&nm1, inc, &x[x_offset], &lenx, &wsave[*n + 1], &lnsv, work, &
	    lnwk, &ier1);
    if (ier1 != 0) {
	*ier = 20;
	xerfft_("COSTF1", &c_n5, (ftnlen)6);
	goto L200;
    }

    snm1 = 1. / (doublereal) nm1;
    dsum = snm1 * dsum;
    if (nm1 % 2 != 0) {
	goto L30;
    }
    x[nm1 * x_dim1 + 1] += x[nm1 * x_dim1 + 1];
L30:
    i__1 = *n;
    for (i__ = 3; i__ <= i__1; i__ += 2) {
	xi = x[i__ * x_dim1 + 1] * .5;
	x[i__ * x_dim1 + 1] = x[(i__ - 1) * x_dim1 + 1] * .5;
	x[(i__ - 1) * x_dim1 + 1] = dsum;
	dsum += xi;
/* L105: */
    }
    if (modn != 0) {
	goto L117;
    }
    x[*n * x_dim1 + 1] = dsum;
L117:
    x[x_dim1 + 1] *= .5;
    x[*n * x_dim1 + 1] *= .5;
L200:
    return 0;
} /* costf1_ */

