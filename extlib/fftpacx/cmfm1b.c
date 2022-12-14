/* cmfm1b.f -- translated by f2c (version 20100827).
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

static integer c__1 = 1;

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

/* Subroutine */ int cmfm1b_(integer *lot, integer *jump, integer *n, integer 
	*inc, doublecomplex *c__, doublereal *ch, doublereal *wa, doublereal *
	fnf, doublereal *fac)
{
    /* System generated locals */
    integer i__1, i__2;

    /* Local variables */
    integer k1, l1, l2, na, nf, ip, iw, lid, ido, nbr;
    extern /* Subroutine */ int cmf2kb_(), cmf3kb_(), cmf4kb_(), cmf5kb_(), 
	    cmfgkb_();


/* FFTPACK 5.0 auxiliary routine */

    /* Parameter adjustments */
    --fac;
    --wa;
    --ch;
    --c__;

    /* Function Body */
    nf = (integer) (*fnf);
    na = 0;
    l1 = 1;
    iw = 1;
    i__1 = nf;
    for (k1 = 1; k1 <= i__1; ++k1) {
	ip = (integer) fac[k1];
	l2 = ip * l1;
	ido = *n / l2;
	lid = l1 * ido;
/* Computing MIN */
	i__2 = ip - 2;
	nbr = na + 1 + (min(i__2,4) << 1);
	switch (nbr) {
	    case 1:  goto L52;
	    case 2:  goto L62;
	    case 3:  goto L53;
	    case 4:  goto L63;
	    case 5:  goto L54;
	    case 6:  goto L64;
	    case 7:  goto L55;
	    case 8:  goto L65;
	    case 9:  goto L56;
	    case 10:  goto L66;
	}
L52:
	cmf2kb_(lot, &ido, &l1, &na, &c__[1], jump, inc, &ch[1], &c__1, lot, &
		wa[iw]);
	goto L120;
L62:
	cmf2kb_(lot, &ido, &l1, &na, &ch[1], &c__1, lot, &c__[1], jump, inc, &
		wa[iw]);
	goto L120;
L53:
	cmf3kb_(lot, &ido, &l1, &na, &c__[1], jump, inc, &ch[1], &c__1, lot, &
		wa[iw]);
	goto L120;
L63:
	cmf3kb_(lot, &ido, &l1, &na, &ch[1], &c__1, lot, &c__[1], jump, inc, &
		wa[iw]);
	goto L120;
L54:
	cmf4kb_(lot, &ido, &l1, &na, &c__[1], jump, inc, &ch[1], &c__1, lot, &
		wa[iw]);
	goto L120;
L64:
	cmf4kb_(lot, &ido, &l1, &na, &ch[1], &c__1, lot, &c__[1], jump, inc, &
		wa[iw]);
	goto L120;
L55:
	cmf5kb_(lot, &ido, &l1, &na, &c__[1], jump, inc, &ch[1], &c__1, lot, &
		wa[iw]);
	goto L120;
L65:
	cmf5kb_(lot, &ido, &l1, &na, &ch[1], &c__1, lot, &c__[1], jump, inc, &
		wa[iw]);
	goto L120;
L56:
	cmfgkb_(lot, &ido, &ip, &l1, &lid, &na, &c__[1], &c__[1], jump, inc, &
		ch[1], &ch[1], &c__1, lot, &wa[iw]);
	goto L120;
L66:
	cmfgkb_(lot, &ido, &ip, &l1, &lid, &na, &ch[1], &ch[1], &c__1, lot, &
		c__[1], &c__[1], jump, inc, &wa[iw]);
L120:
	l1 = l2;
	iw += (ip - 1) * (ido + ido);
	if (ip <= 5) {
	    na = 1 - na;
	}
/* L125: */
    }
    return 0;
} /* cmfm1b_ */

