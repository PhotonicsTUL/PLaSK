/* r1f4kb.f -- translated by f2c (version 20100827).
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

/* Subroutine */ int r1f4kb_(integer *ido, integer *l1, doublereal *cc, 
	integer *in1, doublereal *ch, integer *in2, doublereal *wa1, 
	doublereal *wa2, doublereal *wa3)
{
    /* System generated locals */
    integer cc_dim1, cc_dim2, cc_offset, ch_dim1, ch_dim2, ch_dim3, ch_offset,
	     i__1, i__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, k, ic, idp2;
    doublereal sqrt2;


    /* Parameter adjustments */
    --wa3;
    --wa2;
    --wa1;
    cc_dim1 = *in1;
    cc_dim2 = *ido;
    cc_offset = 1 + cc_dim1 * (1 + cc_dim2 * 5);
    cc -= cc_offset;
    ch_dim1 = *in2;
    ch_dim2 = *ido;
    ch_dim3 = *l1;
    ch_offset = 1 + ch_dim1 * (1 + ch_dim2 * (1 + ch_dim3));
    ch -= ch_offset;

    /* Function Body */
    sqrt2 = sqrt(2.);
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ch[((k + ch_dim3 * 3) * ch_dim2 + 1) * ch_dim1 + 1] = cc[(((k << 2) + 
		1) * cc_dim2 + 1) * cc_dim1 + 1] + cc[(*ido + ((k << 2) + 4) *
		 cc_dim2) * cc_dim1 + 1] - (cc[(*ido + ((k << 2) + 2) * 
		cc_dim2) * cc_dim1 + 1] + cc[(*ido + ((k << 2) + 2) * cc_dim2)
		 * cc_dim1 + 1]);
	ch[((k + ch_dim3) * ch_dim2 + 1) * ch_dim1 + 1] = cc[(((k << 2) + 1) *
		 cc_dim2 + 1) * cc_dim1 + 1] + cc[(*ido + ((k << 2) + 4) * 
		cc_dim2) * cc_dim1 + 1] + (cc[(*ido + ((k << 2) + 2) * 
		cc_dim2) * cc_dim1 + 1] + cc[(*ido + ((k << 2) + 2) * cc_dim2)
		 * cc_dim1 + 1]);
	ch[((k + (ch_dim3 << 2)) * ch_dim2 + 1) * ch_dim1 + 1] = cc[(((k << 2)
		 + 1) * cc_dim2 + 1) * cc_dim1 + 1] - cc[(*ido + ((k << 2) + 
		4) * cc_dim2) * cc_dim1 + 1] + (cc[(((k << 2) + 3) * cc_dim2 
		+ 1) * cc_dim1 + 1] + cc[(((k << 2) + 3) * cc_dim2 + 1) * 
		cc_dim1 + 1]);
	ch[((k + (ch_dim3 << 1)) * ch_dim2 + 1) * ch_dim1 + 1] = cc[(((k << 2)
		 + 1) * cc_dim2 + 1) * cc_dim1 + 1] - cc[(*ido + ((k << 2) + 
		4) * cc_dim2) * cc_dim1 + 1] - (cc[(((k << 2) + 3) * cc_dim2 
		+ 1) * cc_dim1 + 1] + cc[(((k << 2) + 3) * cc_dim2 + 1) * 
		cc_dim1 + 1]);
/* L101: */
    }
    if ((i__1 = *ido - 2) < 0) {
	goto L107;
    } else if (i__1 == 0) {
	goto L105;
    } else {
	goto L102;
    }
L102:
    idp2 = *ido + 2;
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	i__2 = *ido;
	for (i__ = 3; i__ <= i__2; i__ += 2) {
	    ic = idp2 - i__;
	    ch[(i__ - 1 + (k + ch_dim3) * ch_dim2) * ch_dim1 + 1] = cc[(i__ - 
		    1 + ((k << 2) + 1) * cc_dim2) * cc_dim1 + 1] + cc[(ic - 1 
		    + ((k << 2) + 4) * cc_dim2) * cc_dim1 + 1] + (cc[(i__ - 1 
		    + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1] + cc[(ic - 1 + 
		    ((k << 2) + 2) * cc_dim2) * cc_dim1 + 1]);
	    ch[(i__ + (k + ch_dim3) * ch_dim2) * ch_dim1 + 1] = cc[(i__ + ((k 
		    << 2) + 1) * cc_dim2) * cc_dim1 + 1] - cc[(ic + ((k << 2) 
		    + 4) * cc_dim2) * cc_dim1 + 1] + (cc[(i__ + ((k << 2) + 3)
		     * cc_dim2) * cc_dim1 + 1] - cc[(ic + ((k << 2) + 2) * 
		    cc_dim2) * cc_dim1 + 1]);
	    ch[(i__ - 1 + (k + (ch_dim3 << 1)) * ch_dim2) * ch_dim1 + 1] = 
		    wa1[i__ - 2] * (cc[(i__ - 1 + ((k << 2) + 1) * cc_dim2) * 
		    cc_dim1 + 1] - cc[(ic - 1 + ((k << 2) + 4) * cc_dim2) * 
		    cc_dim1 + 1] - (cc[(i__ + ((k << 2) + 3) * cc_dim2) * 
		    cc_dim1 + 1] + cc[(ic + ((k << 2) + 2) * cc_dim2) * 
		    cc_dim1 + 1])) - wa1[i__ - 1] * (cc[(i__ + ((k << 2) + 1) 
		    * cc_dim2) * cc_dim1 + 1] + cc[(ic + ((k << 2) + 4) * 
		    cc_dim2) * cc_dim1 + 1] + (cc[(i__ - 1 + ((k << 2) + 3) * 
		    cc_dim2) * cc_dim1 + 1] - cc[(ic - 1 + ((k << 2) + 2) * 
		    cc_dim2) * cc_dim1 + 1]));
	    ch[(i__ + (k + (ch_dim3 << 1)) * ch_dim2) * ch_dim1 + 1] = wa1[
		    i__ - 2] * (cc[(i__ + ((k << 2) + 1) * cc_dim2) * cc_dim1 
		    + 1] + cc[(ic + ((k << 2) + 4) * cc_dim2) * cc_dim1 + 1] 
		    + (cc[(i__ - 1 + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1] 
		    - cc[(ic - 1 + ((k << 2) + 2) * cc_dim2) * cc_dim1 + 1])) 
		    + wa1[i__ - 1] * (cc[(i__ - 1 + ((k << 2) + 1) * cc_dim2) 
		    * cc_dim1 + 1] - cc[(ic - 1 + ((k << 2) + 4) * cc_dim2) * 
		    cc_dim1 + 1] - (cc[(i__ + ((k << 2) + 3) * cc_dim2) * 
		    cc_dim1 + 1] + cc[(ic + ((k << 2) + 2) * cc_dim2) * 
		    cc_dim1 + 1]));
	    ch[(i__ - 1 + (k + ch_dim3 * 3) * ch_dim2) * ch_dim1 + 1] = wa2[
		    i__ - 2] * (cc[(i__ - 1 + ((k << 2) + 1) * cc_dim2) * 
		    cc_dim1 + 1] + cc[(ic - 1 + ((k << 2) + 4) * cc_dim2) * 
		    cc_dim1 + 1] - (cc[(i__ - 1 + ((k << 2) + 3) * cc_dim2) * 
		    cc_dim1 + 1] + cc[(ic - 1 + ((k << 2) + 2) * cc_dim2) * 
		    cc_dim1 + 1])) - wa2[i__ - 1] * (cc[(i__ + ((k << 2) + 1) 
		    * cc_dim2) * cc_dim1 + 1] - cc[(ic + ((k << 2) + 4) * 
		    cc_dim2) * cc_dim1 + 1] - (cc[(i__ + ((k << 2) + 3) * 
		    cc_dim2) * cc_dim1 + 1] - cc[(ic + ((k << 2) + 2) * 
		    cc_dim2) * cc_dim1 + 1]));
	    ch[(i__ + (k + ch_dim3 * 3) * ch_dim2) * ch_dim1 + 1] = wa2[i__ - 
		    2] * (cc[(i__ + ((k << 2) + 1) * cc_dim2) * cc_dim1 + 1] 
		    - cc[(ic + ((k << 2) + 4) * cc_dim2) * cc_dim1 + 1] - (cc[
		    (i__ + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1] - cc[(ic 
		    + ((k << 2) + 2) * cc_dim2) * cc_dim1 + 1])) + wa2[i__ - 
		    1] * (cc[(i__ - 1 + ((k << 2) + 1) * cc_dim2) * cc_dim1 + 
		    1] + cc[(ic - 1 + ((k << 2) + 4) * cc_dim2) * cc_dim1 + 1]
		     - (cc[(i__ - 1 + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1]
		     + cc[(ic - 1 + ((k << 2) + 2) * cc_dim2) * cc_dim1 + 1]))
		    ;
	    ch[(i__ - 1 + (k + (ch_dim3 << 2)) * ch_dim2) * ch_dim1 + 1] = 
		    wa3[i__ - 2] * (cc[(i__ - 1 + ((k << 2) + 1) * cc_dim2) * 
		    cc_dim1 + 1] - cc[(ic - 1 + ((k << 2) + 4) * cc_dim2) * 
		    cc_dim1 + 1] + (cc[(i__ + ((k << 2) + 3) * cc_dim2) * 
		    cc_dim1 + 1] + cc[(ic + ((k << 2) + 2) * cc_dim2) * 
		    cc_dim1 + 1])) - wa3[i__ - 1] * (cc[(i__ + ((k << 2) + 1) 
		    * cc_dim2) * cc_dim1 + 1] + cc[(ic + ((k << 2) + 4) * 
		    cc_dim2) * cc_dim1 + 1] - (cc[(i__ - 1 + ((k << 2) + 3) * 
		    cc_dim2) * cc_dim1 + 1] - cc[(ic - 1 + ((k << 2) + 2) * 
		    cc_dim2) * cc_dim1 + 1]));
	    ch[(i__ + (k + (ch_dim3 << 2)) * ch_dim2) * ch_dim1 + 1] = wa3[
		    i__ - 2] * (cc[(i__ + ((k << 2) + 1) * cc_dim2) * cc_dim1 
		    + 1] + cc[(ic + ((k << 2) + 4) * cc_dim2) * cc_dim1 + 1] 
		    - (cc[(i__ - 1 + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1] 
		    - cc[(ic - 1 + ((k << 2) + 2) * cc_dim2) * cc_dim1 + 1])) 
		    + wa3[i__ - 1] * (cc[(i__ - 1 + ((k << 2) + 1) * cc_dim2) 
		    * cc_dim1 + 1] - cc[(ic - 1 + ((k << 2) + 4) * cc_dim2) * 
		    cc_dim1 + 1] + (cc[(i__ + ((k << 2) + 3) * cc_dim2) * 
		    cc_dim1 + 1] + cc[(ic + ((k << 2) + 2) * cc_dim2) * 
		    cc_dim1 + 1]));
/* L103: */
	}
/* L104: */
    }
    if (*ido % 2 == 1) {
	return 0;
    }
L105:
    i__1 = *l1;
    for (k = 1; k <= i__1; ++k) {
	ch[(*ido + (k + ch_dim3) * ch_dim2) * ch_dim1 + 1] = cc[(*ido + ((k <<
		 2) + 1) * cc_dim2) * cc_dim1 + 1] + cc[(*ido + ((k << 2) + 3)
		 * cc_dim2) * cc_dim1 + 1] + (cc[(*ido + ((k << 2) + 1) * 
		cc_dim2) * cc_dim1 + 1] + cc[(*ido + ((k << 2) + 3) * cc_dim2)
		 * cc_dim1 + 1]);
	ch[(*ido + (k + (ch_dim3 << 1)) * ch_dim2) * ch_dim1 + 1] = sqrt2 * (
		cc[(*ido + ((k << 2) + 1) * cc_dim2) * cc_dim1 + 1] - cc[(*
		ido + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1] - (cc[(((k << 
		2) + 2) * cc_dim2 + 1) * cc_dim1 + 1] + cc[(((k << 2) + 4) * 
		cc_dim2 + 1) * cc_dim1 + 1]));
	ch[(*ido + (k + ch_dim3 * 3) * ch_dim2) * ch_dim1 + 1] = cc[(((k << 2)
		 + 4) * cc_dim2 + 1) * cc_dim1 + 1] - cc[(((k << 2) + 2) * 
		cc_dim2 + 1) * cc_dim1 + 1] + (cc[(((k << 2) + 4) * cc_dim2 + 
		1) * cc_dim1 + 1] - cc[(((k << 2) + 2) * cc_dim2 + 1) * 
		cc_dim1 + 1]);
	ch[(*ido + (k + (ch_dim3 << 2)) * ch_dim2) * ch_dim1 + 1] = -sqrt2 * (
		cc[(*ido + ((k << 2) + 1) * cc_dim2) * cc_dim1 + 1] - cc[(*
		ido + ((k << 2) + 3) * cc_dim2) * cc_dim1 + 1] + (cc[(((k << 
		2) + 2) * cc_dim2 + 1) * cc_dim1 + 1] + cc[(((k << 2) + 4) * 
		cc_dim2 + 1) * cc_dim1 + 1]));
/* L106: */
    }
L107:
    return 0;
} /* r1f4kb_ */

